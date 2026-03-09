//! Multi-stage async ingestion pipeline.
//!
//! Three tasks run concurrently inside a dedicated OS thread with a
//! single-thread tokio runtime + `LocalSet`. This sidesteps the `Send`
//! requirement imposed by `tokio::spawn`, which is incompatible with
//! `libsql::Connection` (which is `Send` but not `Sync`).
//!
//! ```text
//! QueuePoller  ──(ChunkedItem)──►  EmbedderTask  ──(EmbedRequest)──►  DbWriter
//!   rayon par_map                   async API calls                    single writer
//! ```
//!
//! The pipeline communicates with the main runtime exclusively through the
//! persistent SQLite queue: `insert()` enqueues, the QueuePoller dequeues.
//! `Arc<Embedder>` is the only cross-thread shared state (it is `Send + Sync`).

use crate::embedder::Embedder;
use crate::error::Result;
use crate::executor::Executor;
use crate::queue::{Queue, QueueConnection};
use crate::vectordb::VectorDatabase;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc};
use tokio::time::Duration;

const POLL_INTERVAL: Duration = Duration::from_millis(200);
const BATCH_SIZE: usize = 4;
const CHUNK_CHANNEL_CAP: usize = 8;
const EMBED_CHANNEL_CAP: usize = 16;

// ── Message types ────────────────────────────────────────────────────────────

struct ChunkedItem {
    queue_id: i64,
    metadata: Value,
    text: String,
    /// `("search_document: <chunk>", byte_start, byte_end)` tuples.
    chunks: Vec<(String, usize, usize)>,
}

struct EmbedRequest {
    queue_id: i64,
    metadata: Value,
    text: String,
    embeddings: Vec<(Vec<f32>, usize, usize)>,
}

// ── Pipeline ─────────────────────────────────────────────────────────────────

/// Owns the pipeline thread and the shutdown signal.
///
/// Dropping `Pipeline` sends the shutdown signal; the tasks stop at their next
/// natural checkpoint and drain any in-flight items.
pub(crate) struct Pipeline {
    shutdown_tx: broadcast::Sender<()>,
    /// Kept alive so the OS thread is joined on drop (if needed).
    _thread: std::thread::JoinHandle<()>,
}

impl Pipeline {
    /// Starts the pipeline in a dedicated background thread and waits until
    /// the thread has opened all database connections and is ready to accept work.
    ///
    /// # Arguments
    ///
    /// * `queue_db_path` - Path to the queue SQLite database.
    /// * `vdb_path`      - Path to the vector SQLite database (write connection).
    /// * `vec_dim`       - Embedding vector dimension (must match the existing DB).
    /// * `model`         - Model name for metadata validation in the vector DB.
    /// * `embedder`      - Shared embedder (`Arc` so it is `Send`).
    /// * `executor`      - Controls rayon parallelism inside the QueuePoller.
    pub(crate) async fn start(
        queue_db_path: String,
        vdb_path: String,
        vec_dim: usize,
        model: Option<String>,
        embedder: Arc<Embedder>,
        executor: Executor,
    ) -> Result<Self> {
        // One-shot channel: pipeline thread signals readiness (or startup error).
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<()>>();

        let (shutdown_tx, _) = broadcast::channel::<()>(1);
        let shutdown_rx = shutdown_tx.subscribe();

        let thread = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("pipeline: failed to build runtime");

            let local = tokio::task::LocalSet::new();

            local.block_on(&rt, async move {
                // Open all database connections inside this thread's runtime.
                let setup: Result<_> = async {
                    let queue = Queue::new(&queue_db_path).await?;
                    let poller_qconn = queue.connect().await?;
                    let embed_qconn = queue.connect().await?;
                    let writer_qconn = queue.connect().await?;

                    // The pipeline opens its own VectorDatabase handle (write only).
                    // SQLite WAL allows the main runtime to read concurrently.
                    let vdb = VectorDatabase::new(&vdb_path, vec_dim, executor).await?;
                    let write_conn = vdb.connect_with_model(model.as_deref()).await?;

                    Ok((poller_qconn, embed_qconn, writer_qconn, write_conn))
                }
                .await;

                match setup {
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                    }
                    Ok((poller_qconn, embed_qconn, writer_qconn, write_conn)) => {
                        let _ = ready_tx.send(Ok(()));

                        let (chunk_tx, chunk_rx) = mpsc::channel(CHUNK_CHANNEL_CAP);
                        let (embed_tx, embed_rx) = mpsc::channel(EMBED_CHANNEL_CAP);

                        let h1 = tokio::task::spawn_local(queue_poller_task(
                            poller_qconn,
                            embedder.clone(),
                            executor,
                            chunk_tx,
                            shutdown_rx,
                        ));
                        let h2 = tokio::task::spawn_local(embedder_task(
                            embedder,
                            embed_qconn,
                            chunk_rx,
                            embed_tx,
                        ));
                        let h3 = tokio::task::spawn_local(db_writer_task(
                            write_conn,
                            writer_qconn,
                            embed_rx,
                        ));

                        // Wait for all three tasks to finish (they stop when
                        // the shutdown signal cascades through the channels).
                        let (_, _, _) = tokio::join!(h1, h2, h3);
                    }
                }
            });
        });

        // Block (on a spawn_blocking thread) until the pipeline signals readiness.
        let ready = tokio::task::spawn_blocking(move || {
            ready_rx
                .recv()
                .expect("pipeline thread exited before signalling readiness")
        })
        .await
        .expect("spawn_blocking panicked");

        ready?;

        Ok(Pipeline {
            shutdown_tx,
            _thread: thread,
        })
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        // Signal the QueuePoller to stop; shutdown cascades through the channels.
        let _ = self.shutdown_tx.send(());
    }
}

// ── Stage 1: QueuePoller ─────────────────────────────────────────────────────

async fn queue_poller_task(
    queue: QueueConnection,
    embedder: Arc<Embedder>,
    executor: Executor,
    tx: mpsc::Sender<ChunkedItem>,
    mut shutdown: broadcast::Receiver<()>,
) {
    loop {
        tokio::select! {
            biased;
            _ = shutdown.recv() => break,
            _ = tokio::time::sleep(POLL_INTERVAL) => {}
        }

        let items = match queue.dequeue_batch(BATCH_SIZE).await {
            Ok(items) if items.is_empty() => continue,
            Ok(items) => items,
            Err(e) => {
                tracing::error!("queue poll error: {}", e);
                continue;
            }
        };

        // Chunk all items in the batch in parallel (CPU-bound, rayon).
        let chunked = executor.par_map(items, |item| {
            let chunks = embedder.chunk_document(&item.text);
            ChunkedItem {
                queue_id: item.id,
                metadata: item.metadata,
                text: item.text,
                chunks,
            }
        });

        for item in chunked {
            if tx.send(item).await.is_err() {
                return; // downstream closed
            }
        }
    }
    // Dropping `tx` signals the embedder that no more items are coming.
}

// ── Stage 2: EmbedderTask ────────────────────────────────────────────────────

async fn embedder_task(
    embedder: Arc<Embedder>,
    queue: QueueConnection,
    mut rx: mpsc::Receiver<ChunkedItem>,
    tx: mpsc::Sender<EmbedRequest>,
) {
    while let Some(item) = rx.recv().await {
        if item.chunks.is_empty() {
            let _ = queue
                .mark_failed(item.queue_id, "document produced no embeddable chunks")
                .await;
            continue;
        }

        match embedder.embed_chunks(&item.chunks).await {
            Ok(embeddings) => {
                let req = EmbedRequest {
                    queue_id: item.queue_id,
                    metadata: item.metadata,
                    text: item.text,
                    embeddings,
                };
                if tx.send(req).await.is_err() {
                    return;
                }
            }
            Err(e) => {
                tracing::error!("embedding failed for queue_id={}: {}", item.queue_id, e);
                let _ = queue.mark_failed(item.queue_id, &e.to_string()).await;
            }
        }
    }
    // Dropping `tx` signals the writer that no more items are coming.
}

// ── Stage 3: DbWriter ────────────────────────────────────────────────────────

async fn db_writer_task(
    vconn: crate::vectordb::VectorDatabaseConnection,
    queue: QueueConnection,
    mut rx: mpsc::Receiver<EmbedRequest>,
) {
    while let Some(req) = rx.recv().await {
        match vconn
            .insert_document(&req.text, req.metadata, req.embeddings)
            .await
        {
            Ok(_) => {
                if let Err(e) = queue.mark_done(req.queue_id).await {
                    tracing::error!("mark_done failed for queue_id={}: {}", req.queue_id, e);
                }
            }
            Err(e) => {
                tracing::error!("db insert failed for queue_id={}: {}", req.queue_id, e);
                let _ = queue.mark_failed(req.queue_id, &e.to_string()).await;
            }
        }
    }
}
