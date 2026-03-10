//! Multi-stage async ingestion pipeline, fully DB-backed.
//!
//! Three dedicated OS threads poll their respective queue DB tables, each with
//! its own tokio single-thread runtime and its own DB connection.  No data is
//! held in memory across stage boundaries — every handoff is an atomic DB
//! transaction (INSERT into next table + DELETE/UPDATE from current).
//!
//! ```text
//! capsa_queue  ──►  pending_embeds  ──►  pending_writes  ──►  vector DB
//!  ChunkTask        EmbedTask              WriteTask
//! (CPU: chunk)  (net: embed +          (CPU: JSON ser +
//!               CPU: compress,          DB insert)
//!               parallel)
//! ```
//!
//! Crash safety: on startup `reset_pipeline()` clears the two intermediate
//! tables and resets 'processing' items to 'pending'. Each stage atomically
//! moves data forward so a crash mid-processing leaves the item in its
//! *input* table, ready to be retried.
//!
//! Scheduling: each thread drains its input table completely before waiting.
//! It waits on a `tokio::sync::Notify` only when the table is empty; a
//! `notify_one()` call from the upstream stage wakes it immediately.

use crate::embedder::Embedder;
use crate::error::Result;
use crate::executor::Executor;
use crate::queue::Queue;
use crate::vectordb::VectorDatabase;
use std::sync::Arc;
use tokio::sync::{Notify, broadcast};

/// Number of documents chunked concurrently in the chunk stage.
const CHUNK_CONCURRENCY: usize = 4;

// ── Pipeline ─────────────────────────────────────────────────────────────────

pub(crate) struct Pipeline {
    shutdown_tx: broadcast::Sender<()>,
    /// Notify handle: call `notify_one()` whenever a new item is enqueued to
    /// `capsa_queue` so the chunk thread wakes up immediately.
    pub(crate) queue_ready: Arc<Notify>,
    _threads: Vec<std::thread::JoinHandle<()>>,
}

impl Pipeline {
    pub(crate) async fn start(
        queue_db_path: String,
        vdb_path: String,
        vec_dim: usize,
        model: Option<String>,
        embedder: Arc<Embedder>,
        executor: Executor,
    ) -> Result<Self> {
        let (shutdown_tx, _) = broadcast::channel::<()>(1);
        let queue_ready = Arc::new(Notify::new());
        let embeds_ready = Arc::new(Notify::new());
        let writes_ready = Arc::new(Notify::new());

        // ── chunk thread ─────────────────────────────────────────────────────
        let (chunk_ready_tx, chunk_ready_rx) = std::sync::mpsc::channel::<Result<()>>();
        let chunk_thread = {
            let path = queue_db_path.clone();
            let embedder = embedder.clone();
            let queue_ready = queue_ready.clone();
            let embeds_ready = embeds_ready.clone();
            let shutdown_rx = shutdown_tx.subscribe();
            std::thread::spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("chunk: failed to build runtime");
                rt.block_on(async move {
                    let setup: Result<_> = async {
                        let queue = Queue::new(&path).await?;
                        let conn = queue.connect().await?;
                        Ok(conn)
                    }
                    .await;
                    match setup {
                        Err(e) => {
                            let _ = chunk_ready_tx.send(Err(e));
                        }
                        Ok(conn) => {
                            let _ = chunk_ready_tx.send(Ok(()));
                            chunk_task(conn, embedder, queue_ready, embeds_ready, shutdown_rx)
                                .await;
                        }
                    }
                });
            })
        };

        // ── embed thread ─────────────────────────────────────────────────────
        let (embed_ready_tx, embed_ready_rx) = std::sync::mpsc::channel::<Result<()>>();
        let embed_thread = {
            let path = queue_db_path.clone();
            let embedder = embedder.clone();
            let embeds_ready = embeds_ready.clone();
            let writes_ready = writes_ready.clone();
            let shutdown_rx = shutdown_tx.subscribe();
            std::thread::spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("embed: failed to build runtime");
                rt.block_on(async move {
                    let setup: Result<_> = async {
                        let queue = Queue::new(&path).await?;
                        let conn = queue.connect().await?;
                        Ok(conn)
                    }
                    .await;
                    match setup {
                        Err(e) => {
                            let _ = embed_ready_tx.send(Err(e));
                        }
                        Ok(conn) => {
                            let _ = embed_ready_tx.send(Ok(()));
                            embed_task(
                                conn,
                                embedder,
                                executor,
                                embeds_ready,
                                writes_ready,
                                shutdown_rx,
                            )
                            .await;
                        }
                    }
                });
            })
        };

        // ── write thread ─────────────────────────────────────────────────────
        let (write_ready_tx, write_ready_rx) = std::sync::mpsc::channel::<Result<()>>();
        let write_thread = {
            let path = queue_db_path.clone();
            let shutdown_rx = shutdown_tx.subscribe();
            std::thread::spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("write: failed to build runtime");
                rt.block_on(async move {
                    let setup: Result<_> = async {
                        let queue = Queue::new(&path).await?;
                        let conn = queue.connect().await?;
                        let vdb = VectorDatabase::new(&vdb_path, vec_dim, executor).await?;
                        let vconn = vdb.connect_with_model(model.as_deref()).await?;
                        Ok((conn, vconn))
                    }
                    .await;
                    match setup {
                        Err(e) => {
                            let _ = write_ready_tx.send(Err(e));
                        }
                        Ok((conn, vconn)) => {
                            let _ = write_ready_tx.send(Ok(()));
                            write_task(conn, vconn, executor, writes_ready, shutdown_rx).await;
                        }
                    }
                });
            })
        };

        // Wait for all three threads to finish setup (or fail).
        let chunk_ready = tokio::task::spawn_blocking(move || {
            chunk_ready_rx
                .recv()
                .expect("chunk thread exited before signalling readiness")
        })
        .await
        .expect("spawn_blocking panicked");

        let embed_ready = tokio::task::spawn_blocking(move || {
            embed_ready_rx
                .recv()
                .expect("embed thread exited before signalling readiness")
        })
        .await
        .expect("spawn_blocking panicked");

        let write_ready = tokio::task::spawn_blocking(move || {
            write_ready_rx
                .recv()
                .expect("write thread exited before signalling readiness")
        })
        .await
        .expect("spawn_blocking panicked");

        chunk_ready?;
        embed_ready?;
        write_ready?;

        Ok(Pipeline {
            shutdown_tx,
            queue_ready,
            _threads: vec![chunk_thread, embed_thread, write_thread],
        })
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        let _ = self.shutdown_tx.send(());
    }
}

// ── Binary embedding format ───────────────────────────────────────────────────
// Layout: [n_chunks: u32 LE] then for each chunk:
//   [chunk_start: u64 LE][chunk_end: u64 LE][n_floats: u32 LE][f32 * n_floats]

fn pack_embeddings(embeddings: &[(Vec<f32>, usize, usize)]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(embeddings.len() as u32).to_le_bytes());
    for (vec, start, end) in embeddings {
        buf.extend_from_slice(&(*start as u64).to_le_bytes());
        buf.extend_from_slice(&(*end as u64).to_le_bytes());
        buf.extend_from_slice(&(vec.len() as u32).to_le_bytes());
        for &f in vec {
            buf.extend_from_slice(&f.to_le_bytes());
        }
    }
    buf
}

fn unpack_embeddings(data: &[u8]) -> Result<Vec<(Vec<f32>, usize, usize)>> {
    use crate::error::{CapsaError, ProcessingError};
    let bad = || {
        CapsaError::Processing(ProcessingError::MissingData(
            "malformed embeddings blob".into(),
        ))
    };

    if data.len() < 4 {
        return Err(bad());
    }
    let n_chunks = u32::from_le_bytes(data[0..4].try_into().map_err(|_| bad())?) as usize;
    let mut cursor = 4;
    let mut result = Vec::with_capacity(n_chunks);

    for _ in 0..n_chunks {
        if cursor + 20 > data.len() {
            return Err(bad());
        }
        let start =
            u64::from_le_bytes(data[cursor..cursor + 8].try_into().map_err(|_| bad())?) as usize;
        cursor += 8;
        let end =
            u64::from_le_bytes(data[cursor..cursor + 8].try_into().map_err(|_| bad())?) as usize;
        cursor += 8;
        let n =
            u32::from_le_bytes(data[cursor..cursor + 4].try_into().map_err(|_| bad())?) as usize;
        cursor += 4;
        if cursor + n * 4 > data.len() {
            return Err(bad());
        }
        let mut vec = Vec::with_capacity(n);
        for _ in 0..n {
            let f = f32::from_le_bytes(data[cursor..cursor + 4].try_into().map_err(|_| bad())?);
            cursor += 4;
            vec.push(f);
        }
        result.push((vec, start, end));
    }

    Ok(result)
}

// ── Stage 1: ChunkTask ────────────────────────────────────────────────────────

async fn chunk_task(
    queue: crate::queue::QueueConnection,
    embedder: Arc<Embedder>,
    queue_ready: Arc<Notify>,
    embeds_ready: Arc<Notify>,
    mut shutdown: broadcast::Receiver<()>,
) {
    loop {
        // Drain all pending items before waiting, processing up to
        // CHUNK_CONCURRENCY documents concurrently per iteration.
        loop {
            let items = match queue.peek_pending_batch(CHUNK_CONCURRENCY).await {
                Ok(items) if items.is_empty() => break,
                Ok(items) => items,
                Err(e) => {
                    tracing::error!("chunk_task: peek error: {}", e);
                    break;
                }
            };

            // Chunk all documents concurrently on blocking threads; the runtime
            // thread remains free (not blocked) while the work runs.
            let handles: Vec<_> = items
                .into_iter()
                .map(|item| {
                    let embedder = embedder.clone();
                    tokio::task::spawn_blocking(move || {
                        let chunks = embedder.chunk_document(&item.text);
                        (item, chunks)
                    })
                })
                .collect();

            let results: Vec<_> = futures::future::join_all(handles)
                .await
                .into_iter()
                .map(|r| r.expect("chunk_document panicked"))
                .collect();

            // Promote or fail each result (sequential DB ops).
            for (item, chunks) in results {
                if chunks.is_empty() {
                    if let Err(e) = queue
                        .mark_failed(item.id, "document produced no embeddable chunks")
                        .await
                    {
                        tracing::error!(
                            "chunk_task: mark_failed error for queue_id={}: {}",
                            item.id,
                            e
                        );
                    }
                    continue;
                }
                match queue
                    .promote_to_embeds(item.id, &item.source, &item.metadata, &item.text, &chunks)
                    .await
                {
                    Ok(()) => embeds_ready.notify_one(),
                    Err(e) => tracing::error!(
                        "chunk_task: promote_to_embeds failed for queue_id={}: {}",
                        item.id,
                        e
                    ),
                }
            }
        }

        tokio::select! {
            biased;
            _ = shutdown.recv() => break,
            _ = queue_ready.notified() => {}
        }
    }
}

// ── Stage 2: EmbedTask ────────────────────────────────────────────────────────

async fn embed_task(
    queue: crate::queue::QueueConnection,
    embedder: Arc<Embedder>,
    executor: Executor,
    embeds_ready: Arc<Notify>,
    writes_ready: Arc<Notify>,
    mut shutdown: broadcast::Receiver<()>,
) {
    loop {
        // Drain all pending embeds before waiting.
        loop {
            let item = match queue.peek_embed().await {
                Ok(Some(item)) => item,
                Ok(None) => break,
                Err(e) => {
                    tracing::error!("embed_task: peek error: {}", e);
                    break;
                }
            };

            let embed_id = item.id;
            let queue_id = item.queue_id;
            let source = item.source;
            let metadata = item.metadata;
            let chunks = item.chunks;
            let text_bytes = item.text.into_bytes();

            // Embed chunks and compress text in parallel.
            let (embed_result, compress_result) = tokio::join!(
                embedder.embed_chunks(&chunks),
                executor.spawn_blocking(move || -> Result<Vec<u8>> {
                    use std::io::Write;
                    let mut enc = lz4_flex::frame::FrameEncoder::new(Vec::new());
                    enc.write_all(&text_bytes)?;
                    enc.finish().map_err(|e| std::io::Error::other(e).into())
                })
            );

            let embeddings = match embed_result {
                Ok(e) => e,
                Err(e) => {
                    tracing::error!(
                        "embed_task: embedding failed for queue_id={}: {}",
                        queue_id,
                        e
                    );
                    let _ = queue.fail_embed(embed_id, queue_id, &e.to_string()).await;
                    continue;
                }
            };
            let compressed_text = match compress_result {
                Ok(c) => c,
                Err(e) => {
                    tracing::error!(
                        "embed_task: compression failed for queue_id={}: {}",
                        queue_id,
                        e
                    );
                    let _ = queue.fail_embed(embed_id, queue_id, &e.to_string()).await;
                    continue;
                }
            };

            let embeddings_blob = pack_embeddings(&embeddings);

            match queue
                .promote_to_writes(
                    embed_id,
                    queue_id,
                    &source,
                    &metadata,
                    compressed_text,
                    embeddings_blob,
                )
                .await
            {
                Ok(()) => writes_ready.notify_one(),
                Err(e) => tracing::error!(
                    "embed_task: promote_to_writes failed for queue_id={}: {}",
                    queue_id,
                    e
                ),
            }
        }

        tokio::select! {
            biased;
            _ = shutdown.recv() => break,
            _ = embeds_ready.notified() => {}
        }
    }
}

// ── Stage 3: WriteTask ────────────────────────────────────────────────────────

async fn write_task(
    queue: crate::queue::QueueConnection,
    vconn: crate::vectordb::VectorDatabaseConnection,
    executor: Executor,
    writes_ready: Arc<Notify>,
    mut shutdown: broadcast::Receiver<()>,
) {
    loop {
        // Drain all pending writes before waiting.
        loop {
            let item = match queue.peek_write().await {
                Ok(Some(item)) => item,
                Ok(None) => break,
                Err(e) => {
                    tracing::error!("write_task: peek error: {}", e);
                    break;
                }
            };

            let write_id = item.id;
            let queue_id = item.queue_id;

            let embeddings = match unpack_embeddings(&item.embeddings_blob) {
                Ok(e) => e,
                Err(e) => {
                    tracing::error!("write_task: unpack failed for queue_id={}: {}", queue_id, e);
                    let _ = queue.fail_write(write_id, queue_id, &e.to_string()).await;
                    continue;
                }
            };

            // Serialize vectors to JSON in parallel before the DB transaction.
            let indexed: Vec<(usize, Vec<f32>, usize, usize)> = embeddings
                .into_iter()
                .enumerate()
                .map(|(i, (v, s, e))| (i, v, s, e))
                .collect();
            let serialized: crate::error::Result<Vec<_>> = executor
                .par_map(indexed, |(i, vec, s, e)| {
                    serde_json::to_string(&vec)
                        .map(|json| (i, json, s, e))
                        .map_err(crate::error::CapsaError::from)
                })
                .into_iter()
                .collect();
            let serialized = match serialized {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!(
                        "write_task: serialization failed for queue_id={}: {}",
                        queue_id,
                        e
                    );
                    let _ = queue.fail_write(write_id, queue_id, &e.to_string()).await;
                    continue;
                }
            };

            match vconn
                .insert_document_pre_compressed(item.compressed_text, item.metadata, serialized)
                .await
            {
                Ok(_) => {
                    if let Err(e) = queue.complete_write(write_id, queue_id).await {
                        tracing::error!(
                            "write_task: complete_write failed for queue_id={}: {}",
                            queue_id,
                            e
                        );
                    }
                }
                Err(e) => {
                    tracing::error!(
                        "write_task: db insert failed for queue_id={}: {}",
                        queue_id,
                        e
                    );
                    let _ = queue.fail_write(write_id, queue_id, &e.to_string()).await;
                }
            }
        }

        tokio::select! {
            biased;
            _ = shutdown.recv() => break,
            _ = writes_ready.notified() => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_empty() {
        let embeddings: Vec<(Vec<f32>, usize, usize)> = vec![];
        let blob = pack_embeddings(&embeddings);
        let result = unpack_embeddings(&blob).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_pack_unpack_single_chunk() {
        let embeddings = vec![(vec![1.0f32, 2.0, 3.0], 0usize, 10usize)];
        let blob = pack_embeddings(&embeddings);
        let result = unpack_embeddings(&blob).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, vec![1.0f32, 2.0, 3.0]);
        assert_eq!(result[0].1, 0);
        assert_eq!(result[0].2, 10);
    }

    #[test]
    fn test_pack_unpack_multiple_chunks() {
        let embeddings = vec![
            (vec![0.1f32, 0.2, 0.3], 0usize, 50usize),
            (vec![0.4f32, 0.5, 0.6], 45usize, 100usize),
            (vec![0.7f32, 0.8, 0.9], 95usize, 150usize),
        ];
        let blob = pack_embeddings(&embeddings);
        let result = unpack_embeddings(&blob).unwrap();
        assert_eq!(result.len(), 3);
        for (i, (vec, start, end)) in result.iter().enumerate() {
            assert_eq!(vec.len(), 3);
            assert!((vec[0] - embeddings[i].0[0]).abs() < 1e-6);
            assert_eq!(*start, embeddings[i].1);
            assert_eq!(*end, embeddings[i].2);
        }
    }

    #[test]
    fn test_pack_unpack_large_vectors() {
        let vec: Vec<f32> = (0..384).map(|i| i as f32 * 0.001).collect();
        let embeddings = vec![(vec.clone(), 0usize, 1000usize)];
        let blob = pack_embeddings(&embeddings);
        let result = unpack_embeddings(&blob).unwrap();
        assert_eq!(result[0].0.len(), 384);
        for (a, b) in result[0].0.iter().zip(vec.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_unpack_malformed_data() {
        assert!(unpack_embeddings(&[]).is_err());
        assert!(unpack_embeddings(&[1, 0, 0, 0]).is_err()); // claims 1 chunk but no data
    }
}
