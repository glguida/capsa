//! Persistent ingestion queue backed by a separate SQLite database.
//!
//! The queue database lives alongside the main database with a `-queue` suffix
//! (e.g. `capsa.db` → `capsa-queue.db`). Items flow through states:
//! `pending` → `processing` → `done` | `failed`.

use crate::error::Result;
use libsql::{Builder, Connection, Database, TransactionBehavior};
use serde_json::Value;
use std::path::Path;

/// Derives the queue database path from the main database path.
///
/// # Examples
/// - `"capsa.db"` → `"capsa-queue.db"`
/// - `"/data/mydb"` → `"/data/mydb-queue"`
pub fn queue_db_path(db_path: &str) -> String {
    let path = Path::new(db_path);
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    let ext = path
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy()))
        .unwrap_or_default();
    match path.parent() {
        Some(parent) if parent != Path::new("") => parent
            .join(format!("{}-queue{}", stem, ext))
            .to_string_lossy()
            .into_owned(),
        _ => format!("{}-queue{}", stem, ext),
    }
}

/// A document waiting to be ingested, as read from the queue.
pub struct QueueItem {
    pub id: i64,
    pub source: String,
    pub metadata: Value,
    pub text: String,
}

/// Pipeline status counts by stage.
#[derive(Debug, Clone, Default)]
pub struct PipelineStatus {
    /// Waiting to be chunked (in capsa_queue, status='pending').
    pub pending: u64,
    /// Chunked, waiting to be embedded (in pending_embeds).
    pub chunked: u64,
    /// Embedded and compressed, waiting to be written to the vector DB (in pending_writes).
    pub embedded: u64,
    /// Successfully written to the vector database.
    pub done: u64,
    /// Failed at some stage.
    pub failed: u64,
}

/// A failed ingestion item with diagnostic information.
#[derive(Debug, Clone)]
pub struct FailedItem {
    pub job_id: i64,
    pub display_name: String,
    pub source: String,
    pub error: String,
    pub created_at: i64,
}

/// A brief summary of an item in an intermediate pipeline stage (for verbose status).
#[derive(Debug, Clone)]
pub struct StageItem {
    /// Human-readable name derived from metadata when available.
    pub display_name: String,
    /// Size in bytes of the stage's primary payload.
    pub size: usize,
    /// Secondary payload size in bytes when useful for diagnostics.
    pub aux_size: Option<usize>,
    /// Number of chunks associated with the item, if known.
    pub chunk_count: Option<usize>,
}

/// An item awaiting embedding, sitting in the pending_embeds table.
pub struct PendingEmbed {
    pub id: i64,
    pub queue_id: i64,
    pub source: String,
    pub metadata: Value,
    pub text: String,
    pub chunks: Vec<(String, usize, usize)>,
}

/// An item awaiting a vector DB write, sitting in the pending_writes table.
pub struct PendingWrite {
    pub id: i64,
    pub queue_id: i64,
    pub source: String,
    pub metadata: Value,
    pub compressed_text: Vec<u8>,
    pub embeddings_blob: Vec<u8>,
}

/// Owns the queue database and creates connections to it.
pub struct Queue {
    db: Database,
}

impl Queue {
    /// Opens (or creates) the queue database at the given path.
    pub async fn new(path: &str) -> Result<Self> {
        let db = Builder::new_local(path).build().await?;
        Ok(Queue { db })
    }

    /// Opens a new connection to the queue database, setting up schema if needed.
    ///
    /// Schema setup is idempotent (`CREATE IF NOT EXISTS`), and must run on each
    /// connection because in-memory databases are per-connection in SQLite.
    pub async fn connect(&self) -> Result<QueueConnection> {
        let conn = self.db.connect()?;
        // WAL mode (no-op on in-memory, fine to ignore result)
        conn.query("PRAGMA journal_mode = WAL", ())
            .await?
            .next()
            .await?;
        // Schema setup — idempotent, safe to run on every new connection
        conn.execute(
            "CREATE TABLE IF NOT EXISTS capsa_queue (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                source     TEXT    NOT NULL,
                metadata   TEXT    NOT NULL,
                text       TEXT    NOT NULL,
                status     TEXT    NOT NULL DEFAULT 'pending',
                error      TEXT,
                created_at INTEGER NOT NULL DEFAULT (unixepoch()),
                updated_at INTEGER NOT NULL DEFAULT (unixepoch())
            )",
            (),
        )
        .await?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_capsa_queue_status ON capsa_queue (status)",
            (),
        )
        .await?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS pending_embeds (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                queue_id   INTEGER NOT NULL,
                source     TEXT    NOT NULL,
                metadata   TEXT    NOT NULL,
                text       TEXT    NOT NULL,
                chunks     TEXT    NOT NULL,
                created_at INTEGER NOT NULL DEFAULT (unixepoch())
            )",
            (),
        )
        .await?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pending_embeds_id ON pending_embeds (id)",
            (),
        )
        .await?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS pending_writes (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                queue_id        INTEGER NOT NULL,
                source          TEXT    NOT NULL,
                metadata        TEXT    NOT NULL,
                compressed_text BLOB    NOT NULL,
                embeddings      BLOB    NOT NULL,
                created_at      INTEGER NOT NULL DEFAULT (unixepoch())
            )",
            (),
        )
        .await?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pending_writes_id ON pending_writes (id)",
            (),
        )
        .await?;
        Ok(QueueConnection { conn })
    }
}

/// A connection to the queue database. Cheap to clone connections from `Queue`.
#[derive(Debug)]
pub struct QueueConnection {
    conn: Connection,
}

impl QueueConnection {
    fn display_name_from_metadata(metadata: &Value, source: &str) -> String {
        for key in ["drive_name", "title", "name"] {
            if let Some(name) = metadata.get(key).and_then(Value::as_str) {
                let trimmed = name.trim();
                if !trimmed.is_empty() && trimmed != "Unknown" {
                    return trimmed.to_string();
                }
            }
        }

        if let Some(path) = metadata.get("path").and_then(Value::as_str)
            && let Some(name) = Self::basename(path)
        {
            return name;
        }

        if let Some((_, raw)) = source.split_once(':')
            && let Some(name) = Self::basename(raw)
        {
            return name;
        }

        Self::basename(source).unwrap_or_else(|| source.to_string())
    }

    fn basename(raw: &str) -> Option<String> {
        let path = raw.trim().trim_end_matches('/').split('?').next()?;
        let name = Path::new(path).file_name()?.to_str()?.trim();
        if name.is_empty() {
            None
        } else {
            Some(name.to_string())
        }
    }

    /// Enqueues a document for ingestion and returns its job ID.
    pub async fn enqueue(&self, source: &str, metadata: &Value, text: &str) -> Result<i64> {
        let metadata_str = serde_json::to_string(metadata)?;
        self.conn
            .execute(
                "INSERT INTO capsa_queue (source, metadata, text) VALUES (?1, ?2, ?3)",
                (source, metadata_str.as_str(), text),
            )
            .await?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Atomically dequeues up to `limit` pending items, marking them as `processing`.
    pub async fn dequeue_batch(&self, limit: usize) -> Result<Vec<QueueItem>> {
        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .await?;

        let items = {
            let mut rows = tx
                .query(
                    "SELECT id, source, metadata, text \
                     FROM capsa_queue WHERE status = 'pending' ORDER BY id LIMIT ?1",
                    [limit as i64],
                )
                .await?;

            let mut items = Vec::new();
            while let Some(row) = rows.next().await? {
                let id: i64 = row.get(0)?;
                let source: String = row.get(1)?;
                let metadata_str: String = row.get(2)?;
                let text: String = row.get(3)?;
                let metadata: Value = serde_json::from_str(&metadata_str)?;
                items.push(QueueItem {
                    id,
                    source,
                    metadata,
                    text,
                });
            }
            items
        };

        for item in &items {
            tx.execute(
                "UPDATE capsa_queue SET status = 'processing', updated_at = unixepoch() \
                 WHERE id = ?1",
                [item.id],
            )
            .await?;
        }

        tx.commit().await?;
        Ok(items)
    }

    /// Marks a queue item as successfully ingested.
    pub async fn mark_done(&self, id: i64) -> Result<()> {
        self.conn
            .execute(
                "UPDATE capsa_queue SET status = 'done', updated_at = unixepoch() WHERE id = ?1",
                [id],
            )
            .await?;
        Ok(())
    }

    /// Marks a queue item as failed with an error description.
    pub async fn mark_failed(&self, id: i64, error: &str) -> Result<()> {
        self.conn
            .execute(
                "UPDATE capsa_queue \
                 SET status = 'failed', error = ?1, updated_at = unixepoch() \
                 WHERE id = ?2",
                (error, id),
            )
            .await?;
        Ok(())
    }

    /// Resets all `failed` items back to `pending` for re-ingestion.
    ///
    /// Returns the number of items re-queued.
    pub async fn retry_failed(&self) -> Result<u64> {
        let mut rows = self
            .conn
            .query(
                "SELECT COUNT(*) FROM capsa_queue WHERE status = 'failed'",
                (),
            )
            .await?;
        let count: u64 = rows
            .next()
            .await?
            .map(|row| row.get::<i64>(0).unwrap_or(0) as u64)
            .unwrap_or(0);

        if count > 0 {
            self.conn
                .execute(
                    "UPDATE capsa_queue \
                     SET status = 'pending', error = NULL, updated_at = unixepoch() \
                     WHERE status = 'failed'",
                    (),
                )
                .await?;
        }

        Ok(count)
    }

    /// Resets all `processing` items back to `pending`.
    ///
    /// Called on startup to recover items that were in-flight when the process
    /// previously crashed.
    pub async fn reset_processing(&self) -> Result<u64> {
        let mut rows = self
            .conn
            .query(
                "SELECT COUNT(*) FROM capsa_queue WHERE status = 'processing'",
                (),
            )
            .await?;
        let count: u64 = rows
            .next()
            .await?
            .map(|row| row.get::<i64>(0).unwrap_or(0) as u64)
            .unwrap_or(0);

        if count > 0 {
            self.conn
                .execute(
                    "UPDATE capsa_queue SET status = 'pending', updated_at = unixepoch() \
                     WHERE status = 'processing'",
                    (),
                )
                .await?;
        }

        Ok(count)
    }

    /// Returns item counts for each pipeline stage.
    pub async fn status_counts(&self) -> Result<PipelineStatus> {
        let mut status = PipelineStatus::default();

        let mut rows = self
            .conn
            .query(
                "SELECT status, COUNT(*) FROM capsa_queue GROUP BY status",
                (),
            )
            .await?;
        while let Some(row) = rows.next().await? {
            let name: String = row.get(0)?;
            let count: i64 = row.get(1)?;
            match name.as_str() {
                "pending" => status.pending = count as u64,
                "done" => status.done = count as u64,
                "failed" => status.failed = count as u64,
                _ => {}
            }
        }

        let mut rows = self
            .conn
            .query("SELECT COUNT(*) FROM pending_embeds", ())
            .await?;
        if let Some(row) = rows.next().await? {
            status.chunked = row.get::<i64>(0)? as u64;
        }

        let mut rows = self
            .conn
            .query("SELECT COUNT(*) FROM pending_writes", ())
            .await?;
        if let Some(row) = rows.next().await? {
            status.embedded = row.get::<i64>(0)? as u64;
        }

        Ok(status)
    }

    /// Returns all failed items ordered by most recent first.
    pub async fn failed_items(&self) -> Result<Vec<FailedItem>> {
        let mut rows = self
            .conn
            .query(
                "SELECT id, source, metadata, COALESCE(error, ''), created_at \
                 FROM capsa_queue WHERE status = 'failed' ORDER BY created_at DESC",
                (),
            )
            .await?;

        let mut items = Vec::new();
        while let Some(row) = rows.next().await? {
            let source: String = row.get(1)?;
            let metadata_str: String = row.get(2)?;
            let metadata: Value = serde_json::from_str(&metadata_str)?;
            items.push(FailedItem {
                job_id: row.get(0)?,
                display_name: Self::display_name_from_metadata(&metadata, &source),
                source,
                error: row.get(3)?,
                created_at: row.get(4)?,
            });
        }

        Ok(items)
    }

    /// Returns all items currently waiting to be embedded (chunked stage), ordered by id.
    pub async fn chunked_items(&self) -> Result<Vec<StageItem>> {
        let mut rows = self
            .conn
            .query(
                "SELECT source, metadata, LENGTH(text), json_array_length(chunks)
                 FROM pending_embeds ORDER BY id",
                (),
            )
            .await?;
        let mut items = Vec::new();
        while let Some(row) = rows.next().await? {
            let source: String = row.get(0)?;
            let metadata_str: String = row.get(1)?;
            let metadata: Value = serde_json::from_str(&metadata_str)?;
            items.push(StageItem {
                display_name: Self::display_name_from_metadata(&metadata, &source),
                size: row.get::<i64>(2)? as usize,
                aux_size: None,
                chunk_count: Some(row.get::<i64>(3)? as usize),
            });
        }
        Ok(items)
    }

    /// Returns all items currently waiting to be written (embedded stage), ordered by id.
    pub async fn embedded_items(&self) -> Result<Vec<StageItem>> {
        let mut rows = self
            .conn
            .query(
                "SELECT source, metadata, LENGTH(embeddings), LENGTH(compressed_text)
                 FROM pending_writes ORDER BY id",
                (),
            )
            .await?;
        let mut items = Vec::new();
        while let Some(row) = rows.next().await? {
            let source: String = row.get(0)?;
            let metadata_str: String = row.get(1)?;
            let metadata: Value = serde_json::from_str(&metadata_str)?;
            items.push(StageItem {
                display_name: Self::display_name_from_metadata(&metadata, &source),
                size: row.get::<i64>(2)? as usize,
                aux_size: Some(row.get::<i64>(3)? as usize),
                chunk_count: None,
            });
        }
        Ok(items)
    }

    /// Returns one pending queue item without modifying its state.
    pub async fn peek_pending(&self) -> Result<Option<QueueItem>> {
        let mut rows = self.conn.query(
            "SELECT id, source, metadata, text FROM capsa_queue WHERE status = 'pending' ORDER BY id LIMIT 1",
            (),
        ).await?;
        if let Some(row) = rows.next().await? {
            let id: i64 = row.get(0)?;
            let source: String = row.get(1)?;
            let metadata_str: String = row.get(2)?;
            let text: String = row.get(3)?;
            let metadata: Value = serde_json::from_str(&metadata_str)?;
            Ok(Some(QueueItem {
                id,
                source,
                metadata,
                text,
            }))
        } else {
            Ok(None)
        }
    }

    /// Returns up to `limit` pending queue items without modifying their state.
    pub async fn peek_pending_batch(&self, limit: usize) -> Result<Vec<QueueItem>> {
        let mut rows = self.conn.query(
            "SELECT id, source, metadata, text FROM capsa_queue WHERE status = 'pending' ORDER BY id LIMIT ?1",
            [limit as i64],
        ).await?;
        let mut items = Vec::new();
        while let Some(row) = rows.next().await? {
            let id: i64 = row.get(0)?;
            let source: String = row.get(1)?;
            let metadata_str: String = row.get(2)?;
            let text: String = row.get(3)?;
            let metadata: Value = serde_json::from_str(&metadata_str)?;
            items.push(QueueItem {
                id,
                source,
                metadata,
                text,
            });
        }
        Ok(items)
    }

    /// Atomically marks a queue item as 'processing' and inserts it into pending_embeds.
    pub async fn promote_to_embeds(
        &self,
        queue_id: i64,
        source: &str,
        metadata: &Value,
        text: &str,
        chunks: &[(String, usize, usize)],
    ) -> Result<()> {
        let chunks_json = serde_json::to_string(chunks)?;
        let metadata_str = serde_json::to_string(metadata)?;
        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .await?;
        tx.execute(
            "UPDATE capsa_queue SET status = 'processing', updated_at = unixepoch() WHERE id = ?1 AND status = 'pending'",
            [queue_id],
        ).await?;
        tx.execute(
            "INSERT INTO pending_embeds (queue_id, source, metadata, text, chunks) VALUES (?1, ?2, ?3, ?4, ?5)",
            (queue_id, source, metadata_str.as_str(), text, chunks_json.as_str()),
        ).await?;
        tx.commit().await?;
        Ok(())
    }

    /// Returns one pending embed item without modifying state.
    pub async fn peek_embed(&self) -> Result<Option<PendingEmbed>> {
        let mut rows = self.conn.query(
            "SELECT id, queue_id, source, metadata, text, chunks FROM pending_embeds ORDER BY id LIMIT 1",
            (),
        ).await?;
        if let Some(row) = rows.next().await? {
            let id: i64 = row.get(0)?;
            let queue_id: i64 = row.get(1)?;
            let source: String = row.get(2)?;
            let metadata_str: String = row.get(3)?;
            let text: String = row.get(4)?;
            let chunks_json: String = row.get(5)?;
            let metadata: Value = serde_json::from_str(&metadata_str)?;
            let chunks: Vec<(String, usize, usize)> = serde_json::from_str(&chunks_json)?;
            Ok(Some(PendingEmbed {
                id,
                queue_id,
                source,
                metadata,
                text,
                chunks,
            }))
        } else {
            Ok(None)
        }
    }

    /// Atomically promotes an embed item to pending_writes and removes it from pending_embeds.
    pub async fn promote_to_writes(
        &self,
        embed_id: i64,
        queue_id: i64,
        source: &str,
        metadata: &Value,
        compressed_text: Vec<u8>,
        embeddings_blob: Vec<u8>,
    ) -> Result<()> {
        let metadata_str = serde_json::to_string(metadata)?;
        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .await?;
        tx.execute(
            "INSERT INTO pending_writes (queue_id, source, metadata, compressed_text, embeddings) VALUES (?1, ?2, ?3, ?4, ?5)",
            libsql::params![queue_id, source, metadata_str.as_str(), compressed_text, embeddings_blob],
        ).await?;
        tx.execute("DELETE FROM pending_embeds WHERE id = ?1", [embed_id])
            .await?;
        tx.commit().await?;
        Ok(())
    }

    /// Marks a queue item as failed and removes it from pending_embeds.
    pub async fn fail_embed(&self, embed_id: i64, queue_id: i64, error: &str) -> Result<()> {
        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .await?;
        tx.execute(
            "UPDATE capsa_queue SET status = 'failed', error = ?1, updated_at = unixepoch() WHERE id = ?2",
            (error, queue_id),
        ).await?;
        tx.execute("DELETE FROM pending_embeds WHERE id = ?1", [embed_id])
            .await?;
        tx.commit().await?;
        Ok(())
    }

    /// Returns one pending write item without modifying state.
    pub async fn peek_write(&self) -> Result<Option<PendingWrite>> {
        let mut rows = self.conn.query(
            "SELECT id, queue_id, source, metadata, compressed_text, embeddings FROM pending_writes ORDER BY id LIMIT 1",
            (),
        ).await?;
        if let Some(row) = rows.next().await? {
            let id: i64 = row.get(0)?;
            let queue_id: i64 = row.get(1)?;
            let source: String = row.get(2)?;
            let metadata_str: String = row.get(3)?;
            let compressed_text: Vec<u8> = row.get(4)?;
            let embeddings_blob: Vec<u8> = row.get(5)?;
            let metadata: Value = serde_json::from_str(&metadata_str)?;
            Ok(Some(PendingWrite {
                id,
                queue_id,
                source,
                metadata,
                compressed_text,
                embeddings_blob,
            }))
        } else {
            Ok(None)
        }
    }

    /// Atomically marks the queue item done and removes it from pending_writes.
    pub async fn complete_write(&self, write_id: i64, queue_id: i64) -> Result<()> {
        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .await?;
        tx.execute(
            "UPDATE capsa_queue SET status = 'done', updated_at = unixepoch() WHERE id = ?1",
            [queue_id],
        )
        .await?;
        tx.execute("DELETE FROM pending_writes WHERE id = ?1", [write_id])
            .await?;
        tx.commit().await?;
        Ok(())
    }

    /// Marks a queue item as failed and removes it from pending_writes.
    pub async fn fail_write(&self, write_id: i64, queue_id: i64, error: &str) -> Result<()> {
        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .await?;
        tx.execute(
            "UPDATE capsa_queue SET status = 'failed', error = ?1, updated_at = unixepoch() WHERE id = ?2",
            (error, queue_id),
        ).await?;
        tx.execute("DELETE FROM pending_writes WHERE id = ?1", [write_id])
            .await?;
        tx.commit().await?;
        Ok(())
    }

    /// Resets pipeline state on startup after a crash.
    ///
    /// Does NOT clear the intermediate tables — items already in `pending_embeds`
    /// or `pending_writes` represent valid completed work and will be picked up
    /// immediately by the embed/write threads' drain-first loops.
    ///
    /// Only fixes true orphans: `'processing'` items in `capsa_queue` that have
    /// no row in either intermediate table. By transaction design this cannot
    /// happen, but we reset them defensively so they are not stuck forever.
    pub async fn reset_pipeline(&self) -> Result<u64> {
        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .await?;
        let mut rows = tx
            .query(
                "SELECT COUNT(*) FROM capsa_queue cq
             WHERE cq.status = 'processing'
               AND NOT EXISTS (SELECT 1 FROM pending_embeds pe WHERE pe.queue_id = cq.id)
               AND NOT EXISTS (SELECT 1 FROM pending_writes pw WHERE pw.queue_id = cq.id)",
                (),
            )
            .await?;
        let count: u64 = rows
            .next()
            .await?
            .map(|row| row.get::<i64>(0).unwrap_or(0) as u64)
            .unwrap_or(0);
        tx.execute(
            "UPDATE capsa_queue SET status = 'pending', updated_at = unixepoch()
             WHERE status = 'processing'
               AND NOT EXISTS (SELECT 1 FROM pending_embeds pe WHERE pe.queue_id = id)
               AND NOT EXISTS (SELECT 1 FROM pending_writes pw WHERE pw.queue_id = id)",
            (),
        )
        .await?;
        tx.commit().await?;
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    async fn make_queue() -> Queue {
        Queue::new(":memory:").await.unwrap()
    }

    #[tokio::test]
    async fn test_enqueue_and_status() {
        let q = make_queue().await;
        let conn = q.connect().await.unwrap();

        conn.enqueue("pdf:a.pdf", &json!({"title": "A"}), "hello")
            .await
            .unwrap();
        conn.enqueue("pdf:b.pdf", &json!({}), "world")
            .await
            .unwrap();

        let s = conn.status_counts().await.unwrap();
        assert_eq!(s.pending, 2);
        assert_eq!(s.chunked, 0);
        assert_eq!(s.embedded, 0);
        assert_eq!(s.done, 0);
        assert_eq!(s.failed, 0);
    }

    #[tokio::test]
    async fn test_dequeue_batch_marks_processing() {
        let q = make_queue().await;
        let conn = q.connect().await.unwrap();

        conn.enqueue("s:a", &json!({}), "doc a").await.unwrap();
        conn.enqueue("s:b", &json!({}), "doc b").await.unwrap();
        conn.enqueue("s:c", &json!({}), "doc c").await.unwrap();

        let items = conn.dequeue_batch(2).await.unwrap();
        assert_eq!(items.len(), 2);

        let s = conn.status_counts().await.unwrap();
        assert_eq!(s.pending, 1);
    }

    #[tokio::test]
    async fn test_dequeue_preserves_order() {
        let q = make_queue().await;
        let conn = q.connect().await.unwrap();

        conn.enqueue("s:first", &json!({}), "first").await.unwrap();
        conn.enqueue("s:second", &json!({}), "second")
            .await
            .unwrap();

        let items = conn.dequeue_batch(2).await.unwrap();
        assert_eq!(items[0].text, "first");
        assert_eq!(items[1].text, "second");
    }

    #[tokio::test]
    async fn test_mark_done() {
        let q = make_queue().await;
        let conn = q.connect().await.unwrap();

        conn.enqueue("s:a", &json!({}), "doc a").await.unwrap();
        let items = conn.dequeue_batch(1).await.unwrap();

        conn.mark_done(items[0].id).await.unwrap();

        let s = conn.status_counts().await.unwrap();
        assert_eq!(s.done, 1);
    }

    #[tokio::test]
    async fn test_mark_failed() {
        let q = make_queue().await;
        let conn = q.connect().await.unwrap();

        conn.enqueue("s:a", &json!({}), "doc a").await.unwrap();
        let items = conn.dequeue_batch(1).await.unwrap();

        conn.mark_failed(items[0].id, "embedding API timeout")
            .await
            .unwrap();

        let s = conn.status_counts().await.unwrap();
        assert_eq!(s.failed, 1);

        let failed = conn.failed_items().await.unwrap();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].error, "embedding API timeout");
        assert_eq!(failed[0].source, "s:a");
    }

    #[tokio::test]
    async fn test_reset_processing() {
        let q = make_queue().await;
        let conn = q.connect().await.unwrap();

        conn.enqueue("s:a", &json!({}), "doc a").await.unwrap();
        conn.enqueue("s:b", &json!({}), "doc b").await.unwrap();
        let _ = conn.dequeue_batch(2).await.unwrap();

        let reset = conn.reset_processing().await.unwrap();
        assert_eq!(reset, 2);

        let s = conn.status_counts().await.unwrap();
        assert_eq!(s.pending, 2);
    }

    #[tokio::test]
    async fn test_empty_dequeue() {
        let q = make_queue().await;
        let conn = q.connect().await.unwrap();

        let items = conn.dequeue_batch(10).await.unwrap();
        assert!(items.is_empty());
    }

    #[test]
    fn test_queue_db_path() {
        assert_eq!(queue_db_path("capsa.db"), "capsa-queue.db");
        assert_eq!(queue_db_path("test.db"), "test-queue.db");
        assert_eq!(queue_db_path("/data/mydb"), "/data/mydb-queue");
        assert_eq!(
            queue_db_path("/path/to/capsa.db"),
            "/path/to/capsa-queue.db"
        );
    }
    // ── Pipeline stage tests ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_peek_pending_empty() {
        let q = make_queue().await;
        let conn = q.connect().await.unwrap();
        assert!(conn.peek_pending().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_peek_pending_does_not_consume() {
        let q = make_queue().await;
        let conn = q.connect().await.unwrap();
        conn.enqueue("pdf:a.pdf", &json!({}), "hello")
            .await
            .unwrap();
        let item = conn.peek_pending().await.unwrap().unwrap();
        assert_eq!(item.source, "pdf:a.pdf");
        assert_eq!(item.text, "hello");
        let s = conn.status_counts().await.unwrap();
        assert_eq!(s.pending, 1);
    }

    #[tokio::test]
    async fn test_promote_to_embeds() {
        let q = make_queue().await;
        let conn = q.connect().await.unwrap();
        let id = conn
            .enqueue("pdf:a.pdf", &json!({"k": "v"}), "text")
            .await
            .unwrap();
        let chunks = vec![("search_document: text".to_string(), 0usize, 4usize)];
        conn.promote_to_embeds(id, "pdf:a.pdf", &json!({"k": "v"}), "text", &chunks)
            .await
            .unwrap();
        let s = conn.status_counts().await.unwrap();
        assert_eq!(s.pending, 0);
        assert_eq!(s.chunked, 1);
        let embed = conn.peek_embed().await.unwrap().unwrap();
        assert_eq!(embed.queue_id, id);
        assert_eq!(embed.source, "pdf:a.pdf");
        assert_eq!(embed.chunks.len(), 1);
        assert_eq!(embed.chunks[0].0, "search_document: text");
        assert_eq!(embed.chunks[0].1, 0);
        assert_eq!(embed.chunks[0].2, 4);
    }

    #[tokio::test]
    async fn test_promote_to_writes() {
        let q = make_queue().await;
        let conn = q.connect().await.unwrap();
        let id = conn.enqueue("pdf:a.pdf", &json!({}), "text").await.unwrap();
        let chunks = vec![("search_document: text".to_string(), 0usize, 4usize)];
        conn.promote_to_embeds(id, "pdf:a.pdf", &json!({}), "text", &chunks)
            .await
            .unwrap();
        let embed_id = conn.peek_embed().await.unwrap().unwrap().id;
        conn.promote_to_writes(
            embed_id,
            id,
            "pdf:a.pdf",
            &json!({}),
            vec![1, 2, 3],
            vec![4, 5, 6],
        )
        .await
        .unwrap();
        assert!(conn.peek_embed().await.unwrap().is_none());
        let write = conn.peek_write().await.unwrap().unwrap();
        assert_eq!(write.queue_id, id);
        assert_eq!(write.compressed_text, vec![1, 2, 3]);
        assert_eq!(write.embeddings_blob, vec![4, 5, 6]);
    }

    #[tokio::test]
    async fn test_chunked_items_use_display_name_and_chunk_count() {
        let q = make_queue().await;
        let conn = q.connect().await.unwrap();
        let metadata = json!({
            "title": "Unknown",
            "drive_name": "real-name.pdf",
        });
        let id = conn
            .enqueue("pdf:/tmp/random.pdf", &metadata, "text")
            .await
            .unwrap();
        let chunks = vec![
            ("search_document: first".to_string(), 0usize, 5usize),
            ("search_document: second".to_string(), 6usize, 12usize),
        ];

        conn.promote_to_embeds(id, "pdf:/tmp/random.pdf", &metadata, "text", &chunks)
            .await
            .unwrap();

        let items = conn.chunked_items().await.unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].display_name, "real-name.pdf");
        assert_eq!(items[0].size, 4);
        assert_eq!(items[0].chunk_count, Some(2));
        assert_eq!(items[0].aux_size, None);
    }

    #[tokio::test]
    async fn test_embedded_items_report_embedding_size_and_display_name() {
        let q = make_queue().await;
        let conn = q.connect().await.unwrap();
        let metadata = json!({
            "title": "spreadsheet-name.xlsx",
        });
        let id = conn
            .enqueue("xls:/tmp/random.xlsx", &metadata, "spreadsheet text")
            .await
            .unwrap();
        let chunks = vec![("search_document: row".to_string(), 0usize, 3usize)];

        conn.promote_to_embeds(
            id,
            "xls:/tmp/random.xlsx",
            &metadata,
            "spreadsheet text",
            &chunks,
        )
        .await
        .unwrap();
        let embed_id = conn.peek_embed().await.unwrap().unwrap().id;
        conn.promote_to_writes(
            embed_id,
            id,
            "xls:/tmp/random.xlsx",
            &metadata,
            vec![1, 2, 3, 4],
            vec![9; 32],
        )
        .await
        .unwrap();

        let items = conn.embedded_items().await.unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].display_name, "spreadsheet-name.xlsx");
        assert_eq!(items[0].size, 32);
        assert_eq!(items[0].aux_size, Some(4));
        assert_eq!(items[0].chunk_count, None);
    }

    #[tokio::test]
    async fn test_complete_write() {
        let q = make_queue().await;
        let conn = q.connect().await.unwrap();
        let id = conn.enqueue("pdf:a.pdf", &json!({}), "text").await.unwrap();
        let chunks = vec![("c".to_string(), 0usize, 1usize)];
        conn.promote_to_embeds(id, "pdf:a.pdf", &json!({}), "text", &chunks)
            .await
            .unwrap();
        let embed_id = conn.peek_embed().await.unwrap().unwrap().id;
        conn.promote_to_writes(embed_id, id, "pdf:a.pdf", &json!({}), vec![], vec![])
            .await
            .unwrap();
        let write_id = conn.peek_write().await.unwrap().unwrap().id;
        conn.complete_write(write_id, id).await.unwrap();
        let s = conn.status_counts().await.unwrap();
        assert_eq!(s.done, 1);
        assert!(conn.peek_write().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_fail_embed() {
        let q = make_queue().await;
        let conn = q.connect().await.unwrap();
        let id = conn.enqueue("pdf:a.pdf", &json!({}), "text").await.unwrap();
        let chunks = vec![("c".to_string(), 0usize, 1usize)];
        conn.promote_to_embeds(id, "pdf:a.pdf", &json!({}), "text", &chunks)
            .await
            .unwrap();
        let embed_id = conn.peek_embed().await.unwrap().unwrap().id;
        conn.fail_embed(embed_id, id, "timeout").await.unwrap();
        let s = conn.status_counts().await.unwrap();
        assert_eq!(s.failed, 1);
        assert!(conn.peek_embed().await.unwrap().is_none());
        assert_eq!(conn.failed_items().await.unwrap()[0].error, "timeout");
    }

    #[tokio::test]
    async fn test_fail_write() {
        let q = make_queue().await;
        let conn = q.connect().await.unwrap();
        let id = conn.enqueue("pdf:a.pdf", &json!({}), "text").await.unwrap();
        let chunks = vec![("c".to_string(), 0usize, 1usize)];
        conn.promote_to_embeds(id, "pdf:a.pdf", &json!({}), "text", &chunks)
            .await
            .unwrap();
        let embed_id = conn.peek_embed().await.unwrap().unwrap().id;
        conn.promote_to_writes(embed_id, id, "pdf:a.pdf", &json!({}), vec![], vec![])
            .await
            .unwrap();
        let write_id = conn.peek_write().await.unwrap().unwrap().id;
        conn.fail_write(write_id, id, "db error").await.unwrap();
        let s = conn.status_counts().await.unwrap();
        assert_eq!(s.failed, 1);
        assert!(conn.peek_write().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_reset_pipeline_preserves_intermediate_tables() {
        let q = make_queue().await;
        let conn = q.connect().await.unwrap();
        let id = conn.enqueue("pdf:a.pdf", &json!({}), "text").await.unwrap();
        let chunks = vec![("c".to_string(), 0usize, 1usize)];
        conn.promote_to_embeds(id, "pdf:a.pdf", &json!({}), "text", &chunks)
            .await
            .unwrap();
        // Item is now 'processing' in capsa_queue with a row in pending_embeds.
        // reset_pipeline should NOT disturb it — the embed thread can resume.
        let reset = conn.reset_pipeline().await.unwrap();
        assert_eq!(reset, 0); // no orphans
        assert!(conn.peek_embed().await.unwrap().is_some()); // still in pending_embeds
        let s = conn.status_counts().await.unwrap();
        assert_eq!(s.pending, 0);
        assert_eq!(s.chunked, 1); // preserved
    }

    #[tokio::test]
    async fn test_reset_pipeline_fixes_orphans() {
        // Simulate an orphan: 'processing' in capsa_queue but nothing in intermediate tables.
        // This can't happen via normal code paths (transactions), but we test the safety net.
        // Use dequeue_batch to move items to 'processing' without inserting into pending_embeds.
        let q = make_queue().await;
        let conn = q.connect().await.unwrap();
        conn.enqueue("pdf:a.pdf", &json!({}), "text").await.unwrap();
        let _ = conn.dequeue_batch(1).await.unwrap(); // sets status='processing', no intermediate row
        let reset = conn.reset_pipeline().await.unwrap();
        assert_eq!(reset, 1);
        let s = conn.status_counts().await.unwrap();
        assert_eq!(s.pending, 1);
    }
}
