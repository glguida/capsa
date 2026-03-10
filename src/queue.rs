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
    pub metadata: Value,
    pub text: String,
}

/// Pipeline status counts by state.
#[derive(Debug, Clone, Default)]
pub struct PipelineStatus {
    pub pending: u64,
    pub processing: u64,
    pub done: u64,
    pub failed: u64,
}

/// A failed ingestion item with diagnostic information.
#[derive(Debug, Clone)]
pub struct FailedItem {
    pub job_id: i64,
    pub source: String,
    pub error: String,
    pub created_at: i64,
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
        Ok(QueueConnection { conn })
    }
}

/// A connection to the queue database. Cheap to clone connections from `Queue`.
#[derive(Debug)]
pub struct QueueConnection {
    conn: Connection,
}

impl QueueConnection {
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
                    "SELECT id, metadata, text \
                     FROM capsa_queue WHERE status = 'pending' ORDER BY id LIMIT ?1",
                    [limit as i64],
                )
                .await?;

            let mut items = Vec::new();
            while let Some(row) = rows.next().await? {
                let id: i64 = row.get(0)?;
                let metadata_str: String = row.get(1)?;
                let text: String = row.get(2)?;
                let metadata: Value = serde_json::from_str(&metadata_str)?;
                items.push(QueueItem { id, metadata, text });
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

    /// Returns item counts grouped by status.
    pub async fn status_counts(&self) -> Result<PipelineStatus> {
        let mut rows = self
            .conn
            .query(
                "SELECT status, COUNT(*) FROM capsa_queue GROUP BY status",
                (),
            )
            .await?;

        let mut status = PipelineStatus::default();
        while let Some(row) = rows.next().await? {
            let name: String = row.get(0)?;
            let count: i64 = row.get(1)?;
            match name.as_str() {
                "pending" => status.pending = count as u64,
                "processing" => status.processing = count as u64,
                "done" => status.done = count as u64,
                "failed" => status.failed = count as u64,
                _ => {}
            }
        }

        Ok(status)
    }

    /// Returns all failed items ordered by most recent first.
    pub async fn failed_items(&self) -> Result<Vec<FailedItem>> {
        let mut rows = self
            .conn
            .query(
                "SELECT id, source, COALESCE(error, ''), created_at \
                 FROM capsa_queue WHERE status = 'failed' ORDER BY created_at DESC",
                (),
            )
            .await?;

        let mut items = Vec::new();
        while let Some(row) = rows.next().await? {
            items.push(FailedItem {
                job_id: row.get(0)?,
                source: row.get(1)?,
                error: row.get(2)?,
                created_at: row.get(3)?,
            });
        }

        Ok(items)
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
        assert_eq!(s.processing, 0);
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
        assert_eq!(s.processing, 2);
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
        assert_eq!(s.processing, 0);
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
        assert_eq!(s.processing, 0);
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
}
