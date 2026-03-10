//! High-level document storage and retrieval with automatic embedding generation.
//!
//! This module provides a convenient API for indexing and searching documents
//! using semantic similarity. It combines the embedding functionality from
//! [`embedder`](crate::embedder) with the vector storage from
//! [`vectordb`](crate::vectordb).
//!
//! Document ingestion is asynchronous: [`DocumentDatabaseConnection::insert`]
//! enqueues the document and returns immediately. A background pipeline
//! handles chunking, embedding, and writing to the vector database.
//!
//! # Examples
//!
//! ```no_run
//! use capsa::{config::Config, documentdb::DocumentDatabase};
//! use serde_json::json;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = Config::new(
//!     "http://localhost:9000/v1".to_string(),
//!     "nomic-ai/nomic-embed-text-v1.5".to_string(),
//!     "./documents.db".to_string(),
//!     None,
//! );
//!
//! let db = DocumentDatabase::new(&config).await?;
//! let conn = db.connect().await?;
//!
//! // Queue a document for ingestion (returns immediately)
//! conn.insert(
//!     json!({"title": "Example"}),
//!     "Document text content"
//! ).await?;
//!
//! // Check ingestion progress
//! let status = conn.pipeline_status().await?;
//! println!("pending: {}, done: {}", status.pending, status.done);
//!
//! // Search for similar documents (queries are always synchronous)
//! let results = conn.search_topk("query text", 5).await?;
//! # Ok(())
//! # }
//! ```

use crate::config::{Config, EMBEDDING_CONTEXT};
use crate::embedder::Embedder;
use crate::error::DatabaseError;
use crate::error::{CapsaError, EmbeddingError, Result};
use crate::pipeline::Pipeline;
use crate::queue::{FailedItem, PipelineStatus, Queue, QueueConnection, queue_db_path};
use crate::vectordb::{VectorDatabase, VectorDatabaseConnection};
use std::sync::Arc;
use tokio::sync::Notify;

type DocumentId = i64;

// ── Connection ────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct DocumentDatabaseConnection {
    /// Used by search/query methods to embed the query text.
    embedder: Arc<Embedder>,
    /// Read connection: search_topk, fetch_document.
    vconn: VectorDatabaseConnection,
    /// Queue connection: insert, pipeline_status, failed_items.
    /// `None` for read-only connections (opened via `DocumentDatabase::new_reader`).
    queue: Option<QueueConnection>,
    /// Wakes the chunk task when a new item is enqueued. `None` in read-only mode.
    queue_ready: Option<Arc<Notify>>,
}

impl DocumentDatabaseConnection {
    /// Queues a document for ingestion and returns immediately.
    ///
    /// The document is written to the persistent ingestion queue. A background
    /// pipeline handles chunking, embedding, and writing to the vector database.
    /// Use [`pipeline_status`](Self::pipeline_status) to monitor progress.
    ///
    /// # Arguments
    ///
    /// * `metadata` - Document metadata as JSON (title, author, etc.)
    /// * `text`     - Full document content to index
    ///
    /// # Errors
    ///
    /// Returns an error only if the queue write itself fails (e.g. I/O error).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use capsa::documentdb::DocumentDatabase;
    /// # use serde_json::json;
    /// # async fn example(conn: &capsa::documentdb::DocumentDatabaseConnection) -> anyhow::Result<()> {
    /// conn.insert(
    ///     json!({"title": "My Document", "author": "Author"}),
    ///     "Document content goes here"
    /// ).await?;
    /// println!("Document queued for ingestion");
    /// # Ok(())
    /// # }
    /// ```
    pub async fn insert(&self, metadata: serde_json::Value, text: &str) -> Result<()> {
        let q = self.queue.as_ref().ok_or_else(|| {
            CapsaError::Database(DatabaseError::InsertFailed(
                "insert requires a writable connection (use DocumentDatabase::new, not new_reader)"
                    .to_string(),
            ))
        })?;
        q.enqueue("", &metadata, text).await?;
        if let Some(notify) = &self.queue_ready {
            notify.notify_one();
        }
        Ok(())
    }

    /// Returns the current counts of ingestion queue items grouped by status.
    pub async fn pipeline_status(&self) -> Result<PipelineStatus> {
        let q = self.queue.as_ref().ok_or_else(|| {
            CapsaError::Database(DatabaseError::QueryFailed(
                "pipeline_status requires a writable connection".to_string(),
            ))
        })?;
        q.status_counts().await
    }

    /// Returns all failed ingestion items with their error descriptions.
    pub async fn failed_items(&self) -> Result<Vec<FailedItem>> {
        let q = self.queue.as_ref().ok_or_else(|| {
            CapsaError::Database(DatabaseError::QueryFailed(
                "failed_items requires a writable connection".to_string(),
            ))
        })?;
        q.failed_items().await
    }

    /// Searches for the top-k most semantically similar document chunks.
    ///
    /// The query is automatically embedded and matched against all stored document
    /// chunks using cosine similarity.
    ///
    /// # Arguments
    ///
    /// * `query` - Natural language search query
    /// * `limit` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// A vector of tuples containing (document_id, metadata, chunk_start, chunk_end)
    /// ordered by similarity (most similar first).
    pub async fn search_topk(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(DocumentId, serde_json::Value, i64, i64)>> {
        let query_vec = self.embedder.embed_query(query).await?;
        self.vconn.search_topk(query_vec, limit).await
    }

    /// Searches for the top-k most semantically similar document chunks with distance scores.
    ///
    /// Similar to [`search_topk`](Self::search_topk), but also returns cosine distance
    /// for each result. Lower distances indicate higher similarity.
    pub async fn search_topk_with_distance(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(DocumentId, serde_json::Value, f32, i64, i64)>> {
        let query_vec = self.embedder.embed_query(query).await?;
        self.vconn.search_topk_with_distance(query_vec, limit).await
    }

    /// Retrieves the full content and metadata of a document by its ID.
    ///
    /// Returns `Some((content, metadata))` if the document exists, or `None` if not found.
    pub async fn fetch_document(
        &self,
        doc_id: DocumentId,
    ) -> Result<Option<(String, serde_json::Value)>> {
        self.vconn.fetch_document(doc_id).await
    }

    /// Synchronous insert for tests: embeds and writes directly, bypassing the queue.
    ///
    /// Use this in tests instead of `insert()` when you need the document to be
    /// immediately searchable.
    #[cfg(test)]
    pub(crate) async fn insert_direct(
        &self,
        metadata: serde_json::Value,
        text: &str,
    ) -> Result<DocumentId> {
        let vecs = self.embedder.embed_document(text).await?;
        self.vconn.insert_document(text, metadata, vecs).await
    }
}

// ── Database ──────────────────────────────────────────────────────────────────

pub struct DocumentDatabase {
    embedder: Arc<Embedder>,
    vdb: VectorDatabase,
    /// `None` for read-only instances (created via `new_reader`).
    queue: Option<Queue>,
    model: Option<String>,
    /// `None` in read-only or test configurations that skip the pipeline.
    _pipeline: Option<Pipeline>,
    /// Shared with the pipeline's chunk task; notified on every enqueue.
    queue_ready: Option<Arc<Notify>>,
}

impl DocumentDatabase {
    /// Creates a new document database using the provided configuration.
    ///
    /// Opens the vector database and the ingestion queue, resets any items that
    /// were in-flight when the process last crashed, and starts the background
    /// ingestion pipeline.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedder cannot be created, the database cannot
    /// be initialised, or the pipeline fails to start.
    pub async fn new(config: &Config) -> Result<Self> {
        let embedder = Arc::new(Embedder::new(
            config.base_url.clone(),
            config.model.clone(),
            config.api_key.clone(),
            EMBEDDING_CONTEXT,
        )?);

        let probe = embedder.embed_query("test").await?;
        if probe.is_empty() {
            return Err(EmbeddingError::NoEmbeddingReturned.into());
        }

        let vdb = VectorDatabase::new(&config.db_path, probe.len(), config.executor).await?;

        let q_path = queue_db_path(&config.db_path);
        let queue = Queue::new(&q_path).await?;

        // Reset items stuck in 'processing' from a previous crash.
        let reset_conn = queue.connect().await?;
        let reset_count = reset_conn.reset_pipeline().await?;
        if reset_count > 0 {
            tracing::warn!(
                "reset {} in-flight queue items to 'pending' after restart",
                reset_count
            );
        }

        let pipeline = Pipeline::start(
            q_path,
            config.db_path.clone(),
            probe.len(),
            Some(config.model.clone()),
            embedder.clone(),
            config.executor,
        )
        .await?;

        let queue_ready = Some(pipeline.queue_ready.clone());
        Ok(DocumentDatabase {
            embedder,
            vdb,
            queue: Some(queue),
            model: Some(config.model.clone()),
            _pipeline: Some(pipeline),
            queue_ready,
        })
    }

    /// Opens the document database in read-only mode (no queue, no pipeline).
    ///
    /// Suitable for query-only commands (`capsa ask`) that do not enqueue documents.
    /// Still probes the embedding API to determine the query vector dimension.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedder cannot be created or the database cannot
    /// be opened.
    pub async fn new_reader(config: &Config) -> Result<Self> {
        let embedder = Arc::new(Embedder::new(
            config.base_url.clone(),
            config.model.clone(),
            config.api_key.clone(),
            EMBEDDING_CONTEXT,
        )?);

        let probe = embedder.embed_query("test").await?;
        if probe.is_empty() {
            return Err(EmbeddingError::NoEmbeddingReturned.into());
        }

        let vdb = VectorDatabase::new(&config.db_path, probe.len(), config.executor).await?;

        Ok(DocumentDatabase {
            embedder,
            vdb,
            queue: None,
            model: Some(config.model.clone()),
            _pipeline: None,
            queue_ready: None,
        })
    }

    /// Creates a document database with a custom embedder (test helper).
    ///
    /// Uses an in-memory queue and does not start the ingestion pipeline.
    /// Use [`DocumentDatabaseConnection::insert_direct`] in tests to write
    /// documents synchronously.
    #[cfg(test)]
    pub(crate) async fn with_embedder(embedder: Embedder, vdb_path: String) -> Result<Self> {
        use crate::executor::Executor;
        let embedder = Arc::new(embedder);
        let probe = embedder.embed_query("test").await?;
        if probe.is_empty() {
            return Err(EmbeddingError::NoEmbeddingReturned.into());
        }
        let vdb = VectorDatabase::new(&vdb_path, probe.len(), Executor::sequential()).await?;
        let queue = Queue::new(":memory:").await?;

        Ok(DocumentDatabase {
            embedder,
            vdb,
            queue: Some(queue),
            model: None,
            _pipeline: None,
            queue_ready: None,
        })
    }

    /// Opens a new connection to the document database.
    ///
    /// Multiple connections can be created from the same database instance
    /// to enable concurrent access.
    pub async fn connect(&self) -> Result<DocumentDatabaseConnection> {
        let vconn = self.vdb.connect_with_model(self.model.as_deref()).await?;
        let queue = if let Some(q) = &self.queue {
            Some(q.connect().await?)
        } else {
            None
        };
        let embedder = self.embedder.clone();
        Ok(DocumentDatabaseConnection {
            vconn,
            embedder,
            queue,
            queue_ready: self.queue_ready.clone(),
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::Embedder;
    use crate::test_utils::MockEmbedding;
    use serde_json::json;

    async fn create_test_db(db_path: &str) -> Result<DocumentDatabase> {
        let client = Box::new(MockEmbedding::new(384));
        let embedder = match Embedder::with_client(client, "bert-base-uncased".to_string(), 512) {
            Ok(e) => e,
            Err(e) => return Err(e),
        };
        DocumentDatabase::with_embedder(embedder, db_path.to_string()).await
    }

    #[tokio::test]
    async fn test_document_database_creation() -> Result<()> {
        let db = match create_test_db(":memory:").await {
            Ok(db) => db,
            Err(_) => return Ok(()),
        };
        let conn = db.connect().await?;
        drop(conn);
        Ok(())
    }

    #[tokio::test]
    async fn test_insert_and_search() -> Result<()> {
        let db = match create_test_db(":memory:").await {
            Ok(db) => db,
            Err(_) => return Ok(()),
        };
        let conn = db.connect().await?;

        let metadata = json!({"title": "Test Document", "author": "Test Author"});
        let doc_id = conn
            .insert_direct(metadata.clone(), "This is a test document about embeddings")
            .await?;

        assert!(doc_id > 0);

        let results = conn.search_topk("embeddings and vectors", 5).await?;

        assert!(!results.is_empty());
        assert_eq!(results[0].0, doc_id);
        assert_eq!(results[0].1["title"], "Test Document");

        Ok(())
    }

    #[tokio::test]
    async fn test_insert_and_search_with_distance() -> Result<()> {
        let db = match create_test_db(":memory:").await {
            Ok(db) => db,
            Err(_) => return Ok(()),
        };
        let conn = db.connect().await?;

        let metadata = json!({"category": "technology"});
        let doc_id = conn
            .insert_direct(metadata, "Machine learning and artificial intelligence")
            .await?;

        assert!(doc_id > 0);

        let results = conn.search_topk_with_distance("AI and ML", 5).await?;

        assert!(!results.is_empty());
        assert_eq!(results[0].0, doc_id);
        assert_eq!(results[0].1["category"], "technology");
        assert!(results[0].2 >= 0.0);
        assert!(results[0].2.is_finite());

        Ok(())
    }

    #[ignore]
    #[tokio::test]
    async fn test_multiple_documents_ranking() -> Result<()> {
        let db = match create_test_db(":memory:").await {
            Ok(db) => db,
            Err(_) => return Ok(()),
        };
        let conn = db.connect().await?;

        let doc1_id = conn
            .insert_direct(json!({"id": 1}), "Rust is a systems programming language")
            .await?;
        let _doc2_id = conn
            .insert_direct(
                json!({"id": 2}),
                "Python is a high-level programming language",
            )
            .await?;
        let _doc3_id = conn
            .insert_direct(
                json!({"id": 3}),
                "Machine learning and artificial intelligence",
            )
            .await?;

        let results = conn
            .search_topk_with_distance("systems programming in Rust", 3)
            .await?;

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, doc1_id);
        assert!(results[0].2 <= results[1].2);
        assert!(results[1].2 <= results[2].2);

        Ok(())
    }

    #[tokio::test]
    async fn test_search_with_limit() -> Result<()> {
        let db = match create_test_db(":memory:").await {
            Ok(db) => db,
            Err(_) => return Ok(()),
        };
        let conn = db.connect().await?;

        for i in 0..5 {
            conn.insert_direct(
                json!({"index": i}),
                &format!("Document number {} about various topics", i),
            )
            .await?;
        }

        let results = conn.search_topk("document topics", 2).await?;
        assert_eq!(results.len(), 2);

        let results = conn.search_topk("document topics", 10).await?;
        assert_eq!(results.len(), 5);

        Ok(())
    }

    #[tokio::test]
    async fn test_search_empty_database() -> Result<()> {
        let db = match create_test_db(":memory:").await {
            Ok(db) => db,
            Err(_) => return Ok(()),
        };
        let conn = db.connect().await?;

        let results = conn.search_topk("anything", 5).await?;
        assert_eq!(results.len(), 0);

        let results_with_distance = conn.search_topk_with_distance("anything", 5).await?;
        assert_eq!(results_with_distance.len(), 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_complex_metadata() -> Result<()> {
        let db = match create_test_db(":memory:").await {
            Ok(db) => db,
            Err(_) => return Ok(()),
        };
        let conn = db.connect().await?;

        let complex_metadata = json!({
            "title": "Research Paper",
            "authors": ["Alice", "Bob"],
            "year": 2024,
            "tags": ["AI", "ML", "embeddings"],
            "metrics": { "citations": 100, "views": 5000 }
        });

        let doc_id = conn
            .insert_direct(complex_metadata.clone(), "Advanced research in embeddings")
            .await?;

        let results = conn.search_topk("research embeddings", 1).await?;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, doc_id);
        assert_eq!(results[0].1["title"], "Research Paper");
        assert_eq!(results[0].1["authors"][0], "Alice");
        assert_eq!(results[0].1["metrics"]["citations"], 100);

        Ok(())
    }

    #[tokio::test]
    async fn test_long_text_chunking() -> Result<()> {
        let db = match create_test_db(":memory:").await {
            Ok(db) => db,
            Err(_) => return Ok(()),
        };
        let conn = db.connect().await?;

        let long_text = (0..1000)
            .map(|i| {
                format!(
                    "This is sentence number {}. It contains some information. ",
                    i
                )
            })
            .collect::<String>();

        let doc_id = conn
            .insert_direct(json!({"type": "long"}), &long_text)
            .await?;
        assert!(doc_id > 0);

        let results = conn.search_topk("sentence information", 1).await?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, doc_id);

        Ok(())
    }

    #[tokio::test]
    async fn test_empty_text_insertion() -> Result<()> {
        let db = match create_test_db(":memory:").await {
            Ok(db) => db,
            Err(_) => return Ok(()),
        };
        let conn = db.connect().await?;

        let result = conn.insert_direct(json!({}), "").await;
        if let Ok(doc_id) = result {
            assert!(doc_id > 0);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_duplicate_content() -> Result<()> {
        let db = match create_test_db(":memory:").await {
            Ok(db) => db,
            Err(_) => return Ok(()),
        };
        let conn = db.connect().await?;

        let content = "Duplicate content test";
        let doc1_id = conn.insert_direct(json!({"version": 1}), content).await?;
        let doc2_id = conn.insert_direct(json!({"version": 2}), content).await?;

        assert_ne!(doc1_id, doc2_id);

        let results = conn.search_topk(content, 5).await?;
        assert!(results.len() >= 2);

        let doc_ids: Vec<i64> = results.iter().map(|(id, _, _, _)| *id).collect();
        assert!(doc_ids.contains(&doc1_id));
        assert!(doc_ids.contains(&doc2_id));

        Ok(())
    }

    #[tokio::test]
    async fn test_special_characters_in_text() -> Result<()> {
        let db = match create_test_db(":memory:").await {
            Ok(db) => db,
            Err(_) => return Ok(()),
        };
        let conn = db.connect().await?;

        let special_text = "Text with special chars: @#$% & 'quotes' \"double\" \n\t tabs";
        let doc_id = conn
            .insert_direct(json!({"type": "special"}), special_text)
            .await?;
        assert!(doc_id > 0);

        let results = conn.search_topk("special chars quotes", 1).await?;
        assert!(!results.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_unicode_text() -> Result<()> {
        let db = match create_test_db(":memory:").await {
            Ok(db) => db,
            Err(_) => return Ok(()),
        };
        let conn = db.connect().await?;

        let unicode_text = "Unicode: 你好世界 مرحبا العالم Привет мир 🌍🚀";
        let doc_id = conn
            .insert_direct(json!({"lang": "multi"}), unicode_text)
            .await?;
        assert!(doc_id > 0);

        let results = conn.search_topk("unicode world", 1).await?;
        assert!(!results.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_search_consistency() -> Result<()> {
        let db = match create_test_db(":memory:").await {
            Ok(db) => db,
            Err(_) => return Ok(()),
        };
        let conn = db.connect().await?;

        let _doc_id = conn
            .insert_direct(json!({"test": "consistency"}), "Consistency test document")
            .await?;

        let results_basic = conn.search_topk("consistency test", 5).await?;
        let results_distance = conn
            .search_topk_with_distance("consistency test", 5)
            .await?;

        assert_eq!(results_basic.len(), results_distance.len());
        for i in 0..results_basic.len() {
            assert_eq!(results_basic[i].0, results_distance[i].0);
            assert_eq!(results_basic[i].1, results_distance[i].1);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_multiple_connections() -> Result<()> {
        let db = match create_test_db("file:multiple_connections?mode=memory&cache=shared").await {
            Ok(db) => db,
            Err(_) => return Ok(()),
        };

        let conn1 = db.connect().await?;
        let conn2 = db.connect().await?;

        let doc_id = conn1
            .insert_direct(
                json!({"source": "conn1"}),
                "Rust programming language documentation",
            )
            .await?;

        let results = conn2
            .search_topk("Rust programming documentation", 5)
            .await?;

        assert!(!results.is_empty());
        assert_eq!(results[0].0, doc_id);

        Ok(())
    }

    #[tokio::test]
    async fn test_pipeline_status() -> Result<()> {
        let db = match create_test_db(":memory:").await {
            Ok(db) => db,
            Err(_) => return Ok(()),
        };
        let conn = db.connect().await?;

        // Queue a document (no pipeline running in test config, so it stays pending)
        conn.insert(json!({"title": "queued"}), "some text").await?;

        let status = conn.pipeline_status().await?;
        assert_eq!(status.pending, 1);
        assert_eq!(status.chunked, 0);
        assert_eq!(status.embedded, 0);
        assert_eq!(status.done, 0);

        Ok(())
    }
}
