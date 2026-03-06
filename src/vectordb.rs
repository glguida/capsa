//! Low-level vector database operations using libSQL.
//!
//! This module provides direct access to vector storage and similarity search
//! functionality. It manages document storage, embedding vectors, and vector
//! indices using libSQL's native vector support.
//!
//! For most use cases, prefer the higher-level [`documentdb`](crate::documentdb)
//! module which handles embedding generation automatically.

use crate::error::{DatabaseError, Result};
use crate::executor::Executor;
use libsql::{Builder, Connection, Database, Transaction, TransactionBehavior};
use serde_json::Value;
use std::sync::Arc;

#[derive(Debug)]
struct VectorDatabaseSchema {
    vec_size: usize,
    create_documents_table: String,
    create_vectors_table: String,
    create_index_table: String,
    insert_document: String,
    insert_vectors: String,
    search_topk: String,
    search_topk_with_distance: String,
    search_topk_with_vectors: String,
    fetch_document: String,
}

impl VectorDatabaseSchema {
    fn new(vec_size: usize) -> Self {
        let create_documents_table = format!(
            "CREATE TABLE IF NOT EXISTS documents_{} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content BLOB,
                metadata JSON
            )",
            vec_size
        );
        let create_vectors_table = format!(
            "CREATE TABLE IF NOT EXISTS document_embeddings_{} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER,
                chunk_index INTEGER,
                chunk_start INTEGER,
                chunk_end INTEGER,
                embedding F32_BLOB({}),
                FOREIGN KEY(doc_id) REFERENCES documents_{}(id) ON DELETE CASCADE
            )",
            vec_size, vec_size, vec_size
        );
        let create_index_table = format!(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_{} 
             ON document_embeddings_{}(libsql_vector_idx(embedding, 'metric=cosine'))",
            vec_size, vec_size
        );
        let insert_document = format!(
            "INSERT INTO documents_{} (content, metadata) VALUES (?1, ?2) RETURNING id",
            vec_size
        );
        let insert_vectors = format!(
            "INSERT INTO document_embeddings_{} (doc_id, chunk_index, chunk_start, chunk_end, embedding)
                 VALUES (?1, ?2, ?3, ?4, vector(?5))",
            vec_size
        );
        let search_topk = format!(
            "SELECT e.doc_id, d.metadata, e.chunk_start, e.chunk_end
             FROM vector_top_k('idx_embeddings_{}', vector(?1), ?2) v
             JOIN document_embeddings_{} e ON e.rowid = v.id
             JOIN documents_{} d ON e.doc_id = d.id",
            vec_size, vec_size, vec_size
        );
        let search_topk_with_distance = format!(
            "SELECT e.doc_id, d.metadata, vector_distance_cos(e.embedding, vector(?1)) AS distance, e.chunk_start, e.chunk_end
             FROM vector_top_k('idx_embeddings_{}', vector(?1), ?2) v
             JOIN document_embeddings_{} e ON e.rowid = v.id
             JOIN documents_{} d ON e.doc_id = d.id",
            vec_size, vec_size, vec_size
        );
        let search_topk_with_vectors = format!(
            "SELECT e.doc_id, d.metadata, e.embedding, e.chunk_start, e.chunk_end
             FROM vector_top_k('idx_embeddings_{}', vector(?1), ?2) v
             JOIN document_embeddings_{} e ON e.rowid = v.id
             JOIN documents_{} d ON e.doc_id = d.id",
            vec_size, vec_size, vec_size
        );
        let fetch_document = format!(
            "SELECT content, metadata FROM documents_{} WHERE id = ?1",
            vec_size
        );
        VectorDatabaseSchema {
            vec_size,
            create_documents_table,
            create_vectors_table,
            create_index_table,
            insert_document,
            insert_vectors,
            search_topk,
            search_topk_with_distance,
            search_topk_with_vectors,
            fetch_document,
        }
    }
}

#[derive(Debug)]
pub struct VectorDatabaseConnection {
    conn: Connection,
    schema: Arc<VectorDatabaseSchema>,
    executor: Executor,
}

impl VectorDatabaseConnection {
    /// Creates a new vector database connection and initializes the schema.
    ///
    /// This sets up the necessary tables and indices for vector storage if they
    /// do not already exist.
    async fn new(
        conn: Connection,
        schema: Arc<VectorDatabaseSchema>,
        executor: Executor,
        expected_model: Option<&str>,
    ) -> Result<Self> {
        // Enable WAL mode for better performance and concurrency
        // See: https://www.sqlite.org/wal.html
        // Note: WAL mode is not available for in-memory databases, which will
        // continue using the default journal mode. This is expected and acceptable.
        // See: https://www.sqlite.org/pragma.html#pragma_journal_mode
        // PRAGMA journal_mode returns a row with the mode that was actually set,
        // so we must use query() instead of execute() (which expects no result rows).
        // We consume the result to ensure the command succeeded.
        conn.query("PRAGMA journal_mode = WAL", ())
            .await?
            .next()
            .await?;

        // Enable foreign keys for the ON DELETE CASCADE behavior
        conn.execute("PRAGMA foreign_keys = ON", ()).await?;
        // Enable WAL. 'query' because there's a "Execute returned rows" error. To investigate.
        conn.query("PRAGMA journal_mode = WAL", ()).await?;

        Self::validate_metadata(&conn, schema.vec_size, expected_model).await?;

        let db = Self {
            conn,
            schema,
            executor,
        };
        db.setup_schema().await?;

        Ok(db)
    }

    async fn setup_metadata_table(conn: &Connection) -> Result<()> {
        conn.execute(
            "CREATE TABLE IF NOT EXISTS capsa_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )",
            (),
        )
        .await?;
        Ok(())
    }

    async fn get_metadata(conn: &Connection, key: &str) -> Result<Option<String>> {
        let mut rows = conn
            .query("SELECT value FROM capsa_metadata WHERE key = ?1", [key])
            .await?;
        if let Some(row) = rows.next().await? {
            let value: String = row.get(0)?;
            Ok(Some(value))
        } else {
            Ok(None)
        }
    }

    async fn get_metadata_tx(tx: &Transaction, key: &str) -> Result<Option<String>> {
        let mut rows = tx
            .query("SELECT value FROM capsa_metadata WHERE key = ?1", [key])
            .await?;
        if let Some(row) = rows.next().await? {
            let value: String = row.get(0)?;
            Ok(Some(value))
        } else {
            Ok(None)
        }
    }

    async fn set_metadata_tx(tx: &Transaction, key: &str, value: &str) -> Result<()> {
        tx.execute(
            "INSERT INTO capsa_metadata (key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        .await?;
        Ok(())
    }

    async fn detect_legacy_dimensions_tx(tx: &Transaction) -> Result<Vec<usize>> {
        let mut rows = tx
            .query(
                "SELECT name FROM sqlite_master
                 WHERE type = 'table' AND name GLOB 'documents_[0-9]*'",
                (),
            )
            .await?;

        let mut dims = Vec::new();
        while let Some(row) = rows.next().await? {
            let table_name: String = row.get(0)?;
            if let Some(raw_dim) = table_name.strip_prefix("documents_") {
                let dim = raw_dim.parse::<usize>().map_err(|e| {
                    DatabaseError::SchemaSetup(format!(
                        "Invalid legacy table name '{}': {}",
                        table_name, e
                    ))
                })?;
                dims.push(dim);
            }
        }

        dims.sort_unstable();
        dims.dedup();
        Ok(dims)
    }

    fn parse_embedding_dim(raw: &str) -> Result<usize> {
        raw.parse::<usize>().map_err(|e| {
            DatabaseError::SchemaSetup(format!(
                "Invalid stored embedding dimension '{}': {}",
                raw, e
            ))
            .into()
        })
    }

    async fn validate_metadata(
        conn: &Connection,
        expected_dim: usize,
        expected_model: Option<&str>,
    ) -> Result<()> {
        Self::setup_metadata_table(conn).await?;
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .await?;

        match Self::get_metadata_tx(&tx, "embedding_dim").await? {
            Some(raw_dim) => {
                let stored_dim = Self::parse_embedding_dim(&raw_dim)?;
                if stored_dim != expected_dim {
                    return Err(DatabaseError::SchemaSetup(format!(
                        "Embedding dimension mismatch: database uses {}, requested {}",
                        stored_dim, expected_dim
                    ))
                    .into());
                }
            }
            None => {
                let legacy_dims = Self::detect_legacy_dimensions_tx(&tx).await?;
                match legacy_dims.as_slice() {
                    [] => {}
                    [legacy_dim] if *legacy_dim == expected_dim => {}
                    [legacy_dim] => {
                        return Err(DatabaseError::SchemaSetup(format!(
                            "Legacy database uses embedding dimension {}, requested {}",
                            legacy_dim, expected_dim
                        ))
                        .into());
                    }
                    _ => {
                        return Err(DatabaseError::SchemaSetup(format!(
                            "Legacy database contains multiple embedding dimensions: {:?}",
                            legacy_dims
                        ))
                        .into());
                    }
                }

                Self::set_metadata_tx(&tx, "embedding_dim", &expected_dim.to_string()).await?;
            }
        }

        if let Some(model) = expected_model {
            match Self::get_metadata_tx(&tx, "embedding_model").await? {
                Some(stored_model) => {
                    if stored_model != model {
                        return Err(DatabaseError::SchemaSetup(format!(
                            "Embedding model mismatch: database uses '{}', requested '{}'",
                            stored_model, model
                        ))
                        .into());
                    }
                }
                None => {
                    Self::set_metadata_tx(&tx, "embedding_model", model).await?;
                }
            }
        }

        tx.commit().await?;
        Ok(())
    }

    async fn setup_schema(&self) -> Result<()> {
        // Create Document storage
        self.conn
            .execute(&self.schema.create_documents_table, ())
            .await?;

        // Create Vector storage
        self.conn
            .execute(&self.schema.create_vectors_table, ())
            .await?;

        // Create the Vector Index (DiskANN) for speed
        self.conn
            .execute(&self.schema.create_index_table, ())
            .await?;

        Ok(())
    }

    /// Stores a document with its embeddings and chunk offsets.
    ///
    /// # Arguments
    ///
    /// * `content` - The full document text
    /// * `metadata` - Document metadata as JSON
    /// * `embeddings` - Vector of (embedding, chunk_start, chunk_end) tuples
    ///
    /// # Returns
    ///
    /// The assigned document ID.
    ///
    /// # Errors
    ///
    /// Returns an error if database insertion fails.
    pub async fn insert_document(
        &self,
        content: &str,
        metadata: Value,
        embeddings: Vec<(Vec<f32>, usize, usize)>,
    ) -> Result<i64> {
        self.insert_document_with_progress(content, metadata, embeddings, None::<fn(usize)>)
            .await
    }

    /// Stores a document with progress tracking during database insertion.
    ///
    /// Similar to [`insert_document`](Self::insert_document), but invokes a callback
    /// after each chunk is inserted, useful for displaying progress to users.
    ///
    /// # Arguments
    ///
    /// * `content` - The full document text
    /// * `metadata` - Document metadata as JSON
    /// * `embeddings` - Vector of (embedding, chunk_start, chunk_end) tuples
    /// * `progress_callback` - Optional callback invoked with cumulative chunk count
    ///
    /// # Returns
    ///
    /// The assigned document ID.
    ///
    /// # Errors
    ///
    /// Returns an error if database insertion fails.
    pub async fn insert_document_with_progress<F>(
        &self,
        content: &str,
        metadata: Value,
        embeddings: Vec<(Vec<f32>, usize, usize)>,
        mut progress_callback: Option<F>,
    ) -> Result<i64>
    where
        F: FnMut(usize),
    {
        // Compress content using lz4_flex inside spawn_blocking
        let bytes = content.as_bytes().to_vec();
        let compressed_content = self
            .executor
            .spawn_blocking(move || -> Result<Vec<u8>> {
                use std::io::Write;
                let mut enc = lz4_flex::frame::FrameEncoder::new(Vec::new());
                enc.write_all(&bytes)?;
                enc.finish().map_err(|e| std::io::Error::other(e).into())
            })
            .await?;

        // Serialize vectors to JSON in parallel before entering the transaction
        let indexed: Vec<(usize, Vec<f32>, usize, usize)> = embeddings
            .into_iter()
            .enumerate()
            .map(|(i, (v, s, e))| (i, v, s, e))
            .collect();
        let serialized: Vec<Result<(usize, String, usize, usize)>> =
            self.executor
                .par_map(indexed, |(i, vec, chunk_start, chunk_end)| {
                    serde_json::to_string(&vec)
                        .map(|json| (i, json, chunk_start, chunk_end))
                        .map_err(Into::into)
                });

        // Wrap everything in a single transaction for atomicity
        let tx = self.conn.transaction().await?;

        // 1. Insert Parent Document with compressed content
        let doc_id: i64 = {
            let mut rows = tx
                .query(
                    &self.schema.insert_document,
                    (compressed_content, metadata.to_string()),
                )
                .await?;

            rows.next()
                .await?
                .ok_or_else(|| DatabaseError::InsertFailed("No ID returned".to_string()))?
                .get(0)?
        };

        // 2. Insert Vectors with offsets in same transaction
        for result in serialized {
            let (i, vec_json, chunk_start, chunk_end) = result?;
            tx.execute(
                &self.schema.insert_vectors,
                (
                    doc_id,
                    i as i64,
                    chunk_start as i64,
                    chunk_end as i64,
                    vec_json,
                ),
            )
            .await?;

            if let Some(ref mut callback) = progress_callback {
                callback(i + 1);
            }
        }

        tx.commit().await?;

        Ok(doc_id)
    }

    /// Searches for the top-k most similar document chunks using vector similarity.
    ///
    /// # Arguments
    ///
    /// * `query_vector` - The embedded query vector
    /// * `limit` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// A vector of tuples containing (doc_id, metadata, chunk_start, chunk_end)
    /// ordered by cosine similarity.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub async fn search_topk(
        &self,
        query_vector: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<(i64, Value, i64, i64)>> {
        let vec_str = serde_json::to_string(&query_vector)?;

        let mut rows = self
            .conn
            .query(&self.schema.search_topk, (vec_str, limit as i64))
            .await?;

        let mut results = Vec::new();
        while let Some(row) = rows.next().await? {
            let doc_id: i64 = row.get(0)?;
            let meta_str: String = row.get(1)?;
            let metadata: Value = serde_json::from_str(&meta_str)?;
            let chunk_start: i64 = row.get(2)?;
            let chunk_end: i64 = row.get(3)?;

            results.push((doc_id, metadata, chunk_start, chunk_end));
        }

        Ok(results)
    }

    /// Searches for the top-k most similar document chunks with cosine distances.
    ///
    /// # Arguments
    ///
    /// * `query_vector` - The embedded query vector
    /// * `limit` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// A vector of tuples containing (doc_id, metadata, distance, chunk_start, chunk_end).
    /// Lower distances indicate higher similarity.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub async fn search_topk_with_distance(
        &self,
        query_vector: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<(i64, Value, f32, i64, i64)>> {
        let vec_str = serde_json::to_string(&query_vector)?;

        let mut rows = self
            .conn
            .query(
                &self.schema.search_topk_with_distance,
                (vec_str, limit as i64),
            )
            .await?;

        let mut results = Vec::new();
        while let Some(row) = rows.next().await? {
            let doc_id: i64 = row.get(0)?;
            let meta_str: String = row.get(1)?;
            let metadata: Value = serde_json::from_str(&meta_str)?;

            // Handle NULL distances (can happen with zero vectors)
            let distance: f32 = match row.get::<Option<f64>>(2)? {
                Some(d) => d as f32,
                None => f32::MAX, // Put NULL distances at the end
            };

            let chunk_start: i64 = row.get(3)?;
            let chunk_end: i64 = row.get(4)?;

            results.push((doc_id, metadata, distance, chunk_start, chunk_end));
        }

        Ok(results)
    }

    /// Searches for the top-k most similar document chunks, including their embedding vectors.
    ///
    /// # Arguments
    ///
    /// * `query_vector` - The embedded query vector
    /// * `limit` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// A vector of tuples containing (doc_id, metadata, embedding_vector, chunk_start, chunk_end).
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub async fn search_topk_with_vectors(
        &self,
        query_vector: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<(i64, Value, Vec<f32>, i64, i64)>> {
        let vec_str = serde_json::to_string(&query_vector)?;

        let mut rows = self
            .conn
            .query(
                &self.schema.search_topk_with_vectors,
                (vec_str, limit as i64),
            )
            .await?;

        let mut results = Vec::new();
        while let Some(row) = rows.next().await? {
            let doc_id: i64 = row.get(0)?;
            let meta_str: String = row.get(1)?;
            let metadata: Value = serde_json::from_str(&meta_str)?;
            let embedding: Vec<u8> = row.get(2)?;

            // Convert F32_BLOB bytes back to Vec<f32>
            let vec_f32: Vec<f32> = embedding
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            let chunk_start: i64 = row.get(3)?;
            let chunk_end: i64 = row.get(4)?;

            results.push((doc_id, metadata, vec_f32, chunk_start, chunk_end));
        }

        Ok(results)
    }

    /// Retrieves a document by its ID.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The document ID to retrieve
    ///
    /// # Returns
    ///
    /// Returns `Some((content, metadata))` if found, or `None` if the document does not exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub async fn fetch_document(&self, doc_id: i64) -> Result<Option<(String, Value)>> {
        let mut rows = self
            .conn
            .query(&self.schema.fetch_document, [doc_id])
            .await?;

        if let Some(row) = rows.next().await? {
            // Decompress content using lz4_flex inside spawn_blocking
            let compressed_content: Vec<u8> = row.get(0)?;
            let content = self
                .executor
                .spawn_blocking(move || -> Result<String> {
                    use std::io::Read;
                    let mut dec = lz4_flex::frame::FrameDecoder::new(&compressed_content[..]);
                    let mut decompressed = Vec::new();
                    dec.read_to_end(&mut decompressed)?;
                    String::from_utf8(decompressed).map_err(Into::into)
                })
                .await?;

            let meta_str: String = row.get(1)?;
            let metadata: Value = serde_json::from_str(&meta_str)?;
            Ok(Some((content, metadata)))
        } else {
            Ok(None)
        }
    }

    #[cfg(test)]
    async fn get_journal_mode(&self) -> Result<String> {
        let mut rows = self.conn.query("PRAGMA journal_mode", ()).await?;
        if let Some(row) = rows.next().await? {
            Ok(row.get(0)?)
        } else {
            Err(crate::error::DatabaseError::QueryFailed(
                "PRAGMA journal_mode returned no results".to_string(),
            )
            .into())
        }
    }
}

#[derive(Debug)]
pub struct VectorDatabase {
    db: Database,
    schema: Arc<VectorDatabaseSchema>,
    executor: Executor,
}

impl VectorDatabase {
    /// Creates a new vector database at the specified path.
    ///
    /// # Arguments
    ///
    /// * `db_path` - Path to the database file, or ":memory:" for in-memory database
    /// * `vec_size` - Dimensionality of the embedding vectors
    /// * `executor` - Executor for CPU-bound parallel work
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be created or opened.
    pub async fn new(db_path: &str, vec_size: usize, executor: Executor) -> Result<Self> {
        let db = Builder::new_local(db_path).build().await?;
        let schema = Arc::new(VectorDatabaseSchema::new(vec_size));
        Ok(VectorDatabase {
            db,
            schema,
            executor,
        })
    }

    /// Creates a new connection to the vector database.
    ///
    /// Multiple connections can share the same database for concurrent access.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection cannot be established.
    pub async fn connect(&self) -> Result<VectorDatabaseConnection> {
        self.connect_with_model(None).await
    }

    /// Creates a new connection and validates embedding metadata against the model.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection cannot be established or metadata mismatches.
    pub async fn connect_with_model(
        &self,
        expected_model: Option<&str>,
    ) -> Result<VectorDatabaseConnection> {
        let conn = self.db.connect()?;
        VectorDatabaseConnection::new(conn, self.schema.clone(), self.executor, expected_model)
            .await
    }

    /// Reads the stored embedding dimension from the database metadata table.
    ///
    /// Returns `Ok(None)` when the metadata table exists but no dimension is stored yet.
    pub async fn read_stored_embedding_dim(db_path: &str) -> Result<Option<usize>> {
        let db = Builder::new_local(db_path).build().await?;
        let conn = db.connect()?;
        VectorDatabaseConnection::setup_metadata_table(&conn).await?;

        if let Some(raw_dim) =
            VectorDatabaseConnection::get_metadata(&conn, "embedding_dim").await?
        {
            let dim = VectorDatabaseConnection::parse_embedding_dim(&raw_dim)?;
            Ok(Some(dim))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod db_tests {
    use super::*;
    use serde_json::json;

    fn seq() -> Executor {
        Executor::sequential()
    }

    // Helper function to add dummy offsets to embeddings for testing
    fn add_offsets(embeddings: Vec<Vec<f32>>) -> Vec<(Vec<f32>, usize, usize)> {
        embeddings
            .into_iter()
            .enumerate()
            .map(|(i, vec)| (vec, i * 100, (i + 1) * 100))
            .collect()
    }

    #[tokio::test]
    async fn test_full_flow() -> Result<()> {
        // 1. Initialize an in-memory database
        // ":memory:" ensures no files are created on disk during testing
        let db = VectorDatabase::new(":memory:", 3, seq()).await?;
        let conn = db.connect().await?;

        // 2. Prepare mock data
        let content = "The quick brown fox jumps over the lazy dog.";
        let metadata = json!({
            "source": "test_suite",
            "author": "rust_bot"
        });
        // Mock 3-dimensional embeddings (1 chunk)
        let embeddings = add_offsets(vec![vec![1.0, 0.0, 0.0]]);

        // 3. Test Insertion
        let doc_id = conn
            .insert_document(content, metadata.clone(), embeddings)
            .await?;
        assert!(doc_id > 0);

        // 4. Test Search
        // Querying with something very similar to our [1.0, 0.0, 0.0]
        let query_vec = vec![0.9, 0.1, 0.0];
        let results = conn.search_topk_with_distance(query_vec, 1).await?;

        assert_eq!(results.len(), 1);
        let (found_doc_id, found_meta, distance, _, _) = &results[0];

        assert_eq!(*found_doc_id, doc_id);
        assert_eq!(found_meta["author"], "rust_bot");
        assert!(*distance < 0.2); // Cosine distance should be small

        // Verify content by fetching the document
        let (fetched_content, _) = conn.fetch_document(*found_doc_id).await?.unwrap();
        assert_eq!(fetched_content, content);

        Ok(())
    }

    #[tokio::test]
    async fn test_embedding_dimension_mismatch_rejected() -> Result<()> {
        let db_uri = "file:dim_mismatch?mode=memory&cache=shared";

        let db_3d = VectorDatabase::new(db_uri, 3, seq()).await?;
        let _conn_3d = db_3d.connect().await?;

        let db_4d = VectorDatabase::new(db_uri, 4, seq()).await?;
        let err = db_4d
            .connect()
            .await
            .expect_err("dimension mismatch should fail");

        assert!(
            err.to_string().contains("Embedding dimension mismatch"),
            "unexpected error: {}",
            err
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_embedding_model_mismatch_rejected() -> Result<()> {
        let db_uri = "file:model_mismatch?mode=memory&cache=shared";

        let db = VectorDatabase::new(db_uri, 3, seq()).await?;
        let _conn = db.connect_with_model(Some("model-a")).await?;

        let err = db
            .connect_with_model(Some("model-b"))
            .await
            .expect_err("model mismatch should fail");

        assert!(
            err.to_string().contains("Embedding model mismatch"),
            "unexpected error: {}",
            err
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_legacy_dimension_mismatch_rejected() -> Result<()> {
        let db_uri = "file:legacy_dim_mismatch?mode=memory&cache=shared";

        // Simulate an older database that has dimension-namespaced tables but no metadata table.
        let legacy_db = Builder::new_local(db_uri).build().await?;
        let legacy_conn = legacy_db.connect()?;
        legacy_conn
            .execute(
                "CREATE TABLE IF NOT EXISTS documents_3 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content BLOB,
                    metadata JSON
                )",
                (),
            )
            .await?;

        let db_4d = VectorDatabase::new(db_uri, 4, seq()).await?;
        let err = db_4d
            .connect()
            .await
            .expect_err("legacy dimension mismatch should fail");

        assert!(
            err.to_string()
                .contains("Legacy database uses embedding dimension 3"),
            "unexpected error: {}",
            err
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_cascade_delete() -> Result<()> {
        let db = VectorDatabase::new(":memory:", 2, seq()).await?;
        let conn = db.connect().await?;

        // Insert a doc with 2 chunks
        let doc_id = conn
            .insert_document(
                "Delete me",
                json!({}),
                add_offsets(vec![vec![1.0, 1.0], vec![0.0, 0.0]]),
            )
            .await?;

        // Manually delete the parent document
        conn.conn
            .execute("DELETE FROM documents_2 WHERE id = ?1", [doc_id])
            .await?;

        // Verify that embeddings are gone (Cascade working)
        let mut rows = conn
            .conn
            .query("SELECT COUNT(*) FROM document_embeddings_2", ())
            .await?;
        let count: i64 = rows.next().await?.unwrap().get(0)?;

        assert_eq!(count, 0, "Vectors should have been deleted by cascade");

        Ok(())
    }

    #[tokio::test]
    async fn test_multiple_documents_search_ordering() -> Result<()> {
        let db = VectorDatabase::new(":memory:", 3, seq()).await?;
        let conn = db.connect().await?;

        // Insert 3 documents with different similarity to our query
        // Doc 1: Very similar to [1, 0, 0]
        conn.insert_document(
            "Document 1",
            json!({"id": 1}),
            add_offsets(vec![vec![0.99, 0.01, 0.0]]),
        )
        .await?;

        // Doc 2: Orthogonal to [1, 0, 0]
        conn.insert_document(
            "Document 2",
            json!({"id": 2}),
            add_offsets(vec![vec![0.0, 1.0, 0.0]]),
        )
        .await?;

        // Doc 3: Somewhat similar to [1, 0, 0]
        conn.insert_document(
            "Document 3",
            json!({"id": 3}),
            add_offsets(vec![vec![0.7, 0.7, 0.0]]),
        )
        .await?;

        // Search with [1, 0, 0]
        let results = conn
            .search_topk_with_distance(vec![1.0, 0.0, 0.0], 3)
            .await?;

        assert_eq!(results.len(), 3);

        // Results should be ordered by similarity
        assert_eq!(results[0].1["id"], 1); // Most similar
        assert_eq!(results[1].1["id"], 3); // Medium similarity
        assert_eq!(results[2].1["id"], 2); // Least similar

        // Check distances are in ascending order
        assert!(results[0].2 < results[1].2);
        assert!(results[1].2 < results[2].2);

        Ok(())
    }

    #[tokio::test]
    async fn test_search_empty_database() -> Result<()> {
        let db = VectorDatabase::new(":memory:", 3, seq()).await?;
        let conn = db.connect().await?;

        let results = conn
            .search_topk_with_distance(vec![1.0, 0.0, 0.0], 10)
            .await?;

        assert_eq!(results.len(), 0, "Empty database should return no results");

        Ok(())
    }

    #[tokio::test]
    async fn test_multiple_chunks_per_document() -> Result<()> {
        let db = VectorDatabase::new(":memory:", 4, seq()).await?;
        let conn = db.connect().await?;

        // Insert a document with 5 chunks
        let chunks = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.5, 0.5, 0.0, 0.0],
        ];

        let doc_id = conn
            .insert_document(
                "Multi-chunk document",
                json!({"chunks": 5}),
                add_offsets(chunks),
            )
            .await?;

        assert!(doc_id > 0);

        // Search should find the most similar chunk
        let results = conn
            .search_topk_with_distance(vec![0.95, 0.05, 0.0, 0.0], 1)
            .await?;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, doc_id);

        // Search with higher limit should return all chunks from the same document
        let results = conn
            .search_topk_with_distance(vec![1.0, 0.0, 0.0, 0.0], 10)
            .await?;

        assert_eq!(results.len(), 5, "Should find all 5 chunks");

        // All results should be from the same document
        for result in &results {
            assert_eq!(result.0, doc_id);
            assert_eq!(result.1["chunks"], 5);
        }

        // Verify content by fetching the document
        let (fetched_content, _) = conn.fetch_document(doc_id).await?.unwrap();
        assert_eq!(fetched_content, "Multi-chunk document");

        Ok(())
    }

    #[tokio::test]
    async fn test_search_limit_parameter() -> Result<()> {
        let db = VectorDatabase::new(":memory:", 2, seq()).await?;
        let conn = db.connect().await?;

        // Insert 10 documents
        for i in 0..10 {
            conn.insert_document(
                &format!("Document {}", i),
                json!({"index": i}),
                add_offsets(vec![vec![i as f32 / 10.0, 1.0 - i as f32 / 10.0]]),
            )
            .await?;
        }

        // Test different limits
        let results_1 = conn.search_topk_with_distance(vec![0.0, 1.0], 1).await?;
        assert_eq!(results_1.len(), 1);

        let results_5 = conn.search_topk_with_distance(vec![0.0, 1.0], 5).await?;
        assert_eq!(results_5.len(), 5);

        let results_all = conn.search_topk_with_distance(vec![0.0, 1.0], 100).await?;
        assert_eq!(
            results_all.len(),
            10,
            "Should return all available documents"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_complex_metadata() -> Result<()> {
        let db = VectorDatabase::new(":memory:", 3, seq()).await?;
        let conn = db.connect().await?;

        let complex_metadata = json!({
            "title": "Research Paper",
            "authors": ["Alice", "Bob", "Charlie"],
            "year": 2024,
            "tags": ["machine-learning", "embeddings", "vector-search"],
            "stats": {
                "citations": 42,
                "views": 1337
            },
            "available": true
        });

        let doc_id = conn
            .insert_document(
                "Paper content",
                complex_metadata.clone(),
                add_offsets(vec![vec![1.0, 0.0, 0.0]]),
            )
            .await?;

        assert!(doc_id > 0);

        let results = conn
            .search_topk_with_distance(vec![1.0, 0.0, 0.0], 1)
            .await?;

        assert_eq!(results.len(), 1);
        let (_, metadata, _, _, _) = &results[0];

        assert_eq!(metadata["title"], "Research Paper");
        assert_eq!(metadata["authors"][0], "Alice");
        assert_eq!(metadata["stats"]["citations"], 42);
        assert_eq!(metadata["available"], true);

        Ok(())
    }

    #[tokio::test]
    async fn test_realistic_embedding_dimensions() -> Result<()> {
        // Test with common embedding dimensions (384 from sentence-transformers)
        let db = VectorDatabase::new(":memory:", 384, seq()).await?;
        let conn = db.connect().await?;

        let mut embedding = vec![0.0; 384];
        embedding[0] = 1.0; // Set first dimension
        embedding[100] = 0.5; // Set some other dimensions
        embedding[200] = 0.3;

        let doc_id = conn
            .insert_document(
                "Large embedding doc",
                json!({}),
                add_offsets(vec![embedding.clone()]),
            )
            .await?;

        assert!(doc_id > 0);

        let results = conn.search_topk_with_distance(embedding, 1).await?;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, doc_id);
        assert!(results[0].2 < 0.01); // Should be very close (almost identical)

        // Verify content by fetching the document
        let (fetched_content, _) = conn.fetch_document(doc_id).await?.unwrap();
        assert_eq!(fetched_content, "Large embedding doc");

        Ok(())
    }

    #[tokio::test]
    async fn test_zero_vectors() -> Result<()> {
        let db = VectorDatabase::new(":memory:", 3, seq()).await?;
        let conn = db.connect().await?;

        // Insert a document with a zero vector
        let doc_id = conn
            .insert_document(
                "Zero vector doc",
                json!({}),
                add_offsets(vec![vec![0.0, 0.0, 0.0]]),
            )
            .await?;

        assert!(doc_id > 0);

        // Search with a non-zero vector
        let results = conn
            .search_topk_with_distance(vec![1.0, 0.0, 0.0], 1)
            .await?;

        // Should still return the result (cosine distance handles zero vectors)
        assert_eq!(results.len(), 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_duplicate_documents() -> Result<()> {
        let db = VectorDatabase::new(":memory:", 3, seq()).await?;
        let conn = db.connect().await?;

        // Insert the same content twice
        let content = "Duplicate content";
        let embedding = add_offsets(vec![vec![1.0, 0.0, 0.0]]);

        let doc_id_1 = conn
            .insert_document(content, json!({"version": 1}), embedding.clone())
            .await?;

        let doc_id_2 = conn
            .insert_document(content, json!({"version": 2}), embedding.clone())
            .await?;

        assert_ne!(doc_id_1, doc_id_2, "Should create separate documents");

        // Search should find both
        let results = conn
            .search_topk_with_distance(vec![1.0, 0.0, 0.0], 10)
            .await?;

        assert_eq!(results.len(), 2);

        // Both results should have different doc IDs
        assert_ne!(results[0].0, results[1].0);

        // But both should have the same content
        let (content_1, _) = conn.fetch_document(results[0].0).await?.unwrap();
        let (content_2, _) = conn.fetch_document(results[1].0).await?.unwrap();
        assert_eq!(content_1, content);
        assert_eq!(content_2, content);

        Ok(())
    }

    #[tokio::test]
    async fn test_normalized_vs_unnormalized_vectors() -> Result<()> {
        let db = VectorDatabase::new(":memory:", 3, seq()).await?;
        let conn = db.connect().await?;

        // Insert normalized vector
        conn.insert_document(
            "Normalized",
            json!({"type": "normalized"}),
            add_offsets(vec![vec![1.0, 0.0, 0.0]]),
        )
        .await?;

        // Insert unnormalized vector (same direction, different magnitude)
        conn.insert_document(
            "Unnormalized",
            json!({"type": "unnormalized"}),
            add_offsets(vec![vec![10.0, 0.0, 0.0]]),
        )
        .await?;

        // Cosine distance should treat them similarly (direction matters, not magnitude)
        let results = conn
            .search_topk_with_distance(vec![1.0, 0.0, 0.0], 2)
            .await?;

        assert_eq!(results.len(), 2);

        // Both should have very low distance (cosine is about direction)
        assert!(results[0].2 < 0.01);
        assert!(results[1].2 < 0.01);

        Ok(())
    }

    #[tokio::test]
    async fn test_empty_embeddings_list() -> Result<()> {
        let db = VectorDatabase::new(":memory:", 3, seq()).await?;
        let conn = db.connect().await?;

        // Insert a document with no embeddings
        let doc_id = conn
            .insert_document("No embeddings", json!({}), vec![])
            .await?;

        assert!(
            doc_id > 0,
            "Should successfully insert document even with no embeddings"
        );

        // Search should not find this document (no vectors to match)
        let results = conn
            .search_topk_with_distance(vec![1.0, 0.0, 0.0], 10)
            .await?;

        assert_eq!(results.len(), 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_all_search_functions_consistency() -> Result<()> {
        let db = VectorDatabase::new(":memory:", 3, seq()).await?;
        let conn = db.connect().await?;

        // Insert 5 documents with different similarity to query [1, 0, 0]
        let doc_ids = [
            conn.insert_document(
                "Doc 1",
                json!({"name": "doc1"}),
                add_offsets(vec![vec![0.99, 0.01, 0.0]]),
            )
            .await?,
            conn.insert_document(
                "Doc 2",
                json!({"name": "doc2"}),
                add_offsets(vec![vec![0.0, 1.0, 0.0]]),
            )
            .await?,
            conn.insert_document(
                "Doc 3",
                json!({"name": "doc3"}),
                add_offsets(vec![vec![0.7, 0.7, 0.0]]),
            )
            .await?,
            conn.insert_document(
                "Doc 4",
                json!({"name": "doc4"}),
                add_offsets(vec![vec![1.0, 0.0, 0.0]]),
            )
            .await?,
            conn.insert_document(
                "Doc 5",
                json!({"name": "doc5"}),
                add_offsets(vec![vec![0.5, 0.5, 0.5]]),
            )
            .await?,
        ];

        let query = vec![1.0, 0.0, 0.0];
        let k = 3;

        // Call all three search functions
        let results_basic = conn.search_topk(query.clone(), k).await?;
        let results_distance = conn.search_topk_with_distance(query.clone(), k).await?;
        let results_vectors = conn.search_topk_with_vectors(query.clone(), k).await?;

        // All should return same number of results
        assert_eq!(results_basic.len(), k);
        assert_eq!(results_distance.len(), k);
        assert_eq!(results_vectors.len(), k);

        // Verify all three return the same doc_ids in the same order
        for i in 0..k {
            let doc_id_basic = results_basic[i].0;
            let doc_id_distance = results_distance[i].0;
            let doc_id_vectors = results_vectors[i].0;

            assert_eq!(
                doc_id_basic, doc_id_distance,
                "search_topk and search_topk_with_distance returned different doc_id at position {}",
                i
            );
            assert_eq!(
                doc_id_basic, doc_id_vectors,
                "search_topk and search_topk_with_vectors returned different doc_id at position {}",
                i
            );

            // Verify metadata is the same across all three
            assert_eq!(
                results_basic[i].1, results_distance[i].1,
                "Metadata mismatch between search_topk and search_topk_with_distance at position {}",
                i
            );
            assert_eq!(
                results_basic[i].1, results_vectors[i].1,
                "Metadata mismatch between search_topk and search_topk_with_vectors at position {}",
                i
            );
        }

        // Verify distances are in ascending order (most similar first)
        for i in 0..results_distance.len() - 1 {
            assert!(
                results_distance[i].2 <= results_distance[i + 1].2,
                "Distances not in ascending order: {} > {}",
                results_distance[i].2,
                results_distance[i + 1].2
            );
        }

        // Verify vectors have correct dimensions
        for (_, _, vec, _, _) in &results_vectors {
            assert_eq!(vec.len(), 3, "Vector should have 3 dimensions");
        }

        // Verify the most similar document is doc 4 (exact match [1, 0, 0])
        assert_eq!(results_basic[0].0, doc_ids[3]);
        assert_eq!(results_basic[0].1["name"], "doc4");

        // Second most similar should be doc 1 ([0.99, 0.01, 0])
        assert_eq!(results_basic[1].0, doc_ids[0]);
        assert_eq!(results_basic[1].1["name"], "doc1");

        Ok(())
    }

    #[tokio::test]
    async fn test_concurrent_inserts() -> Result<()> {
        let db = VectorDatabase::new(":memory:", 3, seq()).await?;
        let conn = db.connect().await?;

        // Insert multiple documents sequentially (simulating concurrent operations)
        let mut doc_ids = Vec::new();
        for i in 0..5 {
            let id = conn
                .insert_document(
                    &format!("Concurrent doc {}", i),
                    json!({"index": i}),
                    add_offsets(vec![vec![i as f32 / 5.0, 0.0, 0.0]]),
                )
                .await?;
            doc_ids.push(id);
        }

        assert_eq!(doc_ids.len(), 5);

        // All IDs should be unique
        let mut unique_ids = doc_ids.clone();
        unique_ids.sort();
        unique_ids.dedup();
        assert_eq!(unique_ids.len(), 5);

        // Search should find all documents
        let results = conn
            .search_topk_with_distance(vec![0.5, 0.0, 0.0], 10)
            .await?;
        assert_eq!(results.len(), 5);

        Ok(())
    }

    #[tokio::test]
    async fn test_wal_mode_enabled() -> Result<()> {
        // Test that WAL mode is enabled on file-based databases
        let temp_dir = tempfile::tempdir()?;
        let db_path = temp_dir.path().join("test_wal.db");
        let db = VectorDatabase::new(db_path.to_str().unwrap(), 3, seq()).await?;
        let conn = db.connect().await?;

        // Query the journal_mode to verify WAL is enabled
        let mode = conn.get_journal_mode().await?;
        assert_eq!(
            mode.to_lowercase(),
            "wal",
            "Expected journal_mode to be 'wal' but got '{}'",
            mode
        );

        Ok(())
    }
}

#[cfg(test)]
mod parallel_executor_tests {
    use super::*;
    use crate::executor::Executor;
    use serde_json::json;

    fn par() -> Executor {
        Executor::parallel()
    }

    fn add_offsets(embeddings: Vec<Vec<f32>>) -> Vec<(Vec<f32>, usize, usize)> {
        embeddings
            .into_iter()
            .enumerate()
            .map(|(i, vec)| (vec, i * 100, (i + 1) * 100))
            .collect()
    }

    #[tokio::test]
    async fn test_parallel_insert_and_fetch() -> Result<()> {
        let db = VectorDatabase::new(":memory:", 3, par()).await?;
        let conn = db.connect().await?;

        let content = "Parallel executor test document";
        let metadata = json!({"executor": "parallel"});
        let embeddings = add_offsets(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]);

        let doc_id = conn
            .insert_document(content, metadata.clone(), embeddings)
            .await?;
        assert!(doc_id > 0);

        let (fetched_content, fetched_meta) = conn.fetch_document(doc_id).await?.unwrap();
        assert_eq!(fetched_content, content);
        assert_eq!(fetched_meta["executor"], "parallel");

        Ok(())
    }

    #[tokio::test]
    async fn test_parallel_search() -> Result<()> {
        let db = VectorDatabase::new(":memory:", 3, par()).await?;
        let conn = db.connect().await?;

        for i in 0..5 {
            conn.insert_document(
                &format!("Document {}", i),
                json!({"index": i}),
                add_offsets(vec![vec![i as f32 / 5.0, 1.0 - i as f32 / 5.0, 0.0]]),
            )
            .await?;
        }

        let results = conn
            .search_topk_with_distance(vec![0.0, 1.0, 0.0], 3)
            .await?;
        assert_eq!(results.len(), 3);

        // Distances must be in ascending order
        for i in 0..results.len() - 1 {
            assert!(results[i].2 <= results[i + 1].2);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_parallel_large_embedding() -> Result<()> {
        // Exercise spawn_blocking and par_map with a realistic embedding size
        let db = VectorDatabase::new(":memory:", 768, par()).await?;
        let conn = db.connect().await?;

        let embedding: Vec<f32> = (0..768).map(|i| (i as f32) / 768.0).collect();
        let embeddings = add_offsets(vec![embedding.clone(), {
            let mut v = embedding.clone();
            v[0] = 0.5;
            v
        }]);

        let doc_id = conn
            .insert_document("Large parallel doc", json!({}), embeddings)
            .await?;
        assert!(doc_id > 0);

        let results = conn.search_topk(embedding, 1).await?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, doc_id);

        let (content, _) = conn.fetch_document(doc_id).await?.unwrap();
        assert_eq!(content, "Large parallel doc");

        Ok(())
    }

    #[tokio::test]
    async fn test_parallel_many_chunks() -> Result<()> {
        // Many chunks exercises par_map with non-trivial parallelism
        let db = VectorDatabase::new(":memory:", 4, par()).await?;
        let conn = db.connect().await?;

        let chunks: Vec<Vec<f32>> = (0..50)
            .map(|i| {
                let mut v = vec![0.0f32; 4];
                v[i % 4] = 1.0;
                v
            })
            .collect();

        let doc_id = conn
            .insert_document(
                "Many chunks doc",
                json!({"chunks": 50}),
                add_offsets(chunks),
            )
            .await?;
        assert!(doc_id > 0);

        let results = conn.search_topk(vec![1.0, 0.0, 0.0, 0.0], 10).await?;
        assert!(!results.is_empty());
        assert!(results.iter().all(|(id, _, _, _)| *id == doc_id));

        Ok(())
    }
}
