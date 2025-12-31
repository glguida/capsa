//! Low-level vector database operations using libSQL.
//!
//! This module provides direct access to vector storage and similarity search
//! functionality. It manages document storage, embedding vectors, and vector
//! indices using libSQL's native vector support.
//!
//! For most use cases, prefer the higher-level [`documentdb`](crate::documentdb)
//! module which handles embedding generation automatically.

use crate::error::{DatabaseError, Result};
use libsql::{Builder, Connection, Database};
use serde_json::Value;
use std::sync::Arc;

#[derive(Debug)]
struct VectorDatabaseSchema {
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
                content TEXT,
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
}

impl VectorDatabaseConnection {
    /// Creates a new vector database connection and initializes the schema.
    ///
    /// This sets up the necessary tables and indices for vector storage if they
    /// do not already exist.
    async fn new(conn: Connection, schema: Arc<VectorDatabaseSchema>) -> Result<Self> {
        // Enable foreign keys for the ON DELETE CASCADE behavior
        conn.execute("PRAGMA foreign_keys = ON", ()).await?;
        // Enable WAL mode for better concurrency and performance
        conn.query("PRAGMA journal_mode = WAL", ()).await?;

        let db = Self { conn, schema };
        db.setup_schema().await?;

        Ok(db)
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
        // Wrap everything in a single transaction for atomicity
        let tx = self.conn.transaction().await?;

        // 1. Insert Parent Document
        let doc_id: i64 = {
            let mut rows = tx
                .query(
                    &self.schema.insert_document,
                    (content.to_string(), metadata.to_string()),
                )
                .await?;

            rows.next()
                .await?
                .ok_or_else(|| DatabaseError::InsertFailed("No ID returned".to_string()))?
                .get(0)?
        };

        // 2. Insert Vectors with offsets in same transaction
        for (i, (vec, chunk_start, chunk_end)) in embeddings.into_iter().enumerate() {
            let vec_json = serde_json::to_string(&vec)?;
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
            let content: String = row.get(0)?;
            let meta_str: String = row.get(1)?;
            let metadata: Value = serde_json::from_str(&meta_str)?;
            Ok(Some((content, metadata)))
        } else {
            Ok(None)
        }
    }
}

#[derive(Debug)]
pub struct VectorDatabase {
    db: Database,
    schema: Arc<VectorDatabaseSchema>,
}

impl VectorDatabase {
    /// Creates a new vector database at the specified path.
    ///
    /// # Arguments
    ///
    /// * `db_path` - Path to the database file, or ":memory:" for in-memory database
    /// * `vec_size` - Dimensionality of the embedding vectors
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be created or opened.
    pub async fn new(db_path: &str, vec_size: usize) -> Result<Self> {
        let db = Builder::new_local(db_path).build().await?;
        let schema = Arc::new(VectorDatabaseSchema::new(vec_size));
        Ok(VectorDatabase { db, schema })
    }

    /// Creates a new connection to the vector database.
    ///
    /// Multiple connections can share the same database for concurrent access.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection cannot be established.
    pub async fn connect(&self) -> Result<VectorDatabaseConnection> {
        let conn = self.db.connect()?;
        VectorDatabaseConnection::new(conn, self.schema.clone()).await
    }
}

#[cfg(test)]
mod db_tests {
    use super::*;
    use serde_json::json;

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
        let db = VectorDatabase::new(":memory:", 3).await?;
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
    async fn test_cascade_delete() -> Result<()> {
        let db = VectorDatabase::new(":memory:", 2).await?;
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
        let db = VectorDatabase::new(":memory:", 3).await?;
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
        let db = VectorDatabase::new(":memory:", 3).await?;
        let conn = db.connect().await?;

        let results = conn
            .search_topk_with_distance(vec![1.0, 0.0, 0.0], 10)
            .await?;

        assert_eq!(results.len(), 0, "Empty database should return no results");

        Ok(())
    }

    #[tokio::test]
    async fn test_multiple_chunks_per_document() -> Result<()> {
        let db = VectorDatabase::new(":memory:", 4).await?;
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
        let db = VectorDatabase::new(":memory:", 2).await?;
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
        let db = VectorDatabase::new(":memory:", 3).await?;
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
        let db = VectorDatabase::new(":memory:", 384).await?;
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
        let db = VectorDatabase::new(":memory:", 3).await?;
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
        let db = VectorDatabase::new(":memory:", 3).await?;
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
        let db = VectorDatabase::new(":memory:", 3).await?;
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
        let db = VectorDatabase::new(":memory:", 3).await?;
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
        let db = VectorDatabase::new(":memory:", 3).await?;
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
        let db = VectorDatabase::new(":memory:", 3).await?;
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
}
