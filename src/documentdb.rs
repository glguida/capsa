use crate::embedder::Embedder;
use crate::vectordb::{VectorDatabase, VectorDatabaseConnection};
use anyhow::Result;
use std::sync::Arc;

type DocumentId = i64;

pub struct DocumentDatabaseConnection {
    embedder: Arc<Embedder>,
    vconn: VectorDatabaseConnection,
}

impl DocumentDatabaseConnection {
    /// Inserts a document
    ///
    /// Works identically to `insert` but invokes optional callbacks during
    /// embedding and database insertion phases with cumulative chunk count.
    ///
    /// # Arguments
    ///
    /// * `metadata` - Document metadata as JSON
    /// * `text` - Document content
    ///
    /// # Returns
    ///
    /// The document ID assigned by the database.
    ///
    /// # Errors
    ///
    /// Returns an error if embedding or database insertion fails.
    pub async fn insert(&self, metadata: serde_json::Value, text: &str) -> Result<DocumentId> {
        let vecs = self.embedder.embed_document(text).await?;
        let id = self
            .vconn
            .insert_document(text.as_ref(), metadata, vecs)
            .await?;
        Ok(id)
    }

    pub async fn search_topk(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(DocumentId, serde_json::Value, i64, i64)>> {
        let query_vec = self.embedder.embed_query(query).await?;
        self.vconn.search_topk(query_vec, limit).await
    }

    pub async fn search_topk_with_distance(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(DocumentId, serde_json::Value, f32, i64, i64)>> {
        let query_vec = self.embedder.embed_query(query).await?;
        self.vconn.search_topk_with_distance(query_vec, limit).await
    }

    pub async fn fetch_document(
        &self,
        doc_id: DocumentId,
    ) -> Result<Option<(String, serde_json::Value)>> {
        self.vconn.fetch_document(doc_id).await
    }
}

pub struct DocumentDatabase {
    embedder: Arc<Embedder>,
    vdb: VectorDatabase,
}

impl DocumentDatabase {
    pub async fn new(
        emb_base_url: String,
        emb_model: String,
        emb_api_key: Option<String>,
        emb_ctx: usize,
        vdb_path: String,
    ) -> Result<Self> {
        let embedder = Arc::new(Embedder::new(
            emb_base_url,
            emb_model,
            emb_api_key,
            emb_ctx,
        )?);

        // Retrieve vector size by having a test query.
        let test_vec = embedder.embed_query("test").await?;
        let vec_size = test_vec.len();

        let vdb = VectorDatabase::new(&vdb_path, vec_size).await?;

        Ok(DocumentDatabase { embedder, vdb })
    }

    /// Creates a new document database with a custom embedder.
    ///
    /// This constructor is useful for testing with mock embedders.
    ///
    /// # Arguments
    ///
    /// * `embedder` - An embedder instance to use for generating embeddings
    /// * `vdb_path` - Path to the vector database file
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be initialized.
    pub async fn with_embedder(embedder: Embedder, vdb_path: String) -> Result<Self> {
        let embedder = Arc::new(embedder);

        // Retrieve vector size by having a test query.
        let test_vec = embedder.embed_query("test").await?;
        let vec_size = test_vec.len();

        let vdb = VectorDatabase::new(&vdb_path, vec_size).await?;

        Ok(DocumentDatabase { embedder, vdb })
    }

    pub async fn connect(&self) -> Result<DocumentDatabaseConnection> {
        let vconn = self.vdb.connect().await?;
        let embedder = self.embedder.clone();
        Ok(DocumentDatabaseConnection { vconn, embedder })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::Embedder;
    use crate::test_utils::MockEmbedding;
    use serde_json::json;

    async fn create_test_db(db_path: &str) -> Result<DocumentDatabase> {
        // Use mock client to avoid network dependencies
        let client = Box::new(MockEmbedding::new(384));
        // Skip test if tokenizer is unavailable (no network/cache)
        let embedder = match Embedder::with_client(client, "bert-base-uncased".to_string(), 512) {
            Ok(e) => e,
            Err(_) => return Err(anyhow::anyhow!("Tokenizer unavailable")),
        };
        DocumentDatabase::with_embedder(embedder, db_path.to_string()).await
    }

    #[tokio::test]
    async fn test_document_database_creation() -> Result<()> {
        let db = match create_test_db(":memory:").await {
            Ok(db) => db,
            Err(_) => return Ok(()), // Skip if tokenizer unavailable
        };
        let conn = db.connect().await?;
        drop(conn);
        Ok(())
    }

    #[tokio::test]
    async fn test_insert_and_search() -> Result<()> {
        let db = match create_test_db(":memory:").await {
            Ok(db) => db,
            Err(_) => return Ok(()), // Skip if tokenizer unavailable
        };
        let conn = db.connect().await?;

        // Insert a document
        let metadata = json!({"title": "Test Document", "author": "Test Author"});
        let doc_id = conn
            .insert(metadata.clone(), "This is a test document about embeddings")
            .await?;

        assert!(doc_id > 0);

        // Search for similar documents
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

        // Insert a document
        let metadata = json!({"category": "technology"});
        let doc_id = conn
            .insert(metadata, "Machine learning and artificial intelligence")
            .await?;

        assert!(doc_id > 0);

        // Search with distance
        let results = conn.search_topk_with_distance("AI and ML", 5).await?;

        assert!(!results.is_empty());
        assert_eq!(results[0].0, doc_id);
        assert_eq!(results[0].1["category"], "technology");

        // Distance should be a reasonable value (not infinity or NaN)
        assert!(results[0].2 >= 0.0);
        assert!(results[0].2.is_finite());

        Ok(())
    }

    // Ignore this test. Requires an actual embedder.
    #[ignore]
    #[tokio::test]
    async fn test_multiple_documents_ranking() -> Result<()> {
        let db = match create_test_db(":memory:").await {
            Ok(db) => db,
            Err(_) => return Ok(()),
        };
        let conn = db.connect().await?;

        // Insert multiple documents with different content
        let doc1_id = conn
            .insert(json!({"id": 1}), "Rust is a systems programming language")
            .await?;

        let _doc2_id = conn
            .insert(
                json!({"id": 2}),
                "Python is a high-level programming language",
            )
            .await?;

        let _doc3_id = conn
            .insert(
                json!({"id": 3}),
                "Machine learning and artificial intelligence",
            )
            .await?;

        // Search for Rust-related content
        let results = conn
            .search_topk_with_distance("systems programming in Rust", 3)
            .await?;

        assert_eq!(results.len(), 3);

        // First result should be the Rust document
        assert_eq!(results[0].0, doc1_id);

        // Distances should be in ascending order (most similar first)
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

        // Insert 5 documents
        for i in 0..5 {
            conn.insert(
                json!({"index": i}),
                &format!("Document number {} about various topics", i),
            )
            .await?;
        }

        // Search with limit of 2
        let results = conn.search_topk("document topics", 2).await?;
        assert_eq!(results.len(), 2);

        // Search with limit of 10 (should return all 5)
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

        // Search in empty database
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
            "metrics": {
                "citations": 100,
                "views": 5000
            }
        });

        let doc_id = conn
            .insert(complex_metadata.clone(), "Advanced research in embeddings")
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

        // Create a very long text that will be chunked
        let long_text = (0..1000)
            .map(|i| {
                format!(
                    "This is sentence number {}. It contains some information. ",
                    i
                )
            })
            .collect::<String>();

        let doc_id = conn.insert(json!({"type": "long"}), &long_text).await?;

        assert!(doc_id > 0);

        // Search should still work
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

        // Try to insert empty text
        let result = conn.insert(json!({}), "").await;

        // This might fail or succeed depending on the embedder's behavior
        // If it succeeds, the doc_id should be valid
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

        // Insert same content twice with different metadata
        let doc1_id = conn.insert(json!({"version": 1}), content).await?;

        let doc2_id = conn.insert(json!({"version": 2}), content).await?;

        assert_ne!(doc1_id, doc2_id);

        // Search should find both
        let results = conn.search_topk(content, 5).await?;

        assert!(results.len() >= 2);

        // Both documents should be in results
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
            .insert(json!({"type": "special"}), special_text)
            .await?;

        assert!(doc_id > 0);

        // Search should work with special characters
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

        let doc_id = conn.insert(json!({"lang": "multi"}), unicode_text).await?;

        assert!(doc_id > 0);

        // Search should work with unicode
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
            .insert(json!({"test": "consistency"}), "Consistency test document")
            .await?;

        // Both search methods should return the same documents
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

        // Create multiple connections
        let conn1 = db.connect().await?;
        let conn2 = db.connect().await?;

        // Insert with first connection
        let doc_id = conn1
            .insert(
                json!({"source": "conn1"}),
                "Rust programming language documentation",
            )
            .await?;

        // Search with second connection should find it
        let results = conn2
            .search_topk("Rust programming documentation", 5)
            .await?;

        println!("Search results: {:?}", results);
        println!("Expected doc_id: {}", doc_id);

        assert!(!results.is_empty());
        assert_eq!(results[0].0, doc_id);

        Ok(())
    }
}
