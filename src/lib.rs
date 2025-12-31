//! Capsa - A compact, lightweight library for embedding-based document storage and retrieval.
//!
//! This library provides the core functionality for implementing RAG (Retrieval-Augmented
//! Generation) systems. It handles document chunking, embedding generation, vector storage,
//! and semantic search.
//!
//! # Quick Start
//!
//! ```no_run
//! use capsa::{config::Config, documentdb::DocumentDatabase};
//! use serde_json::json;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Configure embedding service and database
//!     let config = Config::new(
//!         "http://localhost:9000/v1".to_string(),
//!         "nomic-ai/nomic-embed-text-v1.5".to_string(),
//!         "./documents.db".to_string(),
//!         None,
//!     );
//!
//!     // Connect to database
//!     let db = DocumentDatabase::new(&config).await?;
//!     let conn = db.connect().await?;
//!
//!     // Index a document
//!     let doc_id = conn.insert(
//!         json!({"title": "Example Document"}),
//!         "Your document text here"
//!     ).await?;
//!
//!     // Search for similar content
//!     let results = conn.search_topk("your search query", 5).await?;
//!     for (doc_id, metadata, start, end) in results {
//!         println!("Found match in document {}: bytes {}-{}", doc_id, start, end);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! # Architecture
//!
//! The library is organized into several modules:
//!
//! - [`config`] - Configuration types for embedding services and databases
//! - [`documentdb`] - High-level document storage and retrieval API
//! - [`embedder`] - Text embedding generation and chunking
//! - [`vectordb`] - Low-level vector database operations
//!
//! Most applications should use [`documentdb`] which provides automatic embedding
//! generation. Use [`vectordb`] directly only if you need fine-grained control
//! over vector storage.

pub mod config;
pub mod documentdb;
pub mod embedder;
pub mod error;
pub mod vectordb;

#[cfg(test)]
pub mod test_utils {
    use crate::embedder::EmbeddingInterface;
    use crate::error::{EmbeddingError, Result};
    use async_trait::async_trait;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    /// Mock embedding implementation for testing.
    ///
    /// Generates deterministic fake embeddings based on the input text hash.
    /// This allows tests to run without requiring a real embedding server.
    pub struct MockEmbedding {
        embedding_size: usize,
    }

    impl MockEmbedding {
        pub fn new(embedding_size: usize) -> Self {
            Self { embedding_size }
        }
    }

    #[async_trait]
    impl EmbeddingInterface for MockEmbedding {
        async fn embed_raw(&self, input: &str) -> Result<Vec<f32>> {
            if input.is_empty() {
                return Err(
                    EmbeddingError::InvalidInput("Empty input not allowed".to_string()).into(),
                );
            }

            // Generate a deterministic embedding based on input hash
            let mut hasher = DefaultHasher::new();
            input.hash(&mut hasher);
            let hash = hasher.finish();

            // Create a simple embedding vector
            let mut embedding = vec![0.0; self.embedding_size];
            for (i, item) in embedding.iter_mut().enumerate() {
                // Use different parts of the hash for each dimension
                let val = ((hash.wrapping_mul((i + 1) as u64)) % 1000) as f32 / 1000.0;
                *item = val;
            }

            Ok(embedding)
        }

        async fn embed_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
            let mut embeddings = Vec::with_capacity(inputs.len());
            for input in inputs {
                let embedding = self.embed_raw(input).await?;
                embeddings.push(embedding);
            }
            Ok(embeddings)
        }
    }
}
