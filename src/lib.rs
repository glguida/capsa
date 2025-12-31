pub mod config;
pub mod documentdb;
pub mod embedder;
pub mod vectordb;

#[cfg(test)]
pub mod test_utils {
    use crate::embedder::EmbeddingInterface;
    use anyhow::Result;
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
                return Err(anyhow::anyhow!("Empty input not allowed"));
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
