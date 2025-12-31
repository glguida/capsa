//! Configuration constants for the embedding system.

/// The maximum context size in tokens for text embeddings.
///
/// This value is fixed at 128 tokens, which has been empirically determined
/// to provide excellent search results while maintaining good performance.
pub const EMBEDDING_CONTEXT: usize = 128;

/// Runtime configuration for the embedding system.
#[derive(Debug, Clone)]
pub struct Config {
    /// Base URL for the embedding API endpoint
    pub base_url: String,
    /// Name of the embedding model to use
    pub model: String,
    /// Path to the vector database file
    pub db_path: String,
    /// Optional API key for authentication
    pub api_key: Option<String>,
}

impl Config {
    /// Creates a new configuration.
    ///
    /// # Arguments
    ///
    /// * `base_url` - Base URL for the embedding API endpoint
    /// * `model` - Name of the embedding model to use
    /// * `db_path` - Path to the vector database file
    /// * `api_key` - Optional API key for authentication
    pub fn new(base_url: String, model: String, db_path: String, api_key: Option<String>) -> Self {
        Config {
            base_url,
            model,
            db_path,
            api_key,
        }
    }
}
