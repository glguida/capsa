//! Error types for the Capsa library.
//!
//! This module defines the error types used throughout the library, providing
//! idiomatic Rust error handling with context and proper error chaining.

use thiserror::Error;

/// The main error type for the Capsa library.
///
/// This enum covers all possible errors that can occur when using the library,
/// organized by their source or context.
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum CapsaError {
    /// Error occurred while generating embeddings
    #[error("Embedding error: {0}")]
    Embedding(#[from] EmbeddingError),

    /// Error occurred while interacting with the database
    #[error("Database error: {0}")]
    Database(#[from] DatabaseError),

    /// Error occurred while processing or parsing data
    #[error("Processing error: {0}")]
    Processing(#[from] ProcessingError),
}

/// Errors related to embedding generation
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum EmbeddingError {
    /// Error when no embedding is returned from the API
    #[error("No embedding returned from API")]
    NoEmbeddingReturned,

    /// Error when tokenizer fails to load
    #[error("Failed to load tokenizer '{model}': {message}")]
    TokenizerLoad { model: String, message: String },

    /// Error when tokenization fails
    #[error("Failed to tokenize text: {0}")]
    TokenizationFailed(String),

    /// Error when building an embedding request
    #[error("Failed to build embedding request: {0}")]
    RequestBuild(async_openai::error::OpenAIError),

    /// Error during API communication
    #[error("Embedding API error: {0}")]
    ApiError(#[source] async_openai::error::OpenAIError),

    /// Error when input is invalid (e.g., empty when it shouldn't be)
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Error with text chunking configuration
    #[error("Failed to configure text chunking: {0}")]
    ChunkConfigError(#[from] text_splitter::ChunkConfigError),
}

/// Errors related to database operations
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum DatabaseError {
    /// Error from libsql operations
    #[error("Database operation failed: {0}")]
    LibSql(#[from] libsql::Error),

    /// Error when inserting a document
    #[error("Failed to insert document: {0}")]
    InsertFailed(String),

    /// Error when querying the database
    #[error("Failed to query database: {0}")]
    QueryFailed(String),

    /// Error with database schema setup
    #[error("Failed to setup database schema: {0}")]
    SchemaSetup(String),
}

/// Errors related to data processing and serialization
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum ProcessingError {
    /// JSON serialization/deserialization error
    #[error("JSON processing error: {0}")]
    Json(#[from] serde_json::Error),

    /// Error when required data is missing
    #[error("Missing required data: {0}")]
    MissingData(String),

    /// Generic I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type alias for Capsa operations
pub type Result<T> = std::result::Result<T, CapsaError>;

// Automatic conversions for nested error types
impl From<libsql::Error> for CapsaError {
    fn from(err: libsql::Error) -> Self {
        CapsaError::Database(DatabaseError::LibSql(err))
    }
}

impl From<serde_json::Error> for CapsaError {
    fn from(err: serde_json::Error) -> Self {
        CapsaError::Processing(ProcessingError::Json(err))
    }
}

impl From<text_splitter::ChunkConfigError> for CapsaError {
    fn from(err: text_splitter::ChunkConfigError) -> Self {
        CapsaError::Embedding(EmbeddingError::ChunkConfigError(err))
    }
}

impl From<async_openai::error::OpenAIError> for CapsaError {
    fn from(err: async_openai::error::OpenAIError) -> Self {
        CapsaError::Embedding(EmbeddingError::ApiError(err))
    }
}

impl From<std::io::Error> for CapsaError {
    fn from(err: std::io::Error) -> Self {
        CapsaError::Processing(ProcessingError::Io(err))
    }
}
