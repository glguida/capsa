//! Configuration constants for the embedding system.

use secrecy::SecretString;

/// The maximum context size in tokens for text embeddings.
///
/// This value is fixed at 128 tokens, which has been empirically determined
/// to provide excellent search results while maintaining good performance.
pub const EMBEDDING_CONTEXT: usize = 128;

/// Runtime configuration for the embedding system.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct Config {
    /// Base URL for the embedding API endpoint
    pub base_url: String,
    /// Name of the embedding model to use
    pub model: String,
    /// Path to the vector database file
    pub db_path: String,
    /// Optional API key for authentication
    pub api_key: Option<SecretString>,
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
    pub fn new(
        base_url: String,
        model: String,
        db_path: String,
        api_key: Option<SecretString>,
    ) -> Self {
        Config {
            base_url,
            model,
            db_path,
            api_key,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use secrecy::ExposeSecret;

    #[test]
    fn test_config_debug_redacts_api_key() {
        // Test with API key present
        let config_with_key = Config::new(
            "http://localhost:9000".to_string(),
            "test-model".to_string(),
            "./test.db".to_string(),
            Some(SecretString::from("super-secret-api-key-12345")),
        );

        let debug_output = format!("{:?}", config_with_key);

        // Should contain redacted marker
        assert!(debug_output.contains("[REDACTED]"));

        // Should NOT contain the actual API key
        assert!(!debug_output.contains("super-secret-api-key-12345"));

        // Should still contain other fields
        assert!(debug_output.contains("http://localhost:9000"));
        assert!(debug_output.contains("test-model"));
        assert!(debug_output.contains("./test.db"));
    }

    #[test]
    fn test_config_debug_without_api_key() {
        // Test without API key
        let config_without_key = Config::new(
            "http://localhost:9000".to_string(),
            "test-model".to_string(),
            "./test.db".to_string(),
            None,
        );

        let debug_output = format!("{:?}", config_without_key);

        // Should not contain redacted marker when no key present
        assert!(!debug_output.contains("[REDACTED]"));

        // Should contain None
        assert!(debug_output.contains("None"));

        // Should still contain other fields
        assert!(debug_output.contains("http://localhost:9000"));
        assert!(debug_output.contains("test-model"));
        assert!(debug_output.contains("./test.db"));
    }

    #[test]
    fn test_config_clone() {
        // Test that the derived Clone implementation for Config works properly
        let original = Config::new(
            "http://localhost:9000".to_string(),
            "test-model".to_string(),
            "./test.db".to_string(),
            Some(SecretString::from("api-key")),
        );

        let cloned = original.clone();

        assert_eq!(original.base_url, cloned.base_url);
        assert_eq!(original.model, cloned.model);
        assert_eq!(original.db_path, cloned.db_path);
        assert_eq!(
            original.api_key.as_ref().map(|s| s.expose_secret()),
            cloned.api_key.as_ref().map(|s| s.expose_secret())
        );
    }
}
