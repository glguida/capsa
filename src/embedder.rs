//! A module for generating text embeddings using OpenAI-compatible APIs.
//!
//! This library provides functionality to create embeddings from text documents,
//! handling text splitting, chunking, and parallel processing of large documents.

use anyhow::{Result, anyhow};
use async_openai::{
    Client,
    config::OpenAIConfig,
    types::embeddings::{CreateEmbeddingRequestArgs, EmbeddingInput},
};

/// Client for making embedding requests to OpenAI-compatible APIs.
///
/// This client handles authentication and communication with embedding services
/// that implement the OpenAI embeddings API format.
pub struct EmbeddingClient {
    base_url: String,
    api_key: Option<String>,
    model: String,
}

impl EmbeddingClient {
    /// Creates a new embedding client.
    ///
    /// # Arguments
    ///
    /// * `base_url` - The base URL of the embeddings API endpoint
    /// * `api_key` - Optional API key for authentication
    /// * `model` - Name of the embedding model to use
    pub fn new(base_url: String, api_key: Option<String>, model: String) -> Self {
        Self {
            base_url,
            api_key,
            model,
        }
    }

    /// Generates an embedding vector for the given text input.
    ///
    /// # Arguments
    ///
    /// * `input` - The text to embed
    ///
    /// # Returns
    ///
    /// A vector of f32 values representing the embedding.
    ///
    /// # Errors
    ///
    /// Returns an error if the API request fails or no embedding is returned.
    pub async fn embed_raw(&self, input: &str) -> Result<Vec<f32>> {
        let mut config = OpenAIConfig::default();
        if let Some(ref key) = self.api_key {
            config = config.with_api_key(key);
        }
        config = config.with_api_base(&self.base_url);
        let client = Client::with_config(config);
        let req = CreateEmbeddingRequestArgs::default()
            .model(&self.model)
            .input(EmbeddingInput::String(input.into()))
            .build()?;
        let resp = client.embeddings().create(req).await?;
        resp.data
            .into_iter()
            .next()
            .ok_or(anyhow!("No embedding returned"))
            .map(|e| e.embedding)
    }
}

#[cfg(test)]
mod client_tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_embed_client_openai() -> Result<()> {
        let key = std::env::var("OPENAI_API_KEY").ok();
        if let Some(k) = key {
            let client = EmbeddingClient::new(
                "https://api.openai.com/v1".to_string(),
                Some(k),
                "text-embedding-3-small".to_string(),
            );
            let emb = client.embed_raw("test").await?;
            assert!(!emb.is_empty());
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_embed_client_local() {
        let client = EmbeddingClient::new(
            "http://localhost:9000/v1".to_string(),
            None,
            "model".to_string(),
        );
        let res = client.embed_raw("test").await;
        if res.is_err() { /* Tolerate absent server */ }
    }

    #[tokio::test]
    async fn test_empty_input() {
        let client = EmbeddingClient::new(
            "https://api.openai.com/v1".to_string(),
            None,
            "model".to_string(),
        );
        let res = client.embed_raw("").await;
        assert!(res.is_err());
    }
}

use text_splitter::{ChunkConfig, TextSplitter};
use tokenizers::tokenizer::Tokenizer;

/// Handles text splitting and chunking for embedding generation.
///
/// This splitter manages token limits and overlap for both queries and documents,
/// ensuring text fits within the model's context window.
pub struct EmbeddingSplitter {
    query_splitter: TextSplitter<Tokenizer>,
    document_splitter: TextSplitter<Tokenizer>,
}

impl EmbeddingSplitter {
    fn create_splitter(
        n_ctx: usize,
        prefix: &str,
        tokenizer: Tokenizer,
        overlap: f32,
    ) -> Result<TextSplitter<Tokenizer>> {
        let prefix_len = tokenizer
            .encode(prefix, true)
            .map_err(|e| anyhow!("Failed to tokenize: {}", e))?
            .get_ids()
            .len();
        let chunk_len = n_ctx - prefix_len;
        let overlap_len: usize = ((chunk_len as f32) * overlap) as usize;
        let config = ChunkConfig::new(chunk_len)
            .with_sizer(tokenizer)
            .with_overlap(overlap_len)?;
        Ok(TextSplitter::new(config))
    }

    /// Creates a new embedding splitter with the specified model and context size.
    ///
    /// # Arguments
    ///
    /// * `model` - Name of the pretrained tokenizer model to load
    /// * `n_ctx` - Maximum context size in tokens
    ///
    /// # Returns
    ///
    /// A configured splitter with 5% overlap between chunks.
    ///
    /// # Errors
    ///
    /// Returns an error if the tokenizer cannot be loaded.
    pub fn new(model: &str, n_ctx: usize) -> Result<Self> {
        let overlap = 0.05; /* 5% overlap */
        let tokenizer = Tokenizer::from_pretrained(model, None)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;
        let query_splitter =
            Self::create_splitter(n_ctx, "search_query: ", tokenizer.clone(), overlap)?;
        let document_splitter =
            Self::create_splitter(n_ctx, "search_document: ", tokenizer, overlap)?;
        Ok(EmbeddingSplitter {
            query_splitter,
            document_splitter,
        })
    }

    /// Truncates query text to fit within the context window.
    ///
    /// Returns the first chunk of the text that fits within the query token limit,
    /// accounting for the "query_document: " prefix.
    ///
    /// # Arguments
    ///
    /// * `text` - The query text to truncate
    ///
    /// # Returns
    ///
    /// A string slice containing the truncated text, or empty string if input is empty.
    pub fn truncate_query<'a>(&self, text: &'a str) -> &'a str {
        self.query_splitter.chunks(text).next().unwrap_or("")
    }

    /// Splits document text into overlapping chunks that fit within the context window.
    ///
    /// Returns an iterator over text chunks, each sized to fit within the document
    /// token limit with 5% overlap between consecutive chunks. Accounts for the
    /// "search_document: " prefix.
    ///
    /// # Arguments
    ///
    /// * `text` - The document text to split
    ///
    /// # Returns
    ///
    /// An iterator yielding string slices for each chunk.
    pub fn document_chunks<'a>(&'a self, text: &'a str) -> impl Iterator<Item = &'a str> + 'a {
        self.document_splitter.chunks(text)
    }

    /// Splits document text into overlapping chunks with byte offset positions.
    ///
    /// Returns an iterator over text chunks along with their start and end byte offsets
    /// in the original text. Each chunk is sized to fit within the document token limit
    /// with 5% overlap between consecutive chunks. Accounts for the "search_document: " prefix.
    ///
    /// The byte offsets are guaranteed to fall on UTF-8 character boundaries since the
    /// chunks are valid string slices from the original text.
    ///
    /// # Arguments
    ///
    /// * `text` - The document text to split
    ///
    /// # Returns
    ///
    /// An iterator yielding tuples of (chunk, start_byte, end_byte) for each chunk.
    pub fn document_chunks_with_offsets<'a>(
        &'a self,
        text: &'a str,
    ) -> impl Iterator<Item = (&'a str, usize, usize)> + 'a {
        self.document_splitter.chunks(text).map(move |chunk| {
            // Calculate byte offset from the original string
            let start = chunk.as_ptr() as usize - text.as_ptr() as usize;
            let end = start + chunk.len();
            (chunk, start, end)
        })
    }
}

#[cfg(test)]
mod splitter_tests {
    use super::*;

    /* Test creation with valid model */
    #[test]
    fn create_valid() -> Result<()> {
        let s = EmbeddingSplitter::new("bert-base-uncased", 512)?;
        drop(s);
        Ok(())
    }

    /* Test creation fails with invalid model */
    #[test]
    fn create_invalid() {
        let e = EmbeddingSplitter::new("no-such-model-xyz", 512);
        assert!(e.is_err());
    }

    /* Test query truncation respects token limits */
    #[test]
    fn truncate_query_limits_tokens() -> Result<()> {
        let m = "bert-base-uncased";
        let n = 512;
        let s = EmbeddingSplitter::new(m, n)?;
        let t =
            Tokenizer::from_pretrained(m, None).map_err(|e| anyhow!("tokenizer fail: {}", e))?;
        let p = "query_document: ";
        let txt: String = (0..2000).map(|i| format!("test{} ", i)).collect();
        let tr = s.truncate_query(&txt);
        let full = format!("{}{}", p, tr);
        let enc = t
            .encode(full.as_str(), true)
            .map_err(|_| anyhow!("Failed!"))?;
        let l = enc.get_ids().len();
        assert!(l <= n);
        Ok(())
    }

    /* Test query truncation on short text */
    #[test]
    fn truncate_query_short() -> Result<()> {
        let m = "bert-base-uncased";
        let n = 512;
        let s = EmbeddingSplitter::new(m, n)?;
        let txt = "short text";
        let tr = s.truncate_query(txt);
        assert_eq!(tr, txt);
        Ok(())
    }

    /* Test query truncation on empty text */
    #[test]
    fn truncate_query_empty() -> Result<()> {
        let m = "bert-base-uncased";
        let n = 512;
        let s = EmbeddingSplitter::new(m, n)?;
        let tr = s.truncate_query("");
        assert_eq!(tr, "");
        Ok(())
    }

    /* Test document chunks respect token limits and overlap */
    #[test]
    fn document_chunks_limits_and_overlap() -> Result<()> {
        let m = "bert-base-uncased";
        let n = 512;
        let s = EmbeddingSplitter::new(m, n)?;
        let t =
            Tokenizer::from_pretrained(m, None).map_err(|e| anyhow!("tokenizer fail: {}", e))?;
        let p = "search_document: ";
        let pl = t
            .encode(p, true)
            .map_err(|_| anyhow!("Failed!"))?
            .get_ids()
            .len();
        let cl = n - pl;
        let ol = ((cl as f32) * 0.05) as usize;
        let txt: String = (0..3000).map(|i| format!("chunk{} ", i)).collect();
        let ch: Vec<_> = s.document_chunks(&txt).collect();
        assert!(ch.len() > 1);
        for c in &ch {
            let full = format!("{}{}", p, *c);
            let enc = t
                .encode(full.as_str(), true)
                .map_err(|_| anyhow!("Failed!"))?;
            assert!(enc.get_ids().len() <= n);
        }
        for w in ch.windows(2) {
            let a = w[0];
            let b = w[1];
            let mut overlap_chars = 0;
            for i in (1..=a.len().min(b.len())).rev() {
                if &a[a.len() - i..] == &b[0..i] {
                    overlap_chars = i;
                    break;
                }
            }
            assert!(overlap_chars > 0);
            let o_txt = &a[a.len() - overlap_chars..];
            let enc_o = t.encode(o_txt, false).map_err(|_| anyhow!("Failed!"))?;
            let o_len = enc_o.get_ids().len();
            assert!((o_len as i32 - ol as i32).abs() <= 10); /* allow variance */
        }
        Ok(())
    }

    /* Test document chunks on short text */
    #[test]
    fn document_chunks_short() -> Result<()> {
        let m = "bert-base-uncased";
        let n = 512;
        let s = EmbeddingSplitter::new(m, n)?;
        let txt = "short doc";
        let ch: Vec<_> = s.document_chunks(txt).collect();
        assert_eq!(ch.len(), 1);
        assert_eq!(ch[0], txt);
        Ok(())
    }

    /* Test document chunks on empty text */
    #[test]
    fn document_chunks_empty() -> Result<()> {
        let m = "bert-base-uncased";
        let n = 512;
        let s = EmbeddingSplitter::new(m, n)?;
        let ch: Vec<_> = s.document_chunks("").collect();
        assert!(ch.is_empty());
        Ok(())
    }

    /* Test document chunks with offsets */
    #[test]
    fn document_chunks_with_offsets() -> Result<()> {
        let m = "bert-base-uncased";
        let n = 512;
        let s = EmbeddingSplitter::new(m, n)?;
        let txt: String = (0..1000).map(|i| format!("word{} ", i)).collect();
        let chunks: Vec<_> = s.document_chunks_with_offsets(&txt).collect();

        // Verify offsets are valid and chunks match the original text
        for (chunk, start, end) in &chunks {
            assert_eq!(*chunk, &txt[*start..*end]);
            assert!(start < end);
            assert!(*end <= txt.len());
        }

        // Verify chunks are in order and don't have gaps
        if chunks.len() > 1 {
            let (_, first_start, _) = chunks[0];
            assert_eq!(first_start, 0); // First chunk starts at beginning
        }

        Ok(())
    }
}

use futures::stream::{self, StreamExt, TryStreamExt};

/// High-level API for generating embeddings from various text sources.
///
/// Combines an embedding client with text splitting to handle documents
/// of any size, automatically chunking and processing them in parallel.
pub struct Embedder {
    client: EmbeddingClient,
    splitter: EmbeddingSplitter,
}

impl Embedder {
    /// Creates a new embedder with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `base_url` - The base URL of the embeddings API endpoint
    /// * `model` - Name of the embedding model and tokenizer to use
    /// * `api_key` - Optional API key for authentication
    /// * `n_ctx` - Maximum context size in tokens
    ///
    /// # Errors
    ///
    /// Returns an error if the tokenizer for the model cannot be loaded.
    pub fn new(
        base_url: String,
        model: String,
        api_key: Option<String>,
        n_ctx: usize,
    ) -> Result<Self> {
        let splitter = EmbeddingSplitter::new(&model, n_ctx)?;
        let client = EmbeddingClient::new(base_url, api_key, model);
        Ok(Embedder { client, splitter })
    }

    /// Generates embeddings for a document by splitting it into chunks.
    ///
    /// The text is automatically split into overlapping chunks that fit within
    /// the context window. Each chunk is prefixed with "search_document: " before
    /// embedding. Chunks are processed in parallel with up to 5 concurrent requests.
    ///
    /// # Arguments
    ///
    /// * `text` - The document text to embed
    ///
    /// # Returns
    ///
    /// A vector of tuples (embedding_vector, chunk_start, chunk_end), one per chunk.
    /// The chunk_start and chunk_end are byte offsets in the original text.
    ///
    /// # Errors
    ///
    /// Returns an error if any API request fails.
    pub async fn embed_document(&self, text: &str) -> Result<Vec<(Vec<f32>, usize, usize)>> {
        // Stream chunks and embed them immediately as they're tokenized
        let results: Vec<_> = stream::iter(self.splitter.document_chunks_with_offsets(text))
            .map(|(chunk, start, end)| {
                let prefixed = format!("search_document: {}", chunk);
                async move {
                    let embedding = self.client.embed_raw(&prefixed).await?;
                    Ok::<_, anyhow::Error>((embedding, start, end))
                }
            })
            .buffered(5)
            .try_collect()
            .await?;

        Ok(results)
    }

    /// Generates a single embedding vector for a search query.
    ///
    /// The query is truncated to fit within the context window and prefixed
    /// with "search_query: " before embedding. This is optimized for queries
    /// rather than documents.
    ///
    /// # Arguments
    ///
    /// * `query` - The search query text
    ///
    /// # Returns
    ///
    /// A single embedding vector representing the query.
    ///
    /// # Errors
    ///
    /// Returns an error if the API request fails.
    pub async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let truncated = self.splitter.truncate_query(query);
        let prefixed = format!("search_query: {}", truncated);
        self.client.embed_raw(&prefixed).await
    }
}

#[cfg(test)]
mod embedder_tests {
    use super::*;

    fn test_data_path(filename: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests_data")
            .join(filename)
    }

    #[tokio::test]
    async fn split_large_text() -> Result<()> {
        let embedder = Embedder::new(
            "http://localhost:9000/v1".to_string(),
            "nomic-ai/nomic-embed-text-v1.5".to_string(),
            Some("openai".to_string()),
            512,
        )?;

        let text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ultricies purus sed sapien gravida elementum. Donec ac luctus mauris, sit amet bibendum orci. Vivamus dignissim erat ac laoreet vulputate. Suspendisse volutpat leo sed justo egestas vulputate ut ut eros. Fusce imperdiet justo at molestie vestibulum. In hendrerit sapien ac tincidunt sodales. Sed molestie erat id ante molestie, laoreet finibus enim ullamcorper.

Aenean laoreet convallis risus vitae laoreet. Aenean imperdiet cursus neque, sed rhoncus lacus. Duis dolor magna, fringilla ullamcorper nisl eget, molestie ornare urna. Cras sed ipsum eget est placerat pellentesque. Interdum et malesuada fames ac ante ipsum primis in faucibus. Nunc nulla turpis, condimentum sit amet lacus at, scelerisque viverra tortor. Duis sed viverra felis, sed ullamcorper nunc. Sed semper tristique augue, id posuere turpis tincidunt eget. Quisque elementum risus id sapien venenatis scelerisque. Morbi varius volutpat nunc at fringilla. Etiam nec semper sapien. Nunc id tincidunt lectus. Mauris elit tellus, ultricies elementum sem in, euismod sollicitudin ante. In at ligula mauris. Nam sed eros vel enim sollicitudin suscipit. Curabitur scelerisque lectus non velit porttitor tempus et nec elit.

Aenean cursus ipsum ac suscipit euismod. Nam at sem molestie nunc eleifend posuere. Proin lobortis diam eu tortor scelerisque, sed vulputate nunc pulvinar. Aliquam at tellus quis risus pharetra mollis. Pellentesque mauris nisi, auctor sit amet rhoncus at, lacinia in turpis. Maecenas eget lorem enim. Nunc egestas quam lectus, eget ornare eros sodales quis. Integer nulla nunc, vulputate sit amet pharetra eget, accumsan ac ex. Duis hendrerit maximus felis, ac vehicula lorem.

Nunc ac commodo tortor. In at tellus at mi tempus commodo id ut ipsum. Nam sed tellus tempus, tincidunt eros et, viverra magna. Morbi sed lectus dui. Sed cursus quam urna, non viverra enim vulputate sed. Sed luctus rutrum rhoncus. Maecenas dapibus eros ac orci pharetra gravida. Aenean sit amet quam non ligula mattis egestas fringilla eget elit. In at mollis massa.

Donec lectus nisi, suscipit eu mauris ac, rutrum vehicula nibh. Vestibulum lacinia eget lacus ut fermentum. Morbi varius, purus id ultricies accumsan, lectus arcu ultrices ipsum, commodo molestie elit augue vitae magna. Phasellus sed ipsum ex. Sed lobortis nec justo in eleifend. Fusce ornare ultrices malesuada. Morbi facilisis convallis dui, non luctus sapien varius et. Maecenas et facilisis urna.";

        let vectors = embedder.embed_document(text).await?;
        println!("Vectors String: {:?}", vectors);
        assert!(!vectors.is_empty());
        Ok(())
    }
}
