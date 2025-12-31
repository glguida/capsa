# Capsa

[![CI](https://github.com/glguida/capsa/actions/workflows/ci.yml/badge.svg)](https://github.com/glguida/capsa/actions/workflows/ci.yml)

**A compact, lightweight library for embedding-based document storage and retrieval.**

Capsa is a Rust library that implements the retrieval component of RAG (Retrieval-Augmented Generation) systems. It provides a simple API for ingesting documents, generating embeddings, storing them in a vector database, and performing semantic search through natural language queries.

The repository also includes a fully-functional CLI tool for document indexing and semantic search.

## How It Works

Capsa uses a standard vector database approach:

1. **Document Chunking** - Documents are split into 128-token chunks with overlap to preserve context
2. **Embedding Generation** - Each chunk is converted to a vector representation using an embedding model (via OpenAI-compatible API)
3. **Vector Storage** - Embeddings are stored in [libSQL](https://github.com/tursodatabase/libsql) (Turso's fork of SQLite with vector indexing) for fast similarity search
4. **Semantic Query** - Queries are embedded and matched against stored vectors using cosine similarity

This allows finding relevant content based on semantic meaning rather than exact keyword matches.

## Library Usage

Add Capsa to your `Cargo.toml`:

```toml
[dependencies]
capsa = "0.1"
```

### Example

```rust
use capsa::{config::Config, documentdb::DocumentDatabase};
use serde_json::json;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Configure the embedding service and database
    let config = Config::new(
        "http://localhost:9000/v1".to_string(),
        "nomic-ai/nomic-embed-text-v1.5".to_string(),
        "./documents.db".to_string(),
        None, // API key (optional)
    );

    // Connect to the database
    let db = DocumentDatabase::new(&config).await?;
    let conn = db.connect().await?;

    // Index a document
    let metadata = json!({
        "title": "My Document",
        "author": "Author Name"
    });
    let doc_id = conn.insert(metadata, "Your document text here").await?;
    println!("Indexed document: {}", doc_id);

    // Search
    let results = conn.search_topk("your query", 5).await?;
    for (doc_id, metadata, start, end) in results {
        println!("Found in doc {}: chars {}-{}", doc_id, start, end);
    }

    Ok(())
}
```

## CLI Tool

### Installation

```bash
git clone https://github.com/glguida/capsa
cd capsa
cargo build --release

# Optionally install to ~/.cargo/bin
cargo install --path .
```

### Prerequisites

Capsa requires an embedding service with an OpenAI-compatible API. You have several options:

**Option 1: llama.cpp**
```bash
llama-server -m /path/to/nomic-embed-text-v1.5.Q4_K_M.gguf --embeddings --port 9000
```

**Option 2: text-embeddings-inference**

For GPU/CUDA support:
```bash
docker run -p 9000:80 ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id nomic-ai/nomic-embed-text-v1.5
```

For CPU only support:
```bash
docker run -p 9000:80 ghcr.io/huggingface/text-embeddings-inference:cpu-latest \
  --model-id nomic-ai/nomic-embed-text-v1.5
```

**Option 3: Any OpenAI-compatible API** (remote or local)

### Basic Usage

**Index documents:**
```bash
capsa pdf paper.pdf
capsa yt dQw4w9WgXcQ
capsa yt --lang es VIDEO_ID
```

**Query:**
```bash
capsa ask "your question here"
capsa ask -d -k 20 "detailed query"
```

## Examples

### Indexing Documents

Add a PDF document:
```bash
$ capsa pdf attention-is-all-you-need.pdf
================================================================================
PDF DOCUMENT INGESTION SYSTEM
================================================================================
FILE......: attention-is-all-you-need.pdf

EXTRACTING TEXT...
EXTRACTION COMPLETE
TEXT SIZE.: 33110 CHARACTERS
TITLE.....: Attention is All you Need
AUTHOR....: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, �ukasz Kaiser, Illia Polosukhin

INITIALIZING DATABASE CONNECTION... DONE

PROCESSING... COMPLETE

================================================================================
INGESTION COMPLETE - DOCID=000001
================================================================================
$
```

Add a YouTube video transcript:
```bash
$ capsa yt dQw4w9WgXcQ
================================================================================
YOUTUBE TRANSCRIPT INGESTION SYSTEM
================================================================================
INPUT.....: dQw4w9WgXcQ
LANGUAGE..: en

EXTRACTING VIDEO ID...
VIDEO ID..: dQw4w9WgXcQ

FETCHING VIDEO DETAILS...
TITLE.....: Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster)
AUTHOR....: Rick Astley

FETCHING TRANSCRIPT...
TRANSCRIPT FETCHED
TEXT SIZE.: 2335 CHARACTERS
LANGUAGE..: English

INITIALIZING DATABASE CONNECTION... DONE

PROCESSING... COMPLETE

================================================================================
INGESTION COMPLETE - DOCID=000002
================================================================================
$
```

### Semantic Search

Simple query:
```
$ capsa ask -d -k 1 "What is the transformer architecture?"
================================================================================
DOCUMENT RETRIEVAL SYSTEM
================================================================================
QUERY.....: What is the transformer architecture?
TOP-K.....: 1

INITIALIZING DATABASE CONNECTION... DONE

================================================================================
RECORD 001  DOCID=000001  SIMILARITY= 76.70%
================================================================================
TITLE..: Attention is All you Need
AUTHOR.: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, �ukasz Kaiser, Illia Polosukhin
SUBJECT: Neural Information Processing Systems http://nips.cc/
FILE...: attention-is-all-you-need.pdf
OFFSET.: 4080-4478 (398 BYTES)
--------------------------------------------------------------------------------
CONTENT:
--------------------------------------------------------------------------------
In this work we propose the Transformer, a model architecture eschewing recurrence and instead
relying entirely on an attention mechanism to draw global dependencies between input and output.
The Transformer allows for signiﬁcantly more parallelization and can reach a new state of the art in
translation quality after being trained for as little as twelve hours on eight P100 GPUs.

2 Background
--------------------------------------------------------------------------------
$
```

Another query, on the same database:
```
$ capsa ask -d -k 1 "Will you disappoint me?"
================================================================================
DOCUMENT RETRIEVAL SYSTEM
================================================================================
QUERY.....: Will you disappoint me?
TOP-K.....: 1

INITIALIZING DATABASE CONNECTION... DONE

================================================================================
RECORD 001  DOCID=000002  SIMILARITY= 54.33%
================================================================================
TITLE..: Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster)
AUTHOR.: Rick Astley
OFFSET.: 511-974 (463 BYTES)
--------------------------------------------------------------------------------
CONTENT:
--------------------------------------------------------------------------------
for so long ♪ ♪ Your heart's been aching
but you're too shy to say it ♪ ♪ Inside we both know
what's been going ♪ ♪ We know the game
and we're gonna play it ♪ ♪ And if you ask me
how I'm feeling ♪ ♪ Don't tell me
you're too blind to see ♪ ♪ Never gonna give you up ♪ ♪ Never gonna let you down ♪ ♪ Never gonna run around
and desert you ♪ ♪ Never gonna make you cry ♪ ♪ Never gonna say goodbye ♪ ♪ Never gonna tell a lie
--------------------------------------------------------------------------------
$
```

Output with `-d` shows cosine similarity percentages, helping you gauge result relevance.

## Configuration

### Global Options

Available for all commands:

- `--base-url <url>` - Embedding service URL (default: `http://localhost:9000/v1`)
- `--model <name>` - Model name (default: `nomic-ai/nomic-embed-text-v1.5`)
- `--db-path <path>` - Database path (default: `./documents.db`)

### Environment Variables

- `EMB_API_KEY` - API key for embedding service (optional)

## Command Reference

### `pdf` - Index PDF Documents

```bash
capsa pdf <path>
```

Extracts PDF metadata and text, generates embeddings, and stores them in the vector database.

### `yt` - Index YouTube Transcripts

```bash
capsa yt [--lang <code>] <id_or_url>
```

Downloads YouTube transcript with metadata and indexes it for semantic search.

**Options:**
- `--lang <code>` - Language code (default: `en`)

**Accepts:** Video ID or full YouTube URL

### `ask` - Semantic Search

```bash
capsa ask [-d] [-k <num>] "query"
```

Query your document database using natural language.

**Options:**
- `-d` - Show similarity percentages for each result
- `-k <num>` - Number of results to return (default: `5`)

## License

MIT
