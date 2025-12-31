# Capsa

[![CI](https://github.com/glguida/capsa/actions/workflows/ci.yml/badge.svg)](https://github.com/glguida/capsa/actions/workflows/ci.yml)

**A vector database CLI for semantic document search, written in Rust.**

Capsa implements the retrieval component of RAG (Retrieval-Augmented Generation) systems. It ingests documents, generates embeddings, stores them in a vector database, and enables semantic search through natural language queries.

## How It Works

Capsa uses a standard vector database approach:

1. **Document Chunking** - Documents are split into 128-token chunks with overlap to preserve context
2. **Embedding Generation** - Each chunk is converted to a vector representation using an embedding model (via OpenAI-compatible API)
3. **Vector Storage** - Embeddings are stored in [libSQL](https://github.com/tursodatabase/libsql) (Turso's fork of SQLite with vector indexing) for fast similarity search
4. **Semantic Query** - Queries are embedded and matched against stored vectors using cosine similarity

This allows finding relevant content based on semantic meaning rather than exact keyword matches.

## Quick Start

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
capsa pdf attention-is-all-you-need.pdf
```

Add a YouTube video transcript:
```bash
capsa yt dQw4w9WgXcQ
```

### Semantic Search

Simple query:
```bash
capsa ask "What is the transformer architecture?"
```

With similarity scores and more results:
```bash
capsa ask -d -k 10 "self-attention mechanism"
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
