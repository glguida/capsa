# Capsa

Document management with embeddings.

## Configuration

Global options (available for all commands):

- `--base-url <url>` - Embedding service URL (default: `http://localhost:9000/v1`)
- `--model <name>` - Model name (default: `nomic-ai/nomic-embed-text-v1.5`)
- `--db-path <path>` - Database path (default: `./documents.db`)

Environment variables:

- `EMB_API_KEY` - API key for embedding service (optional)

The embedding context size is fixed at 128 tokens, which provides excellent search results.

## Commands

### Add a PDF document

```bash
capsa [options] pdf <path>
```

Extracts PDF metadata and text, stores in vector database.

Example:
```bash
capsa --base-url http://localhost:9000/v1 --model nomic-ai/nomic-embed-text-v1.5 pdf document.pdf
```

### Add a YouTube transcript

```bash
capsa [options] yt [--lang <code>] <id_or_url>
```

Downloads YouTube transcript, stores with video metadata.

- `--lang <code>` - Language code (default: en)

Example:
```bash
capsa yt --lang en dQw4w9WgXcQ
```

### Query documents

```bash
capsa [options] ask [-d] [-k <num>] "query"
```

- `-d` - Show similarity percentage
- `-k <num>` - Number of results (default: 5)

Example:
```bash
capsa ask -d -k 10 "machine learning techniques"
```
