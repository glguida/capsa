# Capsa

Document management with embeddings.

## Commands

### Add a PDF document

```bash
capsa pdf <path>
```

Extracts PDF metadata and text, stores in vector database.

### Add a YouTube transcript

```bash
capsa yt [--lang <code>] <id_or_url>
```

Downloads YouTube transcript, stores with video metadata.

- `--lang <code>` - Language code (default: en)

### Query documents

```bash
capsa ask [-d] [-k <num>] "query"
```

- `-d` - Show similarity percentage
- `-k <num>` - Number of results (default: 5)

## Configuration

Environment variables:

- `EMB_BASE_URL` - Embedding service URL (default: `http://localhost:9000/v1`)
- `EMB_MODEL` - Model name (default: `nomic-ai/nomic-embed-text-v1.5`)
- `EMB_API_KEY` - API key (optional)
- `EMB_CTX` - Context size (default: 128)
- `VDB_PATH` - Database path (default: `./documents.db`)
