use anyhow::{Context, Result};
use axum::Router;
use capsa::config::Config;
use capsa::documentdb::DocumentDatabase;
use capsa::executor::Executor;
use clap::{Parser, Subcommand};
use lru::LruCache;
use rmcp::{
    ErrorData as McpError, ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::*,
    schemars, tool, tool_handler, tool_router,
    transport::streamable_http_server::{
        StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
    },
};
use serde_json::json;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use yt_transcript_rs::YouTubeTranscriptApi;

static CONFIG: OnceLock<Config> = OnceLock::new();

#[derive(Parser)]
#[command(name = "capsa")]
#[command(about = "Document management with embeddings", long_about = None)]
struct Cli {
    #[arg(
        long,
        default_value = "http://localhost:9000/v1",
        help = "Base URL for the embedding API"
    )]
    base_url: String,

    #[arg(
        long,
        default_value = "nomic-ai/nomic-embed-text-v1.5",
        help = "Embedding model to use"
    )]
    model: String,

    #[arg(
        long,
        default_value = "./documents.db",
        help = "Path to the vector database"
    )]
    db_path: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Pdf {
        #[arg(help = "Path to the PDF file")]
        path: PathBuf,

        #[arg(
            long,
            help = "Path to a JSON object merged into extracted PDF metadata"
        )]
        metadata_json: Option<PathBuf>,
    },
    Yt {
        #[arg(help = "YouTube video ID or URL")]
        id_or_url: String,

        #[arg(long = "lang", default_value = "en", help = "Transcript language code")]
        lang: String,
    },
    Ask {
        #[arg(help = "Query string")]
        query: String,

        #[arg(short = 'd', help = "Show distance in results")]
        distance: bool,

        #[arg(
            short = 'k',
            default_value = "5",
            help = "Number of top results to return"
        )]
        top_k: usize,
    },
    Serve {
        #[arg(
            long,
            default_value = "127.0.0.1:10080",
            help = "Address to bind the MCP server"
        )]
        bind: String,
    },
}

async fn extract_pdf_metadata_and_text(
    path: &std::path::Path,
) -> Result<(serde_json::Value, String)> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || extract_pdf_metadata_and_text_sync(&path))
        .await
        .context("PDF extraction task panicked")?
}

fn extract_pdf_metadata_and_text_sync(
    path: &std::path::Path,
) -> Result<(serde_json::Value, String)> {
    use lopdf::{Document, Object};
    use rayon::prelude::*;

    let doc = Document::load(path)
        .with_context(|| format!("Failed to read PDF file: {}", path.display()))?;

    // Extract metadata from Info dictionary
    let get_info_string = |dict: &lopdf::Dictionary, key: &[u8]| -> Option<String> {
        dict.get(key).ok().and_then(|obj| {
            if let Object::String(bytes, _) = obj {
                String::from_utf8(bytes.clone()).ok()
            } else {
                None
            }
        })
    };

    let (title, author, subject, keywords, creator, producer) = if let Ok(info_ref) =
        doc.trailer.get(b"Info")
    {
        if let Ok(info_obj_id) = info_ref.as_reference() {
            if let Ok(Object::Dictionary(info_dict)) = doc.get_object(info_obj_id) {
                (
                    get_info_string(info_dict, b"Title").unwrap_or_else(|| "Unknown".to_string()),
                    get_info_string(info_dict, b"Author").unwrap_or_else(|| "Unknown".to_string()),
                    get_info_string(info_dict, b"Subject"),
                    get_info_string(info_dict, b"Keywords"),
                    get_info_string(info_dict, b"Creator"),
                    get_info_string(info_dict, b"Producer"),
                )
            } else {
                (
                    "Unknown".to_string(),
                    "Unknown".to_string(),
                    None,
                    None,
                    None,
                    None,
                )
            }
        } else {
            (
                "Unknown".to_string(),
                "Unknown".to_string(),
                None,
                None,
                None,
                None,
            )
        }
    } else {
        (
            "Unknown".to_string(),
            "Unknown".to_string(),
            None,
            None,
            None,
            None,
        )
    };

    let metadata = json!({
        "title": title,
        "author": author,
        "subject": subject,
        "keywords": keywords,
        "creator": creator,
        "producer": producer,
        "path": path.display().to_string(),
    });

    // Extract text from all pages in parallel, then reassemble in page order
    let pages = doc.get_pages();
    let mut page_texts: Vec<(u32, String)> = pages
        .par_iter()
        .filter_map(|(page_num, _)| {
            doc.extract_text(&[*page_num])
                .ok()
                .map(|text| (*page_num, text))
        })
        .collect();
    page_texts.sort_unstable_by_key(|(n, _)| *n);

    let text_accumulated: String = page_texts
        .into_iter()
        .flat_map(|(_, text)| [text, "\n".to_string()])
        .collect();

    if text_accumulated.trim().is_empty() {
        anyhow::bail!("No text could be extracted from the PDF");
    }

    Ok((metadata, text_accumulated))
}

fn merge_metadata(base: &mut serde_json::Value, overlay: serde_json::Value) -> Result<()> {
    let base_obj = base
        .as_object_mut()
        .context("Base metadata must be a JSON object")?;
    let overlay_obj = overlay
        .as_object()
        .context("Overlay metadata must be a JSON object")?;

    for (k, v) in overlay_obj {
        base_obj.insert(k.clone(), v.clone());
    }
    Ok(())
}

fn extract_video_id(id_or_url: &str) -> Result<String> {
    if id_or_url.len() == 11 && !id_or_url.contains('/') && !id_or_url.contains('&') {
        return Ok(id_or_url.to_string());
    }

    if let Some(v_pos) = id_or_url.find("v=") {
        let start = v_pos + 2;
        let end = id_or_url[start..]
            .find('&')
            .map(|pos| start + pos)
            .unwrap_or(id_or_url.len());
        return Ok(id_or_url[start..end].to_string());
    }

    if id_or_url.contains("youtu.be/")
        && let Some(slash_pos) = id_or_url.rfind('/')
    {
        let start = slash_pos + 1;
        let end = id_or_url[start..]
            .find('?')
            .map(|pos| start + pos)
            .unwrap_or(id_or_url.len());
        return Ok(id_or_url[start..end].to_string());
    }

    anyhow::bail!("Could not extract video ID from: {}", id_or_url);
}

async fn add_yt_document(id_or_url: String, lang: String) -> Result<()> {
    println!("================================================================================");
    println!("YOUTUBE TRANSCRIPT INGESTION SYSTEM");
    println!("================================================================================");
    println!("INPUT.....: {}", id_or_url);
    println!("LANGUAGE..: {}", lang);
    println!();

    println!("EXTRACTING VIDEO ID...");
    let video_id = extract_video_id(&id_or_url)?;
    println!("VIDEO ID..: {}", video_id);
    println!();

    println!("FETCHING VIDEO DETAILS...");
    let api = YouTubeTranscriptApi::new(None, None, None)?;

    let details = api
        .fetch_video_details(&video_id)
        .await
        .with_context(|| format!("Failed to fetch video details for: {}", video_id))?;

    let title = details.title;
    let author = details.author;
    let video_url = format!("https://www.youtube.com/watch?v={}", video_id);

    println!("TITLE.....: {}", title);
    println!("AUTHOR....: {}", author);
    println!();

    println!("FETCHING TRANSCRIPT...");
    let languages = &[lang.as_str()];
    let preserve_formatting = false;

    let transcript = api
        .fetch_transcript(&video_id, languages, preserve_formatting)
        .await
        .with_context(|| format!("Failed to fetch transcript for: {}", video_id))?;

    let text: String = transcript
        .snippets
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    println!("TRANSCRIPT FETCHED");
    println!("TEXT SIZE.: {} CHARACTERS", text.len());
    println!("LANGUAGE..: {}", transcript.language);
    println!();

    let metadata = json!({
        "title": title,
        "author": author,
        "video_url": video_url,
        "video_id": video_id,
        "language": transcript.language,
    });

    print!("INITIALIZING DATABASE CONNECTION...");
    let db = DocumentDatabase::new(CONFIG.get().expect("Config not initialized")).await?;

    let conn = db.connect().await?;
    println!(" DONE");
    println!();

    print!("PROCESSING...");
    let doc_id = conn.insert(metadata, &text).await?;
    println!(" COMPLETE");

    println!();
    println!("================================================================================");
    println!("INGESTION COMPLETE - DOCID={:06}", doc_id);
    println!("================================================================================");

    Ok(())
}

async fn add_pdf_document(path: PathBuf, metadata_json: Option<PathBuf>) -> Result<()> {
    println!("================================================================================");
    println!("PDF DOCUMENT INGESTION SYSTEM");
    println!("================================================================================");
    println!("FILE......: {}", path.display());
    println!();

    println!("EXTRACTING TEXT...");
    let (mut metadata, text) = extract_pdf_metadata_and_text(&path).await?;

    if let Some(metadata_path) = metadata_json {
        let raw = std::fs::read_to_string(&metadata_path).with_context(|| {
            format!("Failed to read metadata JSON: {}", metadata_path.display())
        })?;
        let overlay: serde_json::Value = serde_json::from_str(&raw).with_context(|| {
            format!("Failed to parse metadata JSON: {}", metadata_path.display())
        })?;
        merge_metadata(&mut metadata, overlay)?;
    }

    println!("EXTRACTION COMPLETE");
    println!("TEXT SIZE.: {} CHARACTERS", text.len());
    println!(
        "TITLE.....: {}",
        metadata
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("N/A")
    );
    println!(
        "AUTHOR....: {}",
        metadata
            .get("author")
            .and_then(|v| v.as_str())
            .unwrap_or("N/A")
    );
    println!();

    print!("INITIALIZING DATABASE CONNECTION...");
    let db = DocumentDatabase::new(CONFIG.get().expect("Config not initialized")).await?;

    let conn = db.connect().await?;
    println!(" DONE");
    println!();

    print!("PROCESSING...");
    let doc_id = conn.insert(metadata, &text).await?;
    println!(" COMPLETE");

    println!();
    println!("================================================================================");
    println!("INGESTION COMPLETE - DOCID={:06}", doc_id);
    println!("================================================================================");

    Ok(())
}

async fn ask_query(query: String, show_distance: bool, top_k: usize) -> Result<()> {
    println!("================================================================================");
    println!("DOCUMENT RETRIEVAL SYSTEM");
    println!("================================================================================");
    println!("QUERY.....: {}", query);
    println!("TOP-K.....: {}", top_k);
    println!();

    print!("INITIALIZING DATABASE CONNECTION...");
    use std::io::Write;
    std::io::stdout().flush()?;

    let db = DocumentDatabase::new(CONFIG.get().expect("Config not initialized")).await?;

    let conn = db.connect().await?;
    println!(" DONE");
    println!();

    if show_distance {
        let results = conn.search_topk_with_distance(&query, top_k).await?;

        if results.is_empty() {
            println!("*** NO RECORDS FOUND ***");
        } else {
            let mut doc_cache: LruCache<i64, String> =
                LruCache::new(NonZeroUsize::new(64).unwrap());
            for (idx, (doc_id, metadata, distance, chunk_start, chunk_end)) in
                results.iter().enumerate()
            {
                let similarity_pct = (1.0 - distance) * 100.0;

                println!(
                    "================================================================================"
                );
                println!(
                    "RECORD {:03}  DOCID={:06}  SIMILARITY={:6.2}%",
                    idx + 1,
                    doc_id,
                    similarity_pct
                );
                println!(
                    "================================================================================"
                );
                print_metadata(metadata);
                println!(
                    "OFFSET.: {}-{} ({} BYTES)",
                    chunk_start,
                    chunk_end,
                    chunk_end - chunk_start
                );

                if !doc_cache.contains(doc_id)
                    && let Ok(Some((content, _))) = conn.fetch_document(*doc_id).await
                {
                    doc_cache.put(*doc_id, content);
                }
                if let Some(content) = doc_cache.get(doc_id) {
                    let start = *chunk_start as usize;
                    let end = *chunk_end as usize;
                    if end <= content.len() {
                        let chunk_text = &content[start..end];
                        println!(
                            "--------------------------------------------------------------------------------"
                        );
                        println!("CONTENT:");
                        println!(
                            "--------------------------------------------------------------------------------"
                        );
                        println!("{}", chunk_text);
                        println!(
                            "--------------------------------------------------------------------------------"
                        );
                    }
                }

                println!();
            }
        }
    } else {
        let results = conn.search_topk(&query, top_k).await?;

        if results.is_empty() {
            println!("*** NO RECORDS FOUND ***");
        } else {
            let mut doc_cache: LruCache<i64, String> =
                LruCache::new(NonZeroUsize::new(64).unwrap());
            for (idx, (doc_id, metadata, chunk_start, chunk_end)) in results.iter().enumerate() {
                println!(
                    "================================================================================"
                );
                println!("RECORD {:03}  DOCID={:06}", idx + 1, doc_id);
                println!(
                    "================================================================================"
                );
                print_metadata(metadata);
                println!(
                    "OFFSET.: {}-{} ({} BYTES)",
                    chunk_start,
                    chunk_end,
                    chunk_end - chunk_start
                );

                if !doc_cache.contains(doc_id)
                    && let Ok(Some((content, _))) = conn.fetch_document(*doc_id).await
                {
                    doc_cache.put(*doc_id, content);
                }
                if let Some(content) = doc_cache.get(doc_id) {
                    let start = *chunk_start as usize;
                    let end = *chunk_end as usize;
                    if end <= content.len() {
                        let chunk_text = &content[start..end];
                        println!(
                            "--------------------------------------------------------------------------------"
                        );
                        println!("CONTENT:");
                        println!(
                            "--------------------------------------------------------------------------------"
                        );
                        println!("{}", chunk_text);
                        println!(
                            "--------------------------------------------------------------------------------"
                        );
                    }
                }

                println!();
            }
        }
    }

    Ok(())
}

fn print_metadata(metadata: &serde_json::Value) {
    if let Some(obj) = metadata.as_object() {
        let mut keys: Vec<&String> = obj.keys().collect();
        keys.sort();
        for key in keys {
            let value = &obj[key];
            let display = match value {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Null => continue,
                other => other.to_string(),
            };
            if !display.is_empty() {
                println!("{}: {}", key.to_uppercase(), display);
            }
        }
    }
}

// --- MCP Server ---

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct SearchRequest {
    /// Natural language query to search for
    query: String,
    /// Number of results to return (default: 5)
    top_k: Option<usize>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct FetchDocumentRequest {
    /// Document ID as returned by the search tool
    doc_id: i64,
}

#[derive(Clone)]
struct CapsaServer {
    conn: Arc<tokio::sync::Mutex<capsa::documentdb::DocumentDatabaseConnection>>,
    tool_router: ToolRouter<CapsaServer>,
}

#[tool_router]
impl CapsaServer {
    fn new(conn: Arc<tokio::sync::Mutex<capsa::documentdb::DocumentDatabaseConnection>>) -> Self {
        Self {
            conn,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        description = "Search the document database for content semantically matching a query. Returns ranked chunks with similarity scores, metadata, and the matched text."
    )]
    async fn search(
        &self,
        Parameters(req): Parameters<SearchRequest>,
    ) -> Result<CallToolResult, McpError> {
        let conn = self.conn.lock().await;

        let results = conn
            .search_topk_with_distance(&req.query, req.top_k.unwrap_or(5))
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let mut doc_cache: HashMap<i64, String> = HashMap::new();
        let mut formatted = Vec::new();

        for (doc_id, metadata, distance, chunk_start, chunk_end) in &results {
            if !doc_cache.contains_key(doc_id)
                && let Ok(Some((content, _))) = conn.fetch_document(*doc_id).await
            {
                doc_cache.insert(*doc_id, content);
            }
            let chunk = doc_cache.get(doc_id).map(|content| {
                let start = *chunk_start as usize;
                let end = (*chunk_end as usize).min(content.len());
                content[start..end].to_string()
            });
            formatted.push(json!({
                "doc_id": doc_id,
                "similarity_pct": ((1.0 - distance) * 10000.0).round() / 100.0,
                "metadata": metadata,
                "chunk": chunk,
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
            }));
        }

        Ok(CallToolResult::success(vec![Content::text(
            json!({ "results": formatted }).to_string(),
        )]))
    }

    #[tool(description = "Fetch the full text content and metadata of a document by its ID.")]
    async fn fetch_document(
        &self,
        Parameters(req): Parameters<FetchDocumentRequest>,
    ) -> Result<CallToolResult, McpError> {
        let conn = self.conn.lock().await;

        match conn
            .fetch_document(req.doc_id)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?
        {
            Some((content, metadata)) => Ok(CallToolResult::success(vec![Content::text(
                json!({
                    "doc_id": req.doc_id,
                    "metadata": metadata,
                    "content": content,
                })
                .to_string(),
            )])),
            None => Ok(CallToolResult::success(vec![Content::text(
                json!({ "error": format!("document {} not found", req.doc_id) }).to_string(),
            )])),
        }
    }
}

#[tool_handler]
impl ServerHandler for CapsaServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::from_build_env())
    }
}

async fn run_serve(bind: String) -> Result<()> {
    println!("================================================================================");
    println!("CAPSA MCP SERVER");
    println!("================================================================================");
    println!("BIND......: {}", bind);
    println!();

    print!("INITIALIZING DATABASE CONNECTION...");
    let db = DocumentDatabase::new(CONFIG.get().expect("Config not initialized")).await?;
    let conn = Arc::new(tokio::sync::Mutex::new(db.connect().await?));
    println!(" DONE");
    println!();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::DEBUG.into()),
        )
        .init();

    println!("LISTENING ON http://{}/", bind);
    println!("================================================================================");

    let ct = tokio_util::sync::CancellationToken::new();
    let service = StreamableHttpService::new(
        {
            let conn = conn.clone();
            move || Ok(CapsaServer::new(conn.clone()))
        },
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig {
            cancellation_token: ct.child_token(),
            ..Default::default()
        },
    );

    let router = Router::new()
        .fallback_service(service)
        .layer(tower_http::trace::TraceLayer::new_for_http());
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            tokio::signal::ctrl_c().await.unwrap();
            ct.cancel();
        })
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lopdf::{Dictionary, Document, Object, Stream};
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Build a synthetic multi-page PDF entirely in memory.
    ///
    /// Each page contains a unique marker of the form "MARKER_PAGE_N" so tests
    /// can verify both content and page ordering.
    fn make_synthetic_pdf(page_count: usize) -> NamedTempFile {
        let mut doc = Document::with_version("1.5");

        // Pre-allocate the pages-tree object ID so pages can reference it as parent
        let pages_id = doc.new_object_id();

        // Helvetica is a standard Type1 font — no embedding required
        let mut font = Dictionary::new();
        font.set("Type", Object::Name(b"Font".to_vec()));
        font.set("Subtype", Object::Name(b"Type1".to_vec()));
        font.set("BaseFont", Object::Name(b"Helvetica".to_vec()));
        let font_id = doc.add_object(Object::Dictionary(font));

        let mut fonts_dict = Dictionary::new();
        fonts_dict.set("F1", Object::Reference(font_id));
        let mut resources = Dictionary::new();
        resources.set("Font", Object::Dictionary(fonts_dict));

        let mut page_ids = vec![];
        for i in 0..page_count {
            let marker = format!("MARKER_PAGE_{}", i + 1);
            let content_bytes = format!("BT /F1 12 Tf 72 720 Td ({}) Tj ET", marker).into_bytes();
            let content_id = doc.add_object(Object::Stream(Stream::new(
                Dictionary::new(),
                content_bytes,
            )));

            let mut page = Dictionary::new();
            page.set("Type", Object::Name(b"Page".to_vec()));
            page.set("Parent", Object::Reference(pages_id));
            page.set(
                "MediaBox",
                Object::Array(vec![
                    Object::Integer(0),
                    Object::Integer(0),
                    Object::Integer(612),
                    Object::Integer(792),
                ]),
            );
            page.set("Contents", Object::Reference(content_id));
            page.set("Resources", Object::Dictionary(resources.clone()));
            page_ids.push(doc.add_object(Object::Dictionary(page)));
        }

        let kids: Vec<Object> = page_ids.iter().map(|id| Object::Reference(*id)).collect();
        let mut pages = Dictionary::new();
        pages.set("Type", Object::Name(b"Pages".to_vec()));
        pages.set("Kids", Object::Array(kids));
        pages.set("Count", Object::Integer(page_count as i64));
        doc.objects.insert(pages_id, Object::Dictionary(pages));

        let mut catalog = Dictionary::new();
        catalog.set("Type", Object::Name(b"Catalog".to_vec()));
        catalog.set("Pages", Object::Reference(pages_id));
        let catalog_id = doc.add_object(Object::Dictionary(catalog));
        doc.trailer.set("Root", Object::Reference(catalog_id));

        let mut tmp = NamedTempFile::new().expect("tempfile");
        let mut buf = Vec::new();
        doc.save_to(&mut buf).expect("save pdf");
        tmp.write_all(&buf).expect("write pdf");
        tmp
    }

    #[test]
    fn test_pdf_extraction_returns_text() {
        let tmp = make_synthetic_pdf(20);
        let (metadata, text) =
            extract_pdf_metadata_and_text_sync(tmp.path()).expect("extraction should succeed");

        assert!(!text.trim().is_empty(), "extracted text must not be empty");
        assert!(
            metadata.get("title").is_some(),
            "metadata must contain a title field"
        );
        // All 20 pages should contribute text
        for i in 1..=20 {
            assert!(
                text.contains(&format!("MARKER_PAGE_{}", i)),
                "missing marker for page {}",
                i
            );
        }
    }

    #[test]
    fn test_pdf_extraction_page_order() {
        let tmp = make_synthetic_pdf(30);
        let (_, text) = extract_pdf_metadata_and_text_sync(tmp.path()).unwrap();

        // Verify pages appear in ascending order by finding each marker's position
        let positions: Vec<usize> = (1..=30)
            .map(|i| {
                text.find(&format!("MARKER_PAGE_{}", i))
                    .unwrap_or_else(|| panic!("missing marker for page {}", i))
            })
            .collect();

        for w in positions.windows(2) {
            assert!(
                w[0] < w[1],
                "page order not preserved: pos {} >= {}",
                w[0],
                w[1]
            );
        }
    }

    #[tokio::test]
    async fn test_pdf_extraction_async() {
        // Exercises the spawn_blocking wrapper end-to-end
        let tmp = make_synthetic_pdf(5);
        let path = tmp.path().to_path_buf();
        let (_, text) = extract_pdf_metadata_and_text(&path)
            .await
            .expect("async extraction should succeed");
        assert!(text.contains("MARKER_PAGE_1"));
        assert!(text.contains("MARKER_PAGE_5"));
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize global configuration
    let api_key = std::env::var("CAPSA_API_KEY")
        .ok()
        .map(secrecy::SecretString::from);
    CONFIG.get_or_init(|| {
        Config::new(cli.base_url, cli.model, cli.db_path, api_key)
            .with_executor(Executor::parallel())
    });

    match cli.command {
        Commands::Pdf {
            path,
            metadata_json,
        } => {
            add_pdf_document(path, metadata_json).await?;
        }
        Commands::Yt { id_or_url, lang } => {
            add_yt_document(id_or_url, lang).await?;
        }
        Commands::Ask {
            query,
            distance,
            top_k,
        } => {
            ask_query(query, distance, top_k).await?;
        }
        Commands::Serve { bind } => {
            run_serve(bind).await?;
        }
    }

    Ok(())
}
