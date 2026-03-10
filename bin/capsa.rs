use anyhow::{Context, Result};
use axum::Router;
use capsa::config::Config;
use capsa::documentdb::DocumentDatabase;
use capsa::executor::Executor;
use capsa::queue::{Queue, queue_db_path};
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
    Pptx {
        #[arg(help = "Path to the PPTX file")]
        path: PathBuf,

        #[arg(
            long,
            help = "Path to a JSON object merged into extracted PPTX metadata"
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

        #[arg(long, help = "Enable write access (exposes the ingest MCP tool)")]
        rw: bool,
    },
    /// Show the current state of the ingestion pipeline queue.
    Status,
    /// Re-queue all failed ingestion jobs so they are retried.
    Retry,
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

async fn extract_pptx_metadata_and_text(
    path: &std::path::Path,
) -> Result<(serde_json::Value, String)> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || extract_pptx_metadata_and_text_sync(&path))
        .await
        .context("PPTX extraction task panicked")?
}

fn extract_pptx_metadata_and_text_sync(
    path: &std::path::Path,
) -> Result<(serde_json::Value, String)> {
    use rayon::prelude::*;
    use std::io::Read;
    use zip::ZipArchive;

    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open PPTX file: {}", path.display()))?;
    let mut archive =
        ZipArchive::new(file).context("Failed to read archive — is this a valid PPTX file?")?;

    // Extract core properties from docProps/core.xml
    let (title, creator) = {
        let mut title = "Unknown".to_string();
        let mut creator = "Unknown".to_string();
        if let Ok(mut entry) = archive.by_name("docProps/core.xml") {
            let mut xml = String::new();
            let _ = entry.read_to_string(&mut xml);
            if let Some(t) = pptx_core_prop(&xml, "title") {
                title = t;
            }
            if let Some(c) = pptx_core_prop(&xml, "creator") {
                creator = c;
            }
        }
        (title, creator)
    };

    // Collect slide entry names, sorted by slide number
    let mut slide_names: Vec<String> = (0..archive.len())
        .filter_map(|i| {
            let name = archive.by_index(i).ok()?.name().to_string();
            if pptx_is_slide(&name) {
                Some(name)
            } else {
                None
            }
        })
        .collect();
    slide_names.sort_by_key(|n| pptx_slide_number(n));

    let slide_count = slide_names.len();

    // Read all slide XMLs sequentially (ZipArchive is not Sync)
    let slide_xmls: Vec<String> = slide_names
        .iter()
        .map(|name| {
            let mut entry = archive
                .by_name(name)
                .with_context(|| format!("Cannot read {}", name))?;
            let mut xml = String::new();
            entry.read_to_string(&mut xml)?;
            Ok(xml)
        })
        .collect::<Result<Vec<_>>>()?;

    // Parse DrawingML text in parallel
    let slide_texts: Vec<String> = slide_xmls
        .par_iter()
        .map(|xml| pptx_extract_drawingml_text(xml))
        .collect();

    let text: String = slide_texts
        .into_iter()
        .filter(|t| !t.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n");

    if text.trim().is_empty() {
        anyhow::bail!("No text could be extracted from the PPTX");
    }

    let metadata = json!({
        "title": title,
        "creator": creator,
        "slide_count": slide_count,
        "path": path.display().to_string(),
    });

    Ok((metadata, text))
}

/// Returns true for slide XML entries, excluding relationship sidecars.
fn pptx_is_slide(name: &str) -> bool {
    name.starts_with("ppt/slides/slide") && name.ends_with(".xml") && !name.contains("/_rels/")
}

/// Extracts the numeric suffix from a slide entry name for sorting.
fn pptx_slide_number(name: &str) -> usize {
    name.trim_start_matches("ppt/slides/slide")
        .trim_end_matches(".xml")
        .parse()
        .unwrap_or(0)
}

/// Extracts a Dublin-Core property value from docProps/core.xml by local tag name.
///
/// Matches any namespace-prefixed element (`<dc:title>`, `<cp:creator>`, etc.).
/// Content ends at the first `<` after the opening tag, which is always the
/// corresponding close tag in well-formed XML.
fn pptx_core_prop(xml: &str, tag: &str) -> Option<String> {
    let open = format!(":{tag}>");
    let start = xml.find(&open)? + open.len();
    let end = xml[start..].find('<')?;
    let value = xml[start..start + end].trim().to_string();
    if value.is_empty() { None } else { Some(value) }
}

/// Extracts all human-readable text from a DrawingML slide XML.
///
/// Iterates over `<a:t>` text runs and flushes accumulated text on each
/// `</a:p>` paragraph boundary.
fn pptx_extract_drawingml_text(xml: &str) -> String {
    use quick_xml::Reader;
    use quick_xml::events::Event;

    let mut reader = Reader::from_str(xml);
    let mut result = String::new();
    let mut current_para = String::new();
    let mut in_t = false;

    loop {
        match reader.read_event() {
            Ok(Event::Start(ref e)) => {
                if e.local_name().as_ref() == b"t" {
                    in_t = true;
                }
            }
            Ok(Event::Text(ref e)) if in_t => {
                if let Ok(text) = e.unescape() {
                    current_para.push_str(&text);
                }
            }
            Ok(Event::End(ref e)) => match e.local_name().as_ref() {
                b"t" => in_t = false,
                b"p" => {
                    let trimmed = current_para.trim();
                    if !trimmed.is_empty() {
                        result.push_str(trimmed);
                        result.push('\n');
                    }
                    current_para.clear();
                }
                _ => {}
            },
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
    }

    result
}

async fn add_pptx_document(path: PathBuf, metadata_json: Option<PathBuf>) -> Result<()> {
    println!("================================================================================");
    println!("PPTX DOCUMENT INGESTION SYSTEM");
    println!("================================================================================");
    println!("FILE......: {}", path.display());
    println!();

    println!("EXTRACTING TEXT...");
    let (mut metadata, text) = extract_pptx_metadata_and_text(&path).await?;

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
        "CREATOR...: {}",
        metadata
            .get("creator")
            .and_then(|v| v.as_str())
            .unwrap_or("N/A")
    );
    println!(
        "SLIDES....: {}",
        metadata
            .get("slide_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    );
    println!();

    print!("OPENING QUEUE...");
    let config = CONFIG.get().expect("Config not initialized");
    let queue = Queue::new(&queue_db_path(&config.db_path)).await?;
    let conn = queue.connect().await?;
    println!(" DONE");
    println!();

    conn.enqueue(&format!("pptx:{}", path.display()), &metadata, &text)
        .await?;

    println!();
    println!("================================================================================");
    println!("QUEUED FOR INGESTION");
    println!("================================================================================");

    Ok(())
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

    print!("OPENING QUEUE...");
    let config = CONFIG.get().expect("Config not initialized");
    let queue = Queue::new(&queue_db_path(&config.db_path)).await?;
    let conn = queue.connect().await?;
    println!(" DONE");
    println!();

    conn.enqueue(&format!("yt:{}", video_id), &metadata, &text)
        .await?;

    println!();
    println!("================================================================================");
    println!("QUEUED FOR INGESTION");
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

    print!("OPENING QUEUE...");
    let config = CONFIG.get().expect("Config not initialized");
    let queue = Queue::new(&queue_db_path(&config.db_path)).await?;
    let conn = queue.connect().await?;
    println!(" DONE");
    println!();

    conn.enqueue(&format!("pdf:{}", path.display()), &metadata, &text)
        .await?;

    println!();
    println!("================================================================================");
    println!("QUEUED FOR INGESTION");
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

    let db = DocumentDatabase::new_reader(CONFIG.get().expect("Config not initialized")).await?;

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

type CapsaConn = Arc<tokio::sync::Mutex<capsa::documentdb::DocumentDatabaseConnection>>;

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

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct IngestRequest {
    /// Full text content to index
    text: String,
    /// Document metadata as a JSON object. All fields are optional and free-form.
    /// Common fields: "title", "author", "source", "url", "date", "language".
    metadata: serde_json::Value,
}

// ── Shared tool logic ────────────────────────────────────────────────────────

async fn mcp_search(
    conn: &capsa::documentdb::DocumentDatabaseConnection,
    req: SearchRequest,
) -> Result<CallToolResult, McpError> {
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

async fn mcp_fetch_document(
    conn: &capsa::documentdb::DocumentDatabaseConnection,
    req: FetchDocumentRequest,
) -> Result<CallToolResult, McpError> {
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

async fn mcp_ingest(
    conn: &capsa::documentdb::DocumentDatabaseConnection,
    req: IngestRequest,
) -> Result<CallToolResult, McpError> {
    conn.insert(req.metadata, &req.text)
        .await
        .map_err(|e| McpError::internal_error(e.to_string(), None))?;
    Ok(CallToolResult::success(vec![Content::text(
        "queued for ingestion",
    )]))
}

// ── Read-only server (default) ───────────────────────────────────────────────

#[derive(Clone)]
struct CapsaServerRO {
    conn: CapsaConn,
    tool_router: ToolRouter<CapsaServerRO>,
}

#[tool_router]
impl CapsaServerRO {
    fn new(conn: CapsaConn) -> Self {
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
        mcp_search(&conn, req).await
    }

    #[tool(description = "Fetch the full text content and metadata of a document by its ID.")]
    async fn fetch_document(
        &self,
        Parameters(req): Parameters<FetchDocumentRequest>,
    ) -> Result<CallToolResult, McpError> {
        let conn = self.conn.lock().await;
        mcp_fetch_document(&conn, req).await
    }
}

#[tool_handler]
impl ServerHandler for CapsaServerRO {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::from_build_env())
    }
}

// ── Read-write server (--rw) ─────────────────────────────────────────────────

#[derive(Clone)]
struct CapsaServer {
    conn: CapsaConn,
    tool_router: ToolRouter<CapsaServer>,
}

#[tool_router]
impl CapsaServer {
    fn new(conn: CapsaConn) -> Self {
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
        mcp_search(&conn, req).await
    }

    #[tool(description = "Fetch the full text content and metadata of a document by its ID.")]
    async fn fetch_document(
        &self,
        Parameters(req): Parameters<FetchDocumentRequest>,
    ) -> Result<CallToolResult, McpError> {
        let conn = self.conn.lock().await;
        mcp_fetch_document(&conn, req).await
    }

    #[tool(
        description = "Add a document to the ingestion queue. Chunking, embedding, and indexing happen asynchronously in the background."
    )]
    async fn ingest(
        &self,
        Parameters(req): Parameters<IngestRequest>,
    ) -> Result<CallToolResult, McpError> {
        let conn = self.conn.lock().await;
        mcp_ingest(&conn, req).await
    }
}

#[tool_handler]
impl ServerHandler for CapsaServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::from_build_env())
    }
}

async fn show_status() -> Result<()> {
    let config = CONFIG.get().expect("Config not initialized");
    let queue = Queue::new(&queue_db_path(&config.db_path)).await?;
    let conn = queue.connect().await?;

    let status = conn.status_counts().await?;
    println!("================================================================================");
    println!("PIPELINE STATUS");
    println!("================================================================================");
    println!("  PENDING.....: {}", status.pending);
    println!("  CHUNKED.....: {}", status.chunked);
    println!("  EMBEDDED....: {}", status.embedded);
    println!("  DONE........: {}", status.done);
    println!("  FAILED......: {}", status.failed);

    let failed = conn.failed_items().await?;
    if !failed.is_empty() {
        println!();
        println!("FAILED ITEMS");
        println!(
            "--------------------------------------------------------------------------------"
        );
        for item in failed {
            println!("  job_id={} source={}", item.job_id, item.source);
            println!("  error: {}", item.error);
            println!();
        }
    }

    Ok(())
}

async fn retry_failed() -> Result<()> {
    let config = CONFIG.get().expect("Config not initialized");
    let queue = Queue::new(&queue_db_path(&config.db_path)).await?;
    let conn = queue.connect().await?;
    let count = conn.retry_failed().await?;
    if count == 0 {
        println!("No failed jobs to retry.");
    } else {
        println!("Re-queued {} failed job(s).", count);
    }
    Ok(())
}

async fn run_serve(bind: String, rw: bool) -> Result<()> {
    println!("================================================================================");
    println!("CAPSA MCP SERVER");
    println!("================================================================================");
    println!("BIND......: {}", bind);
    println!(
        "MODE......: {}",
        if rw { "READ-WRITE" } else { "READ-ONLY" }
    );
    println!();

    print!("INITIALIZING DATABASE CONNECTION...");
    let db = DocumentDatabase::new(CONFIG.get().expect("Config not initialized")).await?;
    let conn: CapsaConn = Arc::new(tokio::sync::Mutex::new(db.connect().await?));
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
    let server_config = StreamableHttpServerConfig {
        cancellation_token: ct.child_token(),
        ..Default::default()
    };
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    let shutdown = async move {
        tokio::signal::ctrl_c().await.unwrap();
        ct.cancel();
    };

    if rw {
        let service = StreamableHttpService::new(
            {
                let conn = conn.clone();
                move || Ok(CapsaServer::new(conn.clone()))
            },
            LocalSessionManager::default().into(),
            server_config,
        );
        axum::serve(
            listener,
            Router::new()
                .fallback_service(service)
                .layer(tower_http::trace::TraceLayer::new_for_http()),
        )
        .with_graceful_shutdown(shutdown)
        .await?;
    } else {
        let service = StreamableHttpService::new(
            {
                let conn = conn.clone();
                move || Ok(CapsaServerRO::new(conn.clone()))
            },
            LocalSessionManager::default().into(),
            server_config,
        );
        axum::serve(
            listener,
            Router::new()
                .fallback_service(service)
                .layer(tower_http::trace::TraceLayer::new_for_http()),
        )
        .with_graceful_shutdown(shutdown)
        .await?;
    }

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

    /// Build a synthetic PPTX (ZIP) in memory with `slide_count` slides.
    ///
    /// Each slide XML contains a single paragraph with "SLIDE_N_TEXT".
    fn make_synthetic_pptx(slide_count: usize) -> NamedTempFile {
        use std::io::Write as _;

        let tmp = NamedTempFile::new().expect("tempfile");
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(tmp.path())
            .expect("open tempfile");
        let mut zip = zip::ZipWriter::new(file);
        let options = zip::write::FileOptions::<()>::default()
            .compression_method(zip::CompressionMethod::Stored);

        // docProps/core.xml — minimal Dublin Core metadata
        zip.start_file("docProps/core.xml", options).unwrap();
        zip.write_all(
            br#"<?xml version="1.0"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
  xmlns:dc="http://purl.org/dc/elements/1.1/">
  <dc:title>Synthetic Presentation</dc:title>
  <dc:creator>Test Author</dc:creator>
</cp:coreProperties>"#,
        )
        .unwrap();

        // One slide XML per slide
        for i in 1..=slide_count {
            let name = format!("ppt/slides/slide{}.xml", i);
            zip.start_file(name, options).unwrap();
            let xml = format!(
                r#"<?xml version="1.0"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
       xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
  <p:cSld><p:spTree>
    <p:sp><p:txBody>
      <a:p><a:r><a:t>SLIDE_{}_TEXT</a:t></a:r></a:p>
    </p:txBody></p:sp>
  </p:spTree></p:cSld>
</p:sld>"#,
                i
            );
            zip.write_all(xml.as_bytes()).unwrap();
        }

        zip.finish().unwrap();
        tmp
    }

    #[test]
    fn test_pptx_extraction_returns_text() {
        let tmp = make_synthetic_pptx(10);
        let (metadata, text) =
            extract_pptx_metadata_and_text_sync(tmp.path()).expect("extraction should succeed");

        assert!(!text.trim().is_empty(), "extracted text must not be empty");
        assert_eq!(
            metadata.get("title").and_then(|v| v.as_str()),
            Some("Synthetic Presentation")
        );
        assert_eq!(
            metadata.get("creator").and_then(|v| v.as_str()),
            Some("Test Author")
        );
        assert_eq!(
            metadata.get("slide_count").and_then(|v| v.as_u64()),
            Some(10)
        );
        for i in 1..=10 {
            assert!(
                text.contains(&format!("SLIDE_{}_TEXT", i)),
                "missing text for slide {}",
                i
            );
        }
    }

    #[test]
    fn test_pptx_extraction_slide_order() {
        let tmp = make_synthetic_pptx(20);
        let (_, text) = extract_pptx_metadata_and_text_sync(tmp.path()).unwrap();

        let positions: Vec<usize> = (1..=20)
            .map(|i| {
                text.find(&format!("SLIDE_{}_TEXT", i))
                    .unwrap_or_else(|| panic!("missing text for slide {}", i))
            })
            .collect();

        for w in positions.windows(2) {
            assert!(
                w[0] < w[1],
                "slide order not preserved: pos {} >= {}",
                w[0],
                w[1]
            );
        }
    }

    #[tokio::test]
    async fn test_pptx_extraction_async() {
        let tmp = make_synthetic_pptx(5);
        let path = tmp.path().to_path_buf();
        let (metadata, text) = extract_pptx_metadata_and_text(&path)
            .await
            .expect("async extraction should succeed");
        assert!(text.contains("SLIDE_1_TEXT"));
        assert!(text.contains("SLIDE_5_TEXT"));
        assert_eq!(
            metadata.get("slide_count").and_then(|v| v.as_u64()),
            Some(5)
        );
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
        Commands::Pptx {
            path,
            metadata_json,
        } => {
            add_pptx_document(path, metadata_json).await?;
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
        Commands::Serve { bind, rw } => {
            run_serve(bind, rw).await?;
        }
        Commands::Status => {
            show_status().await?;
        }
        Commands::Retry => {
            retry_failed().await?;
        }
    }

    Ok(())
}
