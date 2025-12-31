use anyhow::{Context, Result};
use capsa::config::Config;
use capsa::documentdb::DocumentDatabase;
use clap::{Parser, Subcommand};
use lru::LruCache;
use serde_json::json;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::OnceLock;
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
}

async fn extract_pdf_metadata_and_text(path: &PathBuf) -> Result<(serde_json::Value, String)> {
    let doc = pdf_extract::Document::load(path)
        .with_context(|| format!("Failed to read PDF file: {}", path.display()))?;

    // Extract metadata from Info dictionary
    let get_info_string = |dict: &pdf_extract::Dictionary, key: &[u8]| -> Option<String> {
        dict.get(key)
            .ok()
            .and_then(|obj| obj.as_string().ok())
            .map(|cow_str| cow_str.to_string())
    };

    let (title, author, subject, keywords, creator, producer) =
        if let Ok(info_ref) = doc.trailer.get(b"Info") {
            if let Ok(info_id) = info_ref.as_reference() {
                if let Ok(info_obj) = doc.get_object(info_id) {
                    if let Ok(info_dict) = info_obj.as_dict() {
                        (
                            get_info_string(info_dict, b"Title")
                                .unwrap_or_else(|| "Unknown".to_string()),
                            get_info_string(info_dict, b"Author")
                                .unwrap_or_else(|| "Unknown".to_string()),
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

    // Extract text using pdf-extract
    let text =
        pdf_extract::extract_text(path).with_context(|| "Failed to extract text from PDF")?;

    if text.trim().is_empty() {
        anyhow::bail!("No text could be extracted from the PDF");
    }

    Ok((metadata, text))
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

async fn add_pdf_document(path: PathBuf) -> Result<()> {
    println!("================================================================================");
    println!("PDF DOCUMENT INGESTION SYSTEM");
    println!("================================================================================");
    println!("FILE......: {}", path.display());
    println!();

    println!("EXTRACTING TEXT...");
    let (metadata, text) = extract_pdf_metadata_and_text(&path).await?;

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
                println!(
                    "TITLE..: {}",
                    metadata
                        .get("title")
                        .and_then(|v| v.as_str())
                        .unwrap_or("N/A")
                );
                println!(
                    "AUTHOR.: {}",
                    metadata
                        .get("author")
                        .and_then(|v| v.as_str())
                        .unwrap_or("N/A")
                );
                if let Some(subject) = metadata.get("subject").and_then(|v| v.as_str()) {
                    println!("SUBJECT: {}", subject);
                }
                if let Some(path) = metadata.get("path").and_then(|v| v.as_str()) {
                    println!("FILE...: {}", path);
                }
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
                println!(
                    "TITLE..: {}",
                    metadata
                        .get("title")
                        .and_then(|v| v.as_str())
                        .unwrap_or("N/A")
                );
                println!(
                    "AUTHOR.: {}",
                    metadata
                        .get("author")
                        .and_then(|v| v.as_str())
                        .unwrap_or("N/A")
                );
                if let Some(subject) = metadata.get("subject").and_then(|v| v.as_str()) {
                    println!("SUBJECT: {}", subject);
                }
                if let Some(path) = metadata.get("path").and_then(|v| v.as_str()) {
                    println!("FILE...: {}", path);
                }
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

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize global configuration
    let api_key = std::env::var("CAPSA_API_KEY")
        .ok()
        .map(secrecy::SecretString::from);
    CONFIG.get_or_init(|| Config::new(cli.base_url, cli.model, cli.db_path, api_key));

    match cli.command {
        Commands::Pdf { path } => {
            add_pdf_document(path).await?;
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
    }

    Ok(())
}
