use anyhow::{Context, Result};
use capsa::documentdb::DocumentDatabase;
use clap::{Parser, Subcommand};
use serde_json::json;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "capsa")]
#[command(about = "Document management with embeddings", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Add {
        #[arg(help = "Path to the PDF file")]
        path: PathBuf,
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

async fn add_document(path: PathBuf) -> Result<()> {
    println!("================================================================================");
    println!("DOCUMENT INGESTION SYSTEM");
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

    let emb_base_url =
        std::env::var("EMB_BASE_URL").unwrap_or_else(|_| "http://localhost:9000/v1".to_string());
    let emb_model =
        std::env::var("EMB_MODEL").unwrap_or_else(|_| "nomic-ai/nomic-embed-text-v1.5".to_string());
    let emb_api_key = std::env::var("EMB_API_KEY").ok();
    let emb_ctx = std::env::var("EMB_CTX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let vdb_path = std::env::var("VDB_PATH").unwrap_or_else(|_| "./documents.db".to_string());

    print!("INITIALIZING DATABASE CONNECTION...");
    let db = DocumentDatabase::new(emb_base_url, emb_model, emb_api_key, emb_ctx, vdb_path).await?;

    let conn = db.connect().await?;
    println!(" DONE");
    println!();

    println!("PROCESSING:");
    use std::cell::Cell;
    use std::io::Write;

    let total_chunks = Cell::new(0);
    let doc_id = conn
        .insert_with_progress(
            metadata,
            &text,
            Some(|count| {
                total_chunks.set(count);
                if count % 5 == 0 || count == 1 {
                    let spinner_chars = ['|', '/', '-', '\\'];
                    let spinner = spinner_chars[count % 4];
                    print!("\r  EMBEDDING: {} CHUNKS {}", count, spinner);
                    std::io::stdout().flush().unwrap();
                }
            }),
            Some(|count| {
                total_chunks.set(count);
                if count % 5 == 0 || count == 1 {
                    let spinner_chars = ['|', '/', '-', '\\'];
                    let spinner = spinner_chars[count % 4];
                    print!("\r  DATABASE: {} CHUNKS {}", count, spinner);
                    std::io::stdout().flush().unwrap();
                }
            }),
        )
        .await?;

    println!(
        "\r  DATABASE: {} CHUNKS - COMPLETE        ",
        total_chunks.get()
    );

    println!();
    println!("================================================================================");
    println!("INGESTION COMPLETE - DOCID={:06}", doc_id);
    println!("================================================================================");

    Ok(())
}

async fn ask_query(query: String, show_distance: bool, top_k: usize) -> Result<()> {
    let emb_base_url =
        std::env::var("EMB_BASE_URL").unwrap_or_else(|_| "http://localhost:9000/v1".to_string());
    let emb_model =
        std::env::var("EMB_MODEL").unwrap_or_else(|_| "nomic-ai/nomic-embed-text-v1.5".to_string());
    let emb_api_key = std::env::var("EMB_API_KEY").ok();
    let emb_ctx = std::env::var("EMB_CTX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let vdb_path = std::env::var("VDB_PATH").unwrap_or_else(|_| "./documents.db".to_string());

    println!("================================================================================");
    println!("DOCUMENT RETRIEVAL SYSTEM");
    println!("================================================================================");
    println!("QUERY.....: {}", query);
    println!("TOP-K.....: {}", top_k);
    println!();

    print!("INITIALIZING DATABASE CONNECTION...");
    use std::io::Write;
    std::io::stdout().flush()?;

    let db = DocumentDatabase::new(emb_base_url, emb_model, emb_api_key, emb_ctx, vdb_path).await?;

    let conn = db.connect().await?;
    println!(" DONE");
    println!();

    if show_distance {
        let results = conn.search_topk_with_distance(&query, top_k).await?;

        if results.is_empty() {
            println!("*** NO RECORDS FOUND ***");
        } else {
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

                // Fetch document and show excerpt
                if let Ok(Some((content, _))) = conn.fetch_document(*doc_id).await {
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

                // Fetch document and show excerpt
                if let Ok(Some((content, _))) = conn.fetch_document(*doc_id).await {
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

    match cli.command {
        Commands::Add { path } => {
            add_document(path).await?;
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
