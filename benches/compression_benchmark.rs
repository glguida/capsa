use capsa::vectordb::VectorDatabase;
use serde_json::json;
use std::time::Instant;

fn add_offsets(embeddings: Vec<Vec<f32>>) -> Vec<(Vec<f32>, usize, usize)> {
    embeddings
        .into_iter()
        .enumerate()
        .map(|(i, vec)| (vec, i * 100, (i + 1) * 100))
        .collect()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== LZ4 Compression Benchmark ===\n");

    let db = VectorDatabase::new(":memory:", 384).await?;
    let conn = db.connect().await?;

    let embedding = vec![0.1f32; 384];

    // Test documents of varying sizes
    let test_cases = vec![
        (
            "Small (450B)",
            "The quick brown fox jumps over the lazy dog. ".repeat(10),
        ),
        (
            "Medium (5.8KB)",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(100),
        ),
        (
            "Large (79KB)",
            "This is a sample document with repetitive content that should compress well. "
                .repeat(1000),
        ),
        (
            "Very Large (790KB)",
            "RAG systems require efficient document storage. ".repeat(15000),
        ),
    ];

    for (name, content) in test_cases {
        let original_size = content.len();

        // Measure compression (insert)
        let start = Instant::now();
        let doc_id = conn
            .insert_document(
                &content,
                json!({"test": name}),
                add_offsets(vec![embedding.clone()]),
            )
            .await?;
        let compress_time = start.elapsed();

        // Measure decompression (fetch)
        let start = Instant::now();
        let (fetched, _) = conn.fetch_document(doc_id).await?.unwrap();
        let decompress_time = start.elapsed();

        // Verify correctness
        assert_eq!(fetched.len(), content.len());

        println!("{} document:", name);
        println!("  Original size:      {:>10} bytes", original_size);
        println!(
            "  Compress time:      {:>10.3} ms",
            compress_time.as_secs_f64() * 1000.0
        );
        println!(
            "  Decompress time:    {:>10.3} ms",
            decompress_time.as_secs_f64() * 1000.0
        );
        println!(
            "  Throughput (comp):  {:>10.1} MB/s",
            (original_size as f64 / 1_000_000.0) / compress_time.as_secs_f64()
        );
        println!(
            "  Throughput (decomp): {:>10.1} MB/s\n",
            (original_size as f64 / 1_000_000.0) / decompress_time.as_secs_f64()
        );
    }

    println!("All compression/decompression cycles verified successfully!");
    Ok(())
}
