//! ADAM MCP server binary: reads newline-delimited JSON-RPC messages from
//! stdin, dispatches them against a pool of [`Organism`]s (one per
//! `organism_id`, lazily created), and writes responses to stdout — the
//! standard MCP stdio transport.

use std::io::{self, BufRead, Write};

use adam_mcp::pool::DEFAULT_ORGANISM_ID;
use adam_mcp::{handle_message, OrganismPool};
use adam_organism::Organism;

fn main() {
    let default_memory_path =
        std::env::var("ADAM_MEMORY_PATH").unwrap_or_else(|_| "adam_memory.db".to_string());
    let default_genome_path =
        std::env::var("ADAM_GENOME_PATH").unwrap_or_else(|_| "adam_genome.json".to_string());
    // Additional organism ids (see adam_mcp::pool) get their state under
    // this directory instead of the single-organism env vars above, so
    // existing single-organism deployments are unaffected by default.
    let data_dir = std::env::var("ADAM_DATA_DIR").unwrap_or_else(|_| ".".to_string());

    let mut pool = OrganismPool::new(move |id: &str| {
        let (memory_path, genome_path) = if id == DEFAULT_ORGANISM_ID {
            (default_memory_path.clone(), default_genome_path.clone())
        } else {
            (
                format!("{data_dir}/{id}_memory.db"),
                format!("{data_dir}/{id}_genome.json"),
            )
        };
        Organism::open(
            "ADAM",
            "An autonomous cognitive evolution layer",
            &memory_path,
            &genome_path,
        )
    });

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(_) => break,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let message = match serde_json::from_str::<serde_json::Value>(trimmed) {
            Ok(value) => value,
            Err(err) => {
                let error = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": serde_json::Value::Null,
                    "error": { "code": -32700, "message": format!("parse error: {err}") }
                });
                writeln!(out, "{error}").ok();
                out.flush().ok();
                continue;
            }
        };

        if let Some(response) = handle_message(&mut pool, &message) {
            writeln!(out, "{response}").ok();
            out.flush().ok();
        }
    }
}
