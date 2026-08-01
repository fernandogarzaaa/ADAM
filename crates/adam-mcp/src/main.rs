//! ADAM MCP server binary: reads newline-delimited JSON-RPC messages from
//! stdin, dispatches them against a single in-process [`Organism`], and
//! writes responses to stdout — the standard MCP stdio transport.

use std::io::{self, BufRead, Write};

use adam_mcp::handle_message;
use adam_organism::Organism;

fn main() {
    let memory_path =
        std::env::var("ADAM_MEMORY_PATH").unwrap_or_else(|_| "adam_memory.db".to_string());
    let genome_path =
        std::env::var("ADAM_GENOME_PATH").unwrap_or_else(|_| "adam_genome.json".to_string());
    let mut organism = Organism::open(
        "ADAM",
        "An autonomous cognitive evolution layer",
        &memory_path,
        &genome_path,
    )
    .expect("failed to initialize ADAM organism");

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

        if let Some(response) = handle_message(&mut organism, &message) {
            writeln!(out, "{response}").ok();
            out.flush().ok();
        }
    }
}
