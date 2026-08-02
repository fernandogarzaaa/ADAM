//! Minimal JSON-RPC 2.0 / MCP protocol handling over newline-delimited
//! messages, decoupled from stdio so it can be unit tested directly.

use serde_json::{json, Value};

use crate::dispatch::call_tool;
use crate::pool::{OrganismPool, DEFAULT_ORGANISM_ID};
use crate::tools::tool_definitions;

const PROTOCOL_VERSION: &str = "2024-11-05";
const SERVER_NAME: &str = "adam-mcp";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Handle one incoming JSON-RPC message. Returns `Some(response)` for
/// requests (which carry an `id` and must be answered) and `None` for
/// notifications (which must not be answered per the JSON-RPC spec).
pub fn handle_message(pool: &mut OrganismPool, message: &Value) -> Option<Value> {
    let id = message.get("id").cloned();
    let method = message.get("method").and_then(Value::as_str).unwrap_or("");
    let params = message.get("params").cloned().unwrap_or(json!({}));

    let id = match id {
        Some(id) => id,
        None => {
            // Notification (e.g. "notifications/initialized") — no response.
            return None;
        }
    };

    let result = match method {
        "initialize" => Ok(json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": { "tools": {} },
            "serverInfo": { "name": SERVER_NAME, "version": SERVER_VERSION }
        })),
        "tools/list" => Ok(json!({ "tools": tool_definitions() })),
        "tools/call" => handle_tools_call(pool, &params),
        "ping" => Ok(json!({})),
        other => Err((-32601, format!("method not found: {other}"))),
    };

    Some(match result {
        Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
        Err((code, message)) => json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": { "code": code, "message": message }
        }),
    })
}

fn handle_tools_call(pool: &mut OrganismPool, params: &Value) -> Result<Value, (i64, String)> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or((-32602, "missing 'name'".to_string()))?;
    let arguments = params.get("arguments").cloned().unwrap_or(json!({}));
    let organism_id = arguments
        .get("organism_id")
        .and_then(Value::as_str)
        .unwrap_or(DEFAULT_ORGANISM_ID);

    let organism = pool
        .get_or_create(organism_id)
        .map_err(|e| (-32000, e.to_string()))?;

    match call_tool(organism, name, &arguments) {
        Ok(value) => Ok(json!({
            "content": [{ "type": "text", "text": value.to_string() }],
            "isError": false
        })),
        Err(message) => Ok(json!({
            "content": [{ "type": "text", "text": message }],
            "isError": true
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adam_organism::Organism;

    fn ephemeral_pool() -> OrganismPool {
        OrganismPool::new(|_id| Organism::new("ADAM", "test organism", ":memory:"))
    }

    #[test]
    fn initialize_returns_protocol_and_server_info() {
        let mut pool = ephemeral_pool();
        let request = json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {} });
        let response = handle_message(&mut pool, &request).unwrap();
        assert_eq!(response["result"]["protocolVersion"], PROTOCOL_VERSION);
        assert_eq!(response["result"]["serverInfo"]["name"], SERVER_NAME);
    }

    #[test]
    fn notifications_initialized_produces_no_response() {
        let mut pool = ephemeral_pool();
        let notification = json!({ "jsonrpc": "2.0", "method": "notifications/initialized" });
        assert!(handle_message(&mut pool, &notification).is_none());
    }

    #[test]
    fn tools_list_exposes_all_twelve_adam_tools() {
        let mut pool = ephemeral_pool();
        let request = json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" });
        let response = handle_message(&mut pool, &request).unwrap();
        let tools = response["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 12);
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        for expected in [
            "adam_identity",
            "adam_memory_store",
            "adam_memory_query",
            "adam_beliefs",
            "adam_skills",
            "adam_evolve",
            "adam_propose_mutation",
            "adam_accept_mutation",
            "adam_reject_mutation",
            "adam_genome",
            "adam_history",
            "adam_reflect",
        ] {
            assert!(names.contains(&expected), "missing tool {expected}");
        }
    }

    #[test]
    fn tools_call_dispatches_to_the_organism_and_persists_state() {
        let mut pool = ephemeral_pool();
        let store_request = json!({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {
                "name": "adam_memory_store",
                "arguments": {
                    "kind": "episodic",
                    "content": "the build failed",
                    "origin": "tool:cargo_build",
                    "confidence": 0.8
                }
            }
        });
        let response = handle_message(&mut pool, &store_request).unwrap();
        assert_eq!(response["result"]["isError"], false);

        let reflect_request = json!({
            "jsonrpc": "2.0", "id": 5, "method": "tools/call",
            "params": { "name": "adam_reflect", "arguments": {} }
        });
        let reflect_response = handle_message(&mut pool, &reflect_request).unwrap();
        let text = reflect_response["result"]["content"][0]["text"]
            .as_str()
            .unwrap();
        assert!(text.contains("\"total_memories\":1"));
    }

    #[test]
    fn unknown_method_returns_json_rpc_error() {
        let mut pool = ephemeral_pool();
        let request = json!({ "jsonrpc": "2.0", "id": 6, "method": "not_a_real_method" });
        let response = handle_message(&mut pool, &request).unwrap();
        assert_eq!(response["error"]["code"], -32601);
    }

    #[test]
    fn distinct_organism_ids_do_not_share_memory() {
        let mut pool = ephemeral_pool();
        let store_for_alice = json!({
            "jsonrpc": "2.0", "id": 7, "method": "tools/call",
            "params": {
                "name": "adam_memory_store",
                "arguments": {
                    "organism_id": "alice",
                    "kind": "episodic",
                    "content": "alice's memory",
                    "origin": "test",
                    "confidence": 0.8
                }
            }
        });
        handle_message(&mut pool, &store_for_alice).unwrap();

        let reflect_bob = json!({
            "jsonrpc": "2.0", "id": 8, "method": "tools/call",
            "params": { "name": "adam_reflect", "arguments": { "organism_id": "bob" } }
        });
        let response = handle_message(&mut pool, &reflect_bob).unwrap();
        let text = response["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("\"total_memories\":0"));
    }
}
