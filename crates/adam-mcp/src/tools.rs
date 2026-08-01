//! MCP tool definitions: the 12 `adam_*` tools this server exposes, each
//! with a JSON Schema input contract. Lifecycle-heavy subsystems (skills,
//! genome history) use an `action` sub-command field rather than one tool
//! per verb, keeping the exposed tool surface fixed at the 12 names
//! specified for ADAM while still reaching every underlying capability.

use serde_json::{json, Value};

pub fn tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "name": "adam_identity",
            "description": "Return the organism's current genome identity (version label, id, content hash, and full genome payload).",
            "inputSchema": { "type": "object", "properties": {} }
        }),
        json!({
            "name": "adam_memory_store",
            "description": "Store a new memory (episodic, semantic, procedural, or self_knowledge) with provenance.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "kind": { "type": "string", "enum": ["episodic", "semantic", "procedural", "self_knowledge"] },
                    "content": { "type": "string" },
                    "origin": { "type": "string" },
                    "evidence": { "type": "array", "items": { "type": "string" } },
                    "confidence": { "type": "number" },
                    "decay_rate": { "type": "number" }
                },
                "required": ["kind", "content", "origin", "confidence"]
            }
        }),
        json!({
            "name": "adam_memory_query",
            "description": "Retrieve memories most similar to a query, optionally filtered by kind.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "kind": { "type": "string", "enum": ["episodic", "semantic", "procedural", "self_knowledge"] },
                    "top_k": { "type": "integer" }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "adam_beliefs",
            "description": "List currently active beliefs, or form a new belief from evidence when 'statement' is provided.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "statement": { "type": "string" },
                    "origin": { "type": "string", "enum": ["observation", "memory", "reasoning", "external_source", "user_assertion"] },
                    "description": { "type": "string" },
                    "weight": { "type": "number" }
                }
            }
        }),
        json!({
            "name": "adam_skills",
            "description": "Manage the skill lifecycle. 'action' selects the operation: list (default), discover, define_procedure, record_test, evaluate, promote, evolve.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["list", "discover", "define_procedure", "record_test", "evaluate", "promote", "evolve"] },
                    "name": { "type": "string" },
                    "description": { "type": "string" },
                    "trigger_conditions": { "type": "array", "items": { "type": "string" } },
                    "procedure": { "type": "string" },
                    "dependencies": { "type": "array", "items": { "type": "string" } },
                    "passed": { "type": "boolean" },
                    "note": { "type": "string" },
                    "threshold": { "type": "number" },
                    "reason": { "type": "string" },
                    "new_procedure": { "type": "string" }
                }
            }
        }),
        json!({
            "name": "adam_evolve",
            "description": "Analyze signals (skill failures, belief instability, recurring conflicts, genome drift) and generate evolution proposals. Proposals never auto-apply.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "skill_failures": { "type": "array" },
                    "belief_instabilities": { "type": "array" },
                    "recurring_conflicts": { "type": "array" },
                    "genome_drifts": { "type": "array" }
                }
            }
        }),
        json!({
            "name": "adam_propose_mutation",
            "description": "Manually record an evolution proposal (retire_skill, reconcile_belief, investigate_conflict, or amend_genome).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "kind": { "type": "string", "enum": ["retire_skill", "reconcile_belief", "investigate_conflict", "amend_genome"] },
                    "skill_name": { "type": "string" },
                    "statement": { "type": "string" },
                    "topic": { "type": "string" },
                    "field": { "type": "string" },
                    "current_value": { "type": "string" },
                    "suggested_value": { "type": "string" },
                    "rationale": { "type": "string" },
                    "evidence": { "type": "array", "items": { "type": "string" } },
                    "confidence": { "type": "number" }
                },
                "required": ["kind", "rationale"]
            }
        }),
        json!({
            "name": "adam_accept_mutation",
            "description": "Accept a pending proposal and apply its concrete effect (the only path by which the organism actually changes).",
            "inputSchema": {
                "type": "object",
                "properties": { "proposal_id": { "type": "string" } },
                "required": ["proposal_id"]
            }
        }),
        json!({
            "name": "adam_reject_mutation",
            "description": "Reject a pending proposal without applying any effect.",
            "inputSchema": {
                "type": "object",
                "properties": { "proposal_id": { "type": "string" } },
                "required": ["proposal_id"]
            }
        }),
        json!({
            "name": "adam_genome",
            "description": "Return the current genome payload (identity, values, goals, beliefs, capabilities, skills, preferences, policies).",
            "inputSchema": { "type": "object", "properties": {} }
        }),
        json!({
            "name": "adam_history",
            "description": "Inspect or act on genome version history. 'action' selects: list (default), diff (needs from/to), rollback (needs target/reason).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["list", "diff", "rollback"] },
                    "from": { "type": "string" },
                    "to": { "type": "string" },
                    "target": { "type": "string" },
                    "reason": { "type": "string" }
                }
            }
        }),
        json!({
            "name": "adam_reflect",
            "description": "Return a point-in-time self-assessment summarizing genome, memory, beliefs, skills, and pending proposals.",
            "inputSchema": { "type": "object", "properties": {} }
        }),
    ]
}
