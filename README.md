# ADAM — Autonomous Cognitive Evolution Layer

ADAM is a persistent, MCP-native cognitive substrate for LLM-based agents.
Where a typical chat session forgets everything between turns, ADAM gives an
organism a versioned identity, durable memory, evolving skills, an epistemic
belief system, and a governed mechanism for proposing and applying changes to
itself — all exposed as MCP tools any MCP-compatible client (Claude Desktop,
Codex, ChatGPT, etc.) can call.

## Why "organism," not "chatbot"

A chatbot answers a prompt. An organism:

- **Persists identity** across sessions and even across different LLM
  providers (the genome/memory/skills/beliefs live independently of any one
  model backend).
- **Remembers** what happened, distinguishing specific episodes from
  generalized knowledge, procedures, and self-knowledge.
- **Forms beliefs** from evidence, with confidence that rises and falls as
  evidence accumulates, and competing beliefs that resolve by confidence
  rather than by whichever was said last.
- **Evolves skills** through an enforced lifecycle — nothing is trusted
  until it has been tested and evaluated.
- **Proposes changes to itself**, but never applies them silently: every
  mutation requires an explicit accept, is rate-limited, and is written to
  an immutable audit log.

## Architecture at a glance

```
crates/
├── adam-kernel      Phase 1 — versioned genome (identity/values/goals/…)
├── adam-memory      Phase 2 — episodic/semantic/procedural/self memory
├── adam-skills      Phase 3 — skill lifecycle (discover → … → evolve)
├── adam-beliefs     Phase 4 — epistemic state with evidence-driven confidence
├── adam-evolution   Phase 5 — signals → EvolutionProposals (never auto-applied)
├── adam-eve         Phase 6 — sandboxed simulation scoring for proposals
├── adam-organism    Phase 7 — composition root wiring every subsystem together
├── adam-mcp         Phase 7 — JSON-RPC 2.0 stdio MCP server (the adam_* tools)
└── adam-governance  Phase 8 — rate limiting + immutable audit log
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for how these compose and
[DESIGN.md](DESIGN.md) for the reasoning behind specific tradeoffs.

## Building

```bash
cargo build --release
cargo test --workspace
```

## Running the MCP server

```bash
cargo run --release -p adam-mcp
```

The server speaks newline-delimited JSON-RPC 2.0 over stdio (the standard
MCP stdio transport): `initialize` → `notifications/initialized` →
`tools/list` → `tools/call`. Point `ADAM_MEMORY_PATH` at a file path to
persist memory across restarts (defaults to `adam_memory.db` in the working
directory; use `:memory:` for an ephemeral instance), and `ADAM_GENOME_PATH`
at a file path to persist genome identity/history across restarts (defaults
to `adam_genome.json`). Every `tools/call` also accepts an optional
`organism_id` argument for multi-organism use (defaults to `"default"`,
which uses the two env vars above); other ids get their state under
`ADAM_DATA_DIR` (defaults to `.`) as `<id>_memory.db`/`<id>_genome.json`.

### Claude Desktop configuration

```json
{
  "mcpServers": {
    "adam": {
      "command": "/path/to/target/release/adam-mcp",
      "env": { "ADAM_MEMORY_PATH": "/path/to/adam_memory.db" }
    }
  }
}
```

## The 12 tools

| Tool | Purpose |
|---|---|
| `adam_identity` | Current genome identity snapshot |
| `adam_memory_store` | Store an episodic/semantic/procedural/self memory |
| `adam_memory_query` | Similarity-retrieve memories (`approximate: true` for HNSW ANN search) |
| `adam_beliefs` | List active beliefs, or form a new one from evidence |
| `adam_skills` | Full skill lifecycle (discover → … → evolve) via `action` |
| `adam_evolve` | Analyze signals into evolution proposals |
| `adam_propose_mutation` | Manually record a proposal |
| `adam_accept_mutation` | Accept a proposal and apply its effect |
| `adam_reject_mutation` | Reject a proposal |
| `adam_genome` | Current genome payload |
| `adam_history` | Version history, diffs, rollback, and the audit log via `action` |
| `adam_reflect` | Point-in-time self-assessment summary |

## Docker

```bash
docker build -t adam-mcp .
docker run -i -v adam-data:/data -e ADAM_MEMORY_PATH=/data/adam_memory.db adam-mcp
```

## License

MIT — see [LICENSE](LICENSE).
