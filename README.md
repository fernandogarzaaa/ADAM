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
├── adam-protocol    CP/1 binding — the wire contract shared with EVE and AXIOM
├── adam-eve         CP/1 client for EVE — counterfactual fitness measurement
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

## Installing as a Claude Code plugin

This repository is itself a Claude Code plugin marketplace
(`.claude-plugin/marketplace.json`), so it can be added the same way any
other plugin marketplace is — from a GitHub repo or git URL — without
cloning it yourself first:

```
/plugin marketplace add fernandogarzaaa/ADAM
/plugin install adam@adam
```

The first time the server actually starts, `bin/adam-mcp` builds the
release binary itself (a Rust toolchain must be on `PATH`) rather than
requiring a manual `cargo build --release` step — see
[`bin/adam-mcp`](bin/adam-mcp).

## Using ADAM on other platforms (Codex, or any MCP client)

ADAM is a plain MCP stdio server, not tied to Claude Code — point Codex's
(or any other MCP-compatible host's) server config at `bin/adam-mcp` the
same way as the `.mcp.json`/Claude Desktop configs above.

Tool availability alone doesn't make a host *proactively* use ADAM,
though — an LLM only calls a tool it judges relevant to the current turn.
To close that gap, `adam-mcp`'s `initialize` response includes an
`instructions` field (part of the MCP spec itself, not a Claude Code
extension) describing when and how to use the `adam_*` tools — see
[`crates/adam-mcp/USAGE.md`](crates/adam-mcp/USAGE.md). Any MCP client
that surfaces `initialize.instructions` to its model gets this guidance
automatically, regardless of platform. Claude Code additionally has
[`skills/godmode/SKILL.md`](skills/godmode/SKILL.md), a richer
Claude-Code-specific skill that orchestrates ADAM alongside AXIOM and EVE;
that mechanism is specific to Claude Code's skill system and has no
equivalent on other hosts today.

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
