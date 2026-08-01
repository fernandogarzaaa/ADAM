# Roadmap

## Shipped (this PR)

- Phase 1 — `adam-kernel`: versioned genome, hash-linked history, rollback, diffing.
- Phase 2 — `adam-memory`: SQLite-backed episodic/semantic/procedural/self memory, decay, consolidation, conflict resolution.
- Phase 3 — `adam-skills`: enforced skill lifecycle.
- Phase 4 — `adam-beliefs`: evidence-driven confidence, competing-belief resolution.
- Phase 5 — `adam-evolution`: threshold-driven proposal generation.
- Phase 6 — `adam-eve`: sandboxed simulation scoring.
- Phase 7 — `adam-organism` + `adam-mcp`: composition root and JSON-RPC 2.0 stdio MCP server exposing all 12 `adam_*` tools.
- Phase 8 — `adam-governance`: evolution rate limiting + immutable audit log.
- Phase 9 — Docker, CI, benchmarks, and this documentation set.

## Not yet built

These are explicitly out of scope for this PR and are natural next steps:

- **Real embeddings.** Swap `adam-organism::embed`'s hashed bag-of-tokens
  vector for a real embedding model, without changing `adam-memory`'s
  storage/retrieval API (it already treats embeddings as opaque `Vec<f32>`).
- **Cross-provider migration tooling.** The genome/memory/beliefs/skills
  are already provider-agnostic data, but there is no packaged
  export/import CLI yet for moving an organism's full state between LLM
  backends. The MCP server itself is the practical migration path today
  (point a new client at the same `ADAM_MEMORY_PATH` and genome store).
- **Automatic signal collection.** `adam_evolve` currently requires a
  caller to supply `EvolutionSignals`; a scheduled background job that
  derives signals from live skill/belief/memory state (chronic failures,
  retraction counts, recurring conflicts) instead of requiring an
  external caller to assemble them would make evolution proactive rather
  than on-demand.
- **Genome amendment beyond `preferences.*`.** See DESIGN.md for why this
  is intentionally restricted for now — broadening it requires a more
  deliberate policy (e.g. requiring EVE approval + a cooling-off period)
  before it's safe to automate.
- **Multi-organism / multi-tenant support.** `Organism` and the MCP
  server currently model exactly one organism per process.
- **Vector index beyond O(n) cosine scoring.** Fine at organism scale;
  would need an ANN index (HNSW, etc.) if memory volume grows into the
  millions of records.
