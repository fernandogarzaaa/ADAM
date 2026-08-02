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

## Shipped (follow-up)

- **Automatic signal collection.** `Organism::collect_signals`/`evolve_auto`
  derive `EvolutionSignals` from live skill failures, belief
  retractions, and recurring memory conflicts. `adam_evolve` auto-collects
  when called with no args or `{"auto": true}`; explicit signals are
  still accepted for callers that want to supply their own. Genome drift
  signals are still caller-supplied — there is no structural indicator
  for "this policy is stale."
- **Genome amendment beyond `preferences.*`, gated by EVE.** `values`,
  `goals`, `capabilities`, and `policies` can now be amended via
  `<list>.append`/`<list>.remove` fields, but only after
  `adam_propose_mutation` with `action: "evaluate"` records an EVE
  evaluation on that exact proposal recommending `Approve`. `adam-eve`
  is now actually wired into `adam-organism` (previously an unused
  workspace member) — see DESIGN.md.

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
- **Multi-organism / multi-tenant support.** `Organism` and the MCP
  server currently model exactly one organism per process.
- **Vector index beyond O(n) cosine scoring.** Fine at organism scale;
  would need an ANN index (HNSW, etc.) if memory volume grows into the
  millions of records.
