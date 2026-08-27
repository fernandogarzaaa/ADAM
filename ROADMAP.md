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
- **ANN vector index.** `adam_memory::AnnIndex` (HNSW via
  `instant-distance`) is a caller-built snapshot alternative to the exact
  O(n) cosine scan. `Organism::memory_query_ann` / `adam_memory_query`
  with `approximate: true` use it. It does not update incrementally —
  each call builds a fresh snapshot from current records — so it trades
  index-build cost for approximate results; callers with high query
  volume relative to write volume should build once via
  `MemoryStore::build_ann_index` and reuse it.
- **Multi-organism / multi-tenant support.** `adam_mcp::OrganismPool`
  lazily creates and holds one `Organism` per `organism_id`. Every
  `tools/call` accepts an optional `organism_id` argument (default
  `"default"`, which preserves the original `ADAM_MEMORY_PATH`/
  `ADAM_GENOME_PATH` single-organism behavior); other ids get their
  state under `ADAM_DATA_DIR` (default `.`) as
  `<id>_memory.db`/`<id>_genome.json`. There is no cross-process
  coordination — this is in-process multi-tenancy for one server, not a
  distributed system. `organism_id` is validated against
  `[A-Za-z0-9_-]{1,64}` before use (it is interpolated into a filesystem
  path) and the pool holds at most 64 organisms in memory at once,
  evicting the least-recently-used past that cap; eviction only drops the
  in-memory copy, so a later request for an evicted id transparently
  reloads it via the factory.
- **Hardening pass on the multi-organism/EVE-gating follow-ups.** EVE's
  risk scoring no longer depends on a proposal's self-reported
  `confidence` (see DESIGN.md), evaluations now require a minimum trial
  count to avoid being satisfied by a single self-reported "success",
  `accept_mutation` validates preconditions before mutating proposal
  state instead of after, no-op genome list amendments no longer commit
  an empty-diff genome version, belief-retraction signal aggregation is
  deterministic instead of depending on `HashMap` iteration order, and
  `Organism.evaluations` is pruned as proposals are accepted/rejected
  instead of growing unboundedly.

## Shipped (follow-up, continued)

- **Real embeddings.** `adam-organism::embed` now runs `all-MiniLM-L6-v2`
  locally via `fastembed`/`ort` (an ONNX Runtime session), replacing the
  hashed bag-of-tokens placeholder. `adam-memory`'s storage/retrieval API
  needed no change — it already treats embeddings as opaque `Vec<f32>`.
  The model is fetched from Hugging Face on first use and cached under
  `ADAM_EMBEDDING_CACHE_DIR` (or `fastembed`'s own default cache dir);
  `embed` is now fallible (`Result<Vec<f32>, EmbeddingError>`), and model
  load/inference failure surfaces as `OrganismError::Embedding` rather than
  silently degrading to a placeholder vector. The two tests that exercise
  the real model are `#[ignore]`d by default so `cargo test --workspace`
  stays hermetic; run them explicitly with `cargo test -p adam-organism --
  --ignored` when the model is reachable.

## Not yet built

These are explicitly out of scope for this PR and are natural next steps:

- **Cross-provider migration tooling.** The genome/memory/beliefs/skills
  are already provider-agnostic data, but there is no packaged
  export/import CLI yet for moving an organism's full state between LLM
  backends. The MCP server itself is the practical migration path today
  (point a new client at the same `ADAM_MEMORY_PATH` and genome store).
