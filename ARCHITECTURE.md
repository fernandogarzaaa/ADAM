# Architecture

ADAM is a Cargo workspace of small, single-responsibility crates. Each
crate owns one subsystem's data and logic; nothing reaches into another
crate's internals — composition happens one layer up, in `adam-organism`.

```
                        ┌─────────────┐
                        │   adam-mcp  │  JSON-RPC 2.0 / stdio transport
                        └──────┬──────┘
                               │ wraps
                        ┌──────▼──────┐
                        │adam-organism│  composition root
                        └──────┬──────┘
        ┌───────────┬─────────┼─────────┬────────────┬─────────────┐
        │           │         │         │            │             │
  ┌─────▼────┐ ┌────▼───┐ ┌───▼────┐ ┌──▼────────┐ ┌─▼──────────┐ ┌▼────────────┐
  │adam-kernel│ │adam-   │ │adam-   │ │adam-      │ │adam-       │ │adam-        │
  │ (genome)  │ │memory  │ │skills  │ │beliefs    │ │evolution   │ │governance   │
  └───────────┘ └────────┘ └────────┘ └───────────┘ └─────┬──────┘ └─────────────┘
                                                            │ scored by
                                                      ┌─────▼──────┐
                                                      │  adam-eve  │
                                                      └────────────┘
```

## Crate responsibilities

### `adam-kernel` (Phase 1)
Owns `Genome` (identity, values, goals, beliefs, capabilities, skills,
preferences, policies) and `GenomeHistory`: an append-only, hash-linked
version chain. `rollback` never rewrites history — it commits a new version
whose content matches a prior one. `content_hash` is order-independent
(canonicalized JSON) so two genomes with identical content always hash
identically regardless of `HashMap` iteration order.

### `adam-memory` (Phase 2)
SQLite-backed (`rusqlite`, bundled) storage for four memory kinds
(episodic, semantic, procedural, self-knowledge). Every record carries
`Provenance` (origin + evidence). Retrieval is cosine-similarity over
stored embeddings, scored in-process (O(n) — an intentional simplicity
tradeoff at organism scale, see DESIGN.md). Memories decay exponentially
with disuse, consolidate from repeated episodes into semantic knowledge,
and resolve conflicts by keeping the higher-confidence record and marking
the other `superseded_by` rather than deleting it.

### `adam-skills` (Phase 3)
Skills are versioned artifacts moving through a strict lifecycle —
`Discovered → Created → Tested → Evaluated → Promoted`, with `evolve`
sending a promoted skill back to `Created` after archiving its prior
procedure as an `Improvement`. Every transition is runtime-validated;
nothing is trusted (promoted) without test evidence.

### `adam-beliefs` (Phase 4)
`Belief` confidence is never overwritten — it moves via a bounded update
rule as `Evidence` arrives (supporting evidence pulls confidence toward 1,
contradicting evidence pulls it toward 0; hitting zero retracts the
belief). Competing beliefs resolve by confidence, with the loser marked
`Superseded { by }`.

### `adam-evolution` (Phase 5)
A stateless `EvolutionEngine` turns `EvolutionSignals` (skill failures,
belief instability, recurring conflicts, genome drift) into
`EvolutionProposal`s via explicit, inspectable threshold rules. Proposals
are deliberately decoupled from the crates whose history they describe —
callers translate real state into lightweight signal structs — and they
never self-apply; they sit `Proposed` until something outside this crate
calls `accept()`/`reject()`.

### `adam-eve` (Phase 6)
`SimulationEvaluator` scores a proposal by running it through
caller-supplied sandboxed `TrialFn` closures, aggregating the pass rate
into a fitness score and combining it with proposal-kind-specific
intrinsic risk (genome amendments carry the highest baseline risk). EVE
only scores; it never decides.

### `adam-organism` (Phase 7, composition root)
`Organism` owns one `GenomeHistory`, `MemoryStore`, `SkillRegistry`,
`BeliefRegistry`, `ProposalStore`, `EvolutionEngine`, and
`GovernanceGate`. Every `adam_*` MCP tool maps to exactly one method here.
`accept_mutation` is the *only* place a mutation actually takes effect:
`RetireSkill` removes the skill from the registry, `AmendGenome` commits a
new genome version (scoped to `preferences.*`), and belief/conflict
proposals are recorded as advisory-only since they require human evidence
review that no automated rule should shortcut.

### `adam-mcp` (Phase 7, transport)
A JSON-RPC 2.0 server over stdio implementing the MCP handshake
(`initialize`, `notifications/initialized`, `tools/list`, `tools/call`).
Lifecycle-heavy tools (`adam_skills`, `adam_history`) use an `action`
field to reach their full underlying capability without growing the tool
count beyond the 12 specified names.

### `adam-governance` (Phase 8)
`GovernanceGate` is the single choke point every acceptance, rejection,
and rollback passes through. It enforces a rolling-window evolution rate
limit and writes every decision to an `AuditLog` that has no removal
method — the organism cannot change itself without leaving a permanent
record, and cannot change itself arbitrarily fast.

## Data flow: how a mutation actually happens

1. Something (an MCP client, a scheduled reflection pass) calls
   `adam_evolve` or `adam_propose_mutation` → a `EvolutionProposal` is
   recorded, status `Proposed`.
2. Optionally, EVE scores the proposal via sandboxed trials.
3. A human or governing process calls `adam_accept_mutation`.
4. `GovernanceGate::authorize_acceptance` checks the rate limit. If
   exceeded, the call fails and the proposal is untouched (still
   `Proposed`, safe to retry later).
5. The proposal is marked `Accepted`, then `Organism::apply` executes the
   concrete effect (skill removal, genome commit, or an advisory note).
6. The effect is written to the `AuditLog`.

No step in this chain can be skipped, and no step mutates organism state
silently.
