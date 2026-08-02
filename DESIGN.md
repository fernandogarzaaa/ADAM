# Design Decisions

This document records the *why* behind choices that aren't obvious from
the code, so future changes don't accidentally undo an intentional
tradeoff.

## Why 9 small crates instead of one large one

Each subsystem (genome, memory, skills, beliefs, evolution, EVE,
organism, MCP, governance) has its own error type, its own test suite,
and its own reason to change. Splitting them lets each phase be built,
tested, and reviewed as a complete, independently-compiling unit —
matching the "every phase must result in integrated working software"
requirement — rather than one crate accumulating partial, cross-cutting
state across phases.

## Append-only history everywhere

`GenomeHistory`, `MemoryStore` (via `superseded_by`), `Skill` (via
`improvements`), and `AuditLog` all share one pattern: nothing is ever
deleted or overwritten in place. Rollback commits a new version; conflict
resolution marks a loser `superseded`; skill evolution archives the prior
procedure; governance actions are appended, never edited. This is a
direct consequence of the "no silent evolution" requirement — an
organism that can quietly erase its own history could also quietly erase
evidence of a bad decision.

## Lightweight deterministic embeddings instead of an ML model

`adam-organism::embed` is a 64-dimension bag-of-hashed-tokens vector, not
a real embedding model. Pulling in an embedding runtime (ONNX, a hosted
API, etc.) would add a large dependency surface and an external service
dependency to a project whose acceptance criteria are about identity,
memory, and governance semantics — not retrieval quality. The design
isolates this choice to one function (`embedding.rs`) specifically so a
real embedder can be swapped in later without touching `adam-memory`'s
storage or retrieval logic, which already treats embeddings as opaque
`Vec<f32>`.

## `adam-evolution` never imports `adam-skills`/`adam-beliefs`/`adam-memory`

The evolution engine's signal types (`SkillFailureSignal`,
`BeliefInstabilitySignal`, etc.) are intentionally its own lightweight
structs, not the real `Skill`/`Belief`/`MemoryRecord` types. This keeps
`adam-evolution` unit-testable in complete isolation and avoids a
dependency cycle (evolution analyzes skills/beliefs, but skills/beliefs
should never need to know evolution exists). The composition root
(`adam-organism`, or a future scheduled-reflection job) is responsible
for translating real state into signals.

## EVE only scores; it never decides

`SimulationEvaluator::evaluate` returns a `Recommendation`, not an
`accept`/`reject` action. Keeping evaluation and decision-making as
separate steps means a future governance policy could require, say,
unanimous agreement between EVE's recommendation and a human reviewer
without EVE's code needing to change at all.

## `AmendGenome` beyond `preferences.*` requires a prior EVE approval

`Organism::apply` still applies `preferences.*` amendments unconditionally
— they're low-stakes and reversible. `values`, `goals`, `capabilities`,
and `policies` describe the organism's core identity and behavioral
constraints, so automatically rewriting them from a threshold-triggered
proposal would undermine the "no silent evolution of identity" spirit of
the safety requirements even though the action itself goes through
governance. Rather than refusing these fields outright, `Organism` now
requires a prior `evaluate_mutation`/`evaluate_mutation_from_trials` call
on that exact proposal that resulted in `Recommendation::Approve` — real
trial evidence, not the proposal's self-reported confidence — before
`accept_mutation` will apply it. This is the intentional "EVE approval +
a de facto cooling-off period" policy this document previously described
as a future requirement: a proposal must clear evaluation as a distinct,
auditable step before it can be accepted, and `AmendGenome`'s baseline
risk score (see "EVE only scores" above) is deliberately high enough that
even a flawless trial run needs high proposal confidence to clear the
default risk ceiling, rather than being rubber-stamped by fitness alone.
`beliefs` and `skills` genome fields remain unsupported entirely — they
are redundant with the dedicated `adam-beliefs`/`adam-skills` subsystems
and amending them via the genome would create two sources of truth for
the same state.

## Evolution rate limiting is a rolling window, not a fixed quota

`RateLimiter` counts acceptances within `now - window`, not "N per
calendar day." A fixed daily quota resets predictably at midnight, which
an aggressive proposal pipeline could exploit by batching acceptances
right before and after the reset. A rolling window closes that gap at
the cost of slightly more bookkeeping (a `Vec<DateTime<Utc>>` scan per
check) — acceptable at organism scale, same tradeoff `adam-memory`
already makes for retrieval.

## Belief and conflict-investigation proposals are advisory-only

`Organism::apply` has real, mutating effects for `RetireSkill` and
`AmendGenome`, but `ReconcileBelief` and `InvestigateConflict` only
produce a note. Automatically deciding *which* belief is correct or
*what* a recurring conflict means requires judgment no threshold rule
should have — the proposal correctly flags the problem, but resolving it
is left to whatever reviews the audit log.
