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

## EVE only measures; it never decides

`EveClient::validate` returns a `FitnessResult` carrying a
`Recommendation`, not an `accept`/`reject` action. Keeping measurement and
decision separate means a governance policy could require, say, unanimous
agreement between EVE's recommendation and a human reviewer without EVE's
code changing at all.

## Fitness is measured elsewhere, and ADAM checks that it was

This design previously had ADAM score its own proposals: `adam-eve`
aggregated the results of a closure the caller supplied and reported the
pass rate as evidence. The gate below was real; the evidence behind it was
self-supplied. A component scoring its own proposed changes is not
measuring, it is asserting.

Measurement now happens in the EVE repository, over a process boundary,
and every result must satisfy `FitnessResult::is_authentic`:

- **EVE authored it.** `provenance.authored_by` must be `eve`. ADAM minting
  its own fitness is the exact failure this removes.
- **It concerns this mutation.** A result pinned to a different proposal is
  evidence about something else.
- **The comparison is symmetric.** Baseline and candidate must have run the
  same number of times, or the two sides were not measured alike.

The measurement itself is counterfactual: EVE runs its construct-validated
scenario suite twice at the same seed — once as the organism is, once with
the mutation projected onto the simulated operator — and reports the
difference. An absolute score would be uninterpretable.

The seed is derived deterministically from the mutation id and the genome
hash it applies to (`adam_eve::derive_seed`), so re-validating the same
proposal against the same genome reproduces the same experiment. With a
random seed, "we measured it again and got a different answer" would be
indistinguishable from "the mutation is marginal".

## `AmendGenome` beyond `preferences.*` requires an approving measurement

`Organism::apply` still applies `preferences.*` amendments unconditionally
— they are low-stakes and reversible. `values`, `goals`, `capabilities`
and `policies` describe the organism's core identity and behavioral
constraints, so rewriting them from a threshold-triggered proposal would
undermine the "no silent evolution of identity" requirement even though
the action already passes through governance.

Accepting one therefore requires a prior `validate_mutation` call on that
exact proposal that returned `Recommendation::Approve`. If no fitness
provider is configured at all, acceptance fails with
`NoFitnessProvider` — an organism that cannot measure a change must
refuse to make it, not make it blind.

Risk (`adam_eve::intrinsic_risk`) is a fixed function of what the proposal
touches — never its own self-reported `confidence` — precisely because
factoring in a self-reported field would let a caller lower its own risk
score by asserting higher confidence. EVE additionally withholds automatic
approval from any mutation above its `maxAutoApproveRiskBp` ceiling, and
from any mutation that improves the aggregate while regressing a single
scenario past `maxScenarioRegressionBp`: a change that lifts the mean while
destroying one case is not an improvement, and averaging would hide exactly
that.

EVE declines to score mutations with no operational signature — amending a
goal, reconciling a belief — reporting `needs_review` with that reason
rather than inventing a number. Such a proposal is escalated, not blocked:
governance still decides, now knowing simulation had nothing to say rather
than believing it had approved.

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

## Multi-organism hardening: validated ids, bounded pool, ordered checks

`OrganismPool` (Phase 10 follow-up) treats `organism_id` as
attacker-controlled input arriving straight off the MCP transport, since
the production factory interpolates it into a filesystem path. Ids are
restricted to `[A-Za-z0-9_-]{1,64}` before ever reaching the factory, and
the pool caps resident organisms at `MAX_ORGANISMS` (64), evicting the
least-recently-used one from memory (not from disk) past that cap — an
unbounded pool driven by attacker-controlled ids is a file-descriptor and
memory exhaustion vector.

`Organism::accept_mutation` runs its precondition checks
(`validate_applicable`) *before* calling `proposal.accept()`, not after.
A failing precondition (a missing or non-approving fitness measurement, an already-removed skill)
must leave the proposal retriable or rejectable rather than permanently
stuck `Accepted` with no effect applied and no audit entry — mutating the
proposal's state is the one step in this path that can't be undone, so
every fallible check runs first.

`Organism.evaluations` is pruned on both `accept_mutation` and
`reject_mutation`: once a proposal leaves the `Proposed` state, its
recorded evaluation has served its purpose and would otherwise
accumulate unboundedly across the organism's lifetime.

