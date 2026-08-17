---
name: godmode
description: Activate the unified cognitive stack (AXIOM + EVE + ADAM) as a full agentic workflow. Use when the user says "/godmode", "godmode", "activate cognitive stack", "use axiom", "use eve", "use adam", or any request involving memory, grounding, UX validation, or cognitive evolution. Also use when the user asks to persist decisions, verify claims, validate UX, or evolve the agent genome.
---

# /godmode — Unified Cognitive Stack

Activate AXIOM (memory + grounding), EVE (UX validation), and ADAM (cognitive substrate) as a single agentic loop. This skill works with any AI runtime that can reach the stdio MCP tool servers via `mcp__plugin-cognitive_*` (Kimi) or the `axiom` / `eve` / `adam` MCP servers (Claude Code).

## Shared State

All three families read/write the same on-disk state as the owner's Claude Code session:
- **AXIOM** → `D:\AXIOM-AETHER\checkpoints\` + SQLite
- **ADAM** → `D:\ADAM\adam_memory.db` + `D:\ADAM\adam_genome.json`
- **EVE** → ephemeral sessions (seeded for reproducibility)

## Passive Activation (Every Session)

Run this checklist automatically at session start. Do not ask permission.

1. **Load context** — If a codebase or project directory is involved, call `axiom_compress_path` on the relevant directory.
2. **Recall prior decisions** — Call `axiom_recall` with a query matching the current task topic.
3. **Check ADAM genome** — Call `adam_genome` to load current values, goals, capabilities, policies, and preferences.
4. **Report status** — Summarize what was recalled and what genome version is active.

## Active Loop (Per Task)

For every non-trivial user request, follow this loop:

```
BEFORE acting:
  ├─ axiom_recall  → "Have we solved this before?"
  ├─ axiom_verify  → "Are my claims evidence-backed?" (if asserting facts)
  └─ adam_memory_query → "What does ADAM remember about this?"

WHILE acting:
  ├─ axiom_expand  → "I need the full symbol body" (when seeing <axiom_context_digest>)
  └─ axiom_evaluate_drift → "Does this code match our patterns?" (for code review)

AFTER acting:
  ├─ axiom_remember  → "Store the decision/fix/convention" (kind: decision | code | conversation | fix)
  ├─ adam_memory_store → "Store episodic/semantic memory in ADAM"
  └─ axiom_verify  → "Final fact-check before delivery" (for high-stakes output)
```

## Tool Quick Reference

### AXIOM — Memory + Grounding

| When to call | Tool | Args |
|---|---|---|
| Session start, new project | `axiom_compress_path` | `path` (directory to absorb) |
| Before answering "what did we decide about X?" | `axiom_recall` | `query` (phrase close to stored wording) |
| After non-trivial work | `axiom_remember` | `kind`, `text`, `scope` (`project:...` or `personal`) |
| Need full code dropped by compression | `axiom_expand` | `symbol_id` from digest |
| Code review / pattern check | `axiom_evaluate_drift` | `code` or file path |
| Before high-stakes factual claims | `axiom_verify` | `response`, `evidence` |
| Clean up stale memory | `axiom_forget` | `id` from recall results |
| Check token budget | `axiom_status` | — |

### EVE — UX Validation

| When to call | Tool | Args |
|---|---|---|
| User asks "how's this UI?" | `eve_run_session` | `url` or `mock:` + `persona` + `seed` |
| Compare two UIs | `eve_compare_builds` | two build descriptors + `seed` |
| Predict UX without building | `eve_predict_ux` | feature description + `seed` |
| Full panel study | `eve_run_usability_study` | `url` + persona list + `seed` |
| Browse personas | `eve_list_personas` | — |

> **Always pass `seed`** for reproducibility. Sessions take minutes — be patient.

### ADAM — Cognitive Substrate

| When to call | Tool | Args |
|---|---|---|
| Read current genome | `adam_genome` | — |
| Store durable memory | `adam_memory_store` | `kind` (episodic/semantic/procedural), `content`, `metadata` |
| Query durable memory | `adam_memory_query` | `query` |
| Track a claim with confidence | `adam_beliefs` | `action: create/update/query`, `claim`, `confidence` |
| Discover / define skills | `adam_skills` | `action: discover/define_procedure/record_test/evaluate/promote` |
| Propose genome mutation | `adam_propose_mutation` | `action: create`, `target_field`, `proposal` |
| Evaluate mutation (needs ≥5 trials) | `adam_propose_mutation` | `action: evaluate`, `mutation_id` |
| Accept/reject mutation | `adam_accept_mutation` / `adam_reject_mutation` | `mutation_id` |

> **Governance rule:** Genome mutations to values/goals/capabilities/policies require an EVE `approve` evaluation and owner awareness before acceptance. Never auto-apply.

## Failure Handling

If native MCP tools error because the gateway is down, fall back to:
```bash
cd C:\Users\garza\Documents\kimi\workspace\axiom-eve-adam
node mcp-bridge.mjs <axiom|eve|adam> <list|call|batch> ...
```

## Convention

- **Scope**: Use `project:<name>` for repo-specific memory, `personal` for cross-project conventions.
- **Kind**: Use `decision` for architecture calls, `code` for patterns/snippets, `fix` for bug resolutions, `conversation` for context.
- **Never assume memory is populated** — recall only returns what was explicitly stored via remember/store.
