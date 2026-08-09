//! Dispatch: turns a `tools/call` request into an [`adam_organism::Organism`]
//! method call and a JSON result, independent of the JSON-RPC transport so
//! it can be unit tested directly.

use adam_beliefs::{Belief, EvidenceOrigin};
use adam_evolution::{EvolutionProposal, EvolutionSignals, ProposalKind};
use adam_memory::MemoryKind;
use adam_organism::Organism;
use adam_skills::Skill;
use serde_json::{json, Value};
use uuid::Uuid;

pub fn call_tool(organism: &mut Organism, name: &str, args: &Value) -> Result<Value, String> {
    match name {
        "adam_identity" => Ok(json!(organism.identity())),
        "adam_memory_store" => memory_store(organism, args),
        "adam_memory_query" => memory_query(organism, args),
        "adam_beliefs" => beliefs(organism, args),
        "adam_skills" => skills(organism, args),
        "adam_evolve" => evolve(organism, args),
        "adam_propose_mutation" => propose_mutation(organism, args),
        "adam_accept_mutation" => accept_mutation(organism, args),
        "adam_reject_mutation" => reject_mutation(organism, args),
        "adam_genome" => Ok(json!(organism.genome())),
        "adam_history" => history(organism, args),
        "adam_reflect" => organism
            .reflect()
            .map(|r| json!(r))
            .map_err(|e| e.to_string()),
        other => Err(format!("unknown tool '{other}'")),
    }
}

fn str_field(args: &Value, key: &str) -> Result<String, String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("missing required string field '{key}'"))
}

fn opt_str(args: &Value, key: &str) -> Option<String> {
    args.get(key).and_then(Value::as_str).map(str::to_string)
}

fn opt_f32(args: &Value, key: &str, default: f32) -> f32 {
    args.get(key)
        .and_then(Value::as_f64)
        .map(|v| v as f32)
        .unwrap_or(default)
}

fn opt_u64(args: &Value, key: &str, default: u64) -> u64 {
    args.get(key).and_then(Value::as_u64).unwrap_or(default)
}

fn opt_strings(args: &Value, key: &str) -> Vec<String> {
    args.get(key)
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn parse_uuid(args: &Value, key: &str) -> Result<Uuid, String> {
    let raw = str_field(args, key)?;
    Uuid::parse_str(&raw).map_err(|e| format!("invalid uuid for '{key}': {e}"))
}

fn memory_store(organism: &mut Organism, args: &Value) -> Result<Value, String> {
    let kind =
        MemoryKind::parse(&str_field(args, "kind")?).ok_or_else(|| "invalid 'kind'".to_string())?;
    let content = str_field(args, "content")?;
    let origin = str_field(args, "origin")?;
    let evidence = opt_strings(args, "evidence");
    let confidence = opt_f32(args, "confidence", 0.5);
    let decay_rate = opt_f32(args, "decay_rate", 0.0);

    let id = organism
        .memory_store(kind, &content, &origin, evidence, confidence, decay_rate)
        .map_err(|e| e.to_string())?;
    Ok(json!({ "id": id }))
}

fn memory_query(organism: &mut Organism, args: &Value) -> Result<Value, String> {
    let query = str_field(args, "query")?;
    let kind = opt_str(args, "kind").and_then(|k| MemoryKind::parse(&k));
    let top_k = opt_u64(args, "top_k", 5) as usize;
    let approximate = args.get("approximate").and_then(Value::as_bool) == Some(true);

    let results = if approximate {
        organism.memory_query_ann(&query, kind, top_k)
    } else {
        organism.memory_query(&query, kind, top_k)
    }
    .map_err(|e| e.to_string())?;
    Ok(json!(results
        .into_iter()
        .map(|(record, score)| json!({ "record": record, "score": score }))
        .collect::<Vec<_>>()))
}

fn beliefs(organism: &mut Organism, args: &Value) -> Result<Value, String> {
    if let Some(statement) = opt_str(args, "statement") {
        let origin = match opt_str(args, "origin").as_deref() {
            Some("observation") => EvidenceOrigin::Observation,
            Some("memory") => EvidenceOrigin::Memory,
            Some("external_source") => EvidenceOrigin::ExternalSource,
            Some("user_assertion") => EvidenceOrigin::UserAssertion,
            _ => EvidenceOrigin::Reasoning,
        };
        let description = opt_str(args, "description").unwrap_or_default();
        let weight = opt_f32(args, "weight", 0.5);
        let belief =
            Belief::form(statement, origin, description, weight).map_err(|e| e.to_string())?;
        let id = organism.form_belief(belief);
        return Ok(json!({ "id": id }));
    }
    Ok(json!(organism.beliefs().all_active()))
}

fn skills(organism: &mut Organism, args: &Value) -> Result<Value, String> {
    let action = opt_str(args, "action").unwrap_or_else(|| "list".to_string());
    match action.as_str() {
        "list" => Ok(json!(organism
            .skills()
            .by_stage(adam_skills::SkillStage::Discovered)
            .into_iter()
            .chain(organism.skills().by_stage(adam_skills::SkillStage::Created))
            .chain(organism.skills().by_stage(adam_skills::SkillStage::Tested))
            .chain(
                organism
                    .skills()
                    .by_stage(adam_skills::SkillStage::Evaluated)
            )
            .chain(
                organism
                    .skills()
                    .by_stage(adam_skills::SkillStage::Promoted)
            )
            .chain(
                organism
                    .skills()
                    .by_stage(adam_skills::SkillStage::Rejected)
            )
            .collect::<Vec<_>>())),
        "discover" => {
            let name = str_field(args, "name")?;
            let description = str_field(args, "description")?;
            let triggers = opt_strings(args, "trigger_conditions");
            let skill = Skill::discover(name, description, triggers);
            let id = organism.register_skill(skill);
            Ok(json!({ "id": id }))
        }
        "define_procedure" | "record_test" | "evaluate" | "promote" | "evolve" => {
            let name = str_field(args, "name")?;
            let id = organism
                .skills()
                .find_by_name(&name)
                .map(|s| s.id)
                .ok_or_else(|| format!("skill '{name}' not found"))?;
            let skill = organism
                .skills_mut()
                .get_mut(id)
                .expect("looked up by id above");
            match action.as_str() {
                "define_procedure" => {
                    skill
                        .define_procedure(
                            str_field(args, "procedure")?,
                            opt_strings(args, "dependencies"),
                        )
                        .map_err(|e| e.to_string())?;
                }
                "record_test" => {
                    let passed = args.get("passed").and_then(Value::as_bool).unwrap_or(false);
                    skill
                        .record_test(passed, opt_str(args, "note").unwrap_or_default())
                        .map_err(|e| e.to_string())?;
                }
                "evaluate" => {
                    skill
                        .evaluate(opt_f32(args, "threshold", 0.5))
                        .map_err(|e| e.to_string())?;
                }
                "promote" => {
                    skill.promote().map_err(|e| e.to_string())?;
                }
                "evolve" => {
                    skill
                        .evolve(
                            str_field(args, "reason")?,
                            str_field(args, "new_procedure")?,
                        )
                        .map_err(|e| e.to_string())?;
                }
                _ => unreachable!(),
            }
            Ok(json!(skill))
        }
        other => Err(format!("unknown skills action '{other}'")),
    }
}

fn evolve(organism: &mut Organism, args: &Value) -> Result<Value, String> {
    let auto_requested = args.is_null()
        || args == &json!({})
        || args.get("auto").and_then(Value::as_bool) == Some(true);

    let ids = if auto_requested {
        organism.evolve_auto().map_err(|e| e.to_string())?
    } else {
        let signals: EvolutionSignals = serde_json::from_value(args.clone()).unwrap_or_default();
        organism.evolve(&signals)
    };

    let proposals: Vec<_> = ids
        .iter()
        .filter_map(|id| organism.proposals().get(*id))
        .collect();
    Ok(json!(proposals))
}

fn propose_mutation(organism: &mut Organism, args: &Value) -> Result<Value, String> {
    let action = opt_str(args, "action").unwrap_or_else(|| "create".to_string());
    match action.as_str() {
        "create" => propose_mutation_create(organism, args),
        // "evaluate" remains accepted as an alias so an existing client keeps
        // working; it now performs a real measurement rather than scoring
        // caller-supplied trial outcomes.
        "validate" | "evaluate" => validate_mutation(organism, args),
        other => Err(format!("unknown propose_mutation action '{other}'")),
    }
}

fn propose_mutation_create(organism: &mut Organism, args: &Value) -> Result<Value, String> {
    let kind_str = str_field(args, "kind")?;
    let rationale = str_field(args, "rationale")?;
    let evidence = opt_strings(args, "evidence");
    let confidence = opt_f32(args, "confidence", 0.5);

    let kind = match kind_str.as_str() {
        "retire_skill" => ProposalKind::RetireSkill {
            skill_name: str_field(args, "skill_name")?,
        },
        "reconcile_belief" => ProposalKind::ReconcileBelief {
            statement: str_field(args, "statement")?,
        },
        "investigate_conflict" => ProposalKind::InvestigateConflict {
            topic: str_field(args, "topic")?,
        },
        "amend_genome" => ProposalKind::AmendGenome {
            field: str_field(args, "field")?,
            current_value: str_field(args, "current_value")?,
            suggested_value: str_field(args, "suggested_value")?,
        },
        other => return Err(format!("unknown proposal kind '{other}'")),
    };

    let proposal = EvolutionProposal::new(kind, rationale, evidence, confidence);
    let id = organism.propose_mutation(proposal);
    Ok(json!({ "id": id }))
}

/// Measure a pending proposal in EVE.
///
/// The previous implementation took trial outcomes *from the caller* and
/// scored them, which meant an MCP client could hand ADAM whatever evidence
/// suited it and have the organism treat it as a sandboxed measurement. That
/// is now impossible: measurement happens in EVE, over a process boundary, and
/// ADAM refuses any result EVE did not author. The client chooses when to
/// validate; it does not supply the verdict.
fn validate_mutation(organism: &mut Organism, args: &Value) -> Result<Value, String> {
    let id = parse_uuid(args, "proposal_id")?;
    let correlation_id =
        opt_str(args, "correlation_id").unwrap_or_else(adam_organism::new_correlation_id);
    let result = organism
        .validate_mutation(id, &correlation_id)
        .map_err(|e| e.to_string())?;
    Ok(json!({ "correlation_id": correlation_id, "fitness": result }))
}

fn accept_mutation(organism: &mut Organism, args: &Value) -> Result<Value, String> {
    let id = parse_uuid(args, "proposal_id")?;
    let correlation_id =
        opt_str(args, "correlation_id").unwrap_or_else(adam_organism::new_correlation_id);
    let effect = organism
        .accept_mutation(id, &correlation_id)
        .map_err(|e| e.to_string())?;
    Ok(json!({ "correlation_id": correlation_id, "effect": effect }))
}

fn reject_mutation(organism: &mut Organism, args: &Value) -> Result<Value, String> {
    let id = parse_uuid(args, "proposal_id")?;
    let correlation_id =
        opt_str(args, "correlation_id").unwrap_or_else(adam_organism::new_correlation_id);
    let reason = opt_str(args, "reason").unwrap_or_else(|| "no reason supplied".to_string());
    organism
        .reject_mutation(id, &reason, &correlation_id)
        .map_err(|e| e.to_string())?;
    Ok(json!({ "rejected": id, "correlation_id": correlation_id, "reason": reason }))
}

fn history(organism: &mut Organism, args: &Value) -> Result<Value, String> {
    let action = opt_str(args, "action").unwrap_or_else(|| "list".to_string());
    match action.as_str() {
        "list" => Ok(json!(organism.history())),
        "audit" => Ok(json!(organism.audit_log())),
        "diff" => {
            let from = parse_uuid(args, "from")?;
            let to = parse_uuid(args, "to")?;
            let diff = organism.diff(from, to).map_err(|e| e.to_string())?;
            Ok(json!(diff))
        }
        "rollback" => {
            let target = parse_uuid(args, "target")?;
            let reason = opt_str(args, "reason").unwrap_or_else(|| "rollback via MCP".to_string());
            let new_version = organism
                .rollback(target, reason)
                .map_err(|e| e.to_string())?;
            Ok(json!({ "new_version": new_version }))
        }
        other => Err(format!("unknown history action '{other}'")),
    }
}
