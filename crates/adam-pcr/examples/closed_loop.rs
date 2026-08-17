//! The whole loop, once, against a real environment and a real EVE.
//!
//! Run with:
//!
//! ```text
//! cargo run -p adam-pcr --example closed_loop
//! ```
//!
//! An example rather than a test because it spawns EVE, which takes minutes,
//! and because its output is evidence to be read rather than an assertion to
//! be satisfied. It prints JSON and exits 0 whatever it finds — including when
//! the organism fails to adapt. A harness that could only report success would
//! not be able to falsify anything.
//!
//! # The control
//!
//! A second organism does the same work on an identical workspace and is never
//! offered a mutation. Without it, "the second turn scored higher" has an
//! obvious rival explanation — that any second turn scores higher — and the
//! experiment could not tell adaptation from ordering.

use std::sync::Arc;

use adam_eve::{Cp1Subprocess, EveClient};
use adam_eventlog::SqliteEventLog;
use adam_evolution::{EvolutionProposal, ProposalKind};
use adam_organism::Organism;
use adam_pcr::{act_and_record, action_for, Workspace};

/// The records the organism finds. It did not write these, and the third one
/// is the reason anything happens.
const RECORDS: &[(&str, &str)] = &[
    ("01-alpha.rec", "name=alpha\ncount=3\n"),
    ("02-bravo.rec", "name=bravo\ncount=1\n"),
    (
        "03-charlie.rec",
        "name=charlie\nthis line has no separator\n",
    ),
    ("04-delta.rec", "name=delta\ncount=7\n"),
];

const POLICY: &str = "verify records before processing them";

fn eve_script() -> String {
    std::env::var("EVE_CP1_SCRIPT")
        .unwrap_or_else(|_| "D:/experience-validation-engine/bin/eve-cp1.js".to_string())
}

fn scratch(label: &str) -> std::path::PathBuf {
    // `GENESIS_RUN_DIR` keeps a run's event store where it can be audited
    // afterwards. Without it the run lands in the system temp directory, which
    // is fine for a smoke test and useless as evidence.
    if let Ok(dir) = std::env::var("GENESIS_RUN_DIR") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        return dir;
    }
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("adam-loop-{label}-{nonce}"));
    std::fs::create_dir_all(&dir).expect("mkdir");
    dir
}

/// A workspace in its starting condition. Called again before the second turn
/// so the two turns face the same world, not a world the first turn improved.
fn fresh_workspace(root: &std::path::Path) -> Workspace {
    let _ = std::fs::remove_dir_all(root);
    let ws = Workspace::create(root).expect("create workspace");
    ws.seed(RECORDS).expect("seed");
    ws
}

fn main() {
    let base = scratch("run");
    let events_file = base.join("events.db");
    let events_path = events_file.to_str().expect("utf8").to_string();
    let log = Arc::new(SqliteEventLog::open(&events_path).expect("open log"));

    let mut report = serde_json::Map::new();
    report.insert("events_db".into(), events_path.clone().into());

    // -- the control: same work, never offered a mutation ------------------

    let control_root = base.join("control");
    let control_log = Arc::new(SqliteEventLog::open(":memory:").expect("control log"));
    let control = Organism::new("control", "never adapts", ":memory:")
        .expect("control organism")
        .with_events(control_log);

    let ws = fresh_workspace(&control_root);
    let c1 = act_and_record(&control, &ws, "control-turn-1").expect("control turn 1");
    let ws = fresh_workspace(&control_root);
    let c2 = act_and_record(&control, &ws, "control-turn-2").expect("control turn 2");

    report.insert("control_turn1_objective_bp".into(), c1.objective_bp.into());
    report.insert("control_turn2_objective_bp".into(), c2.objective_bp.into());

    // -- the subject -------------------------------------------------------

    let subject_root = base.join("subject");
    let genome_file = base.join("genome.json");
    let mut organism = Organism::open(
        "subject",
        "adapts if the evidence supports it",
        ":memory:",
        genome_file.to_str().expect("utf8"),
    )
    .expect("subject organism")
    .with_events(log.clone())
    .with_eve(EveClient::new(Box::new(Cp1Subprocess::command(
        "node",
        vec![eve_script()],
    ))));

    // Turn 1: act naively, and find out what the world does about it.
    let ws = fresh_workspace(&subject_root);
    let action_before = action_for(organism.genome());
    let turn1 = act_and_record(&organism, &ws, "turn-1").expect("turn 1");
    report.insert("turn1_action".into(), action_before.as_str().into());
    report.insert("turn1_objective_bp".into(), turn1.objective_bp.into());
    report.insert("observation_event".into(), turn1.event_id.clone().into());

    // The proposal that experience motivates.
    let proposal = EvolutionProposal::new(
        ProposalKind::AmendGenome {
            field: "policies.append".to_string(),
            current_value: String::new(),
            suggested_value: POLICY.to_string(),
        },
        "processing stopped on a malformed record that a check would have caught",
        vec![turn1.memory_id.to_string(), turn1.event_id.clone()],
        0.8,
    );
    let proposal_id = organism.propose_mutation(proposal);
    organism.link_cause(&proposal_id.to_string(), &turn1.event_id);
    organism.announce_proposal(proposal_id, "turn-1");
    report.insert("proposal_id".into(), proposal_id.to_string().into());

    // EVE measures it, or refuses to. Both are reported as found.
    match organism.validate_mutation(proposal_id, "turn-1") {
        Ok(result) => {
            report.insert(
                "recommendation".into(),
                format!("{:?}", result.recommendation).into(),
            );
            report.insert("delta_bp".into(), i64::from(result.delta_bp.raw()).into());
            report.insert("baseline_runs".into(), result.baseline.runs.into());
            report.insert("candidate_runs".into(), result.candidate.runs.into());
            report.insert(
                "baseline_composite_bp".into(),
                i64::from(result.baseline.composite_bp.raw()).into(),
            );
            report.insert(
                "candidate_composite_bp".into(),
                i64::from(result.candidate.composite_bp.raw()).into(),
            );
            report.insert("fitness_reason".into(), result.reason.clone().into());
        }
        Err(err) => {
            report.insert("measurement_error".into(), err.to_string().into());
        }
    }

    // Governance decides. A refusal here is a result, not a failure of the
    // harness, and is recorded with the reason EVE gave.
    let accepted = match organism.accept_mutation(proposal_id, "turn-1") {
        Ok(effect) => {
            report.insert("governance_outcome".into(), format!("{effect:?}").into());
            true
        }
        Err(err) => {
            report.insert("governance_outcome".into(), err.to_string().into());
            // Record the refusal in the log too, so the audit trail shows the
            // decision rather than an absence.
            let _ = organism.reject_mutation(proposal_id, &err.to_string(), "turn-1");
            false
        }
    };
    report.insert("accepted".into(), accepted.into());
    report.insert(
        "policies_after".into(),
        serde_json::Value::from(organism.genome().policies.clone()),
    );

    // Turn 2: the same world again, and whatever the organism now is.
    let ws = fresh_workspace(&subject_root);
    let action_after = action_for(organism.genome());
    let turn2 = act_and_record(&organism, &ws, "turn-2").expect("turn 2");
    report.insert("turn2_action".into(), action_after.as_str().into());
    report.insert("turn2_objective_bp".into(), turn2.objective_bp.into());
    report.insert(
        "behaviour_changed".into(),
        (action_before != action_after).into(),
    );
    report.insert(
        "objective_delta_bp".into(),
        (turn2.objective_bp - turn1.objective_bp).into(),
    );
    report.insert(
        "control_objective_delta_bp".into(),
        (c2.objective_bp - c1.objective_bp).into(),
    );

    // -- what the log can say afterwards -----------------------------------

    drop(organism);
    let log = SqliteEventLog::open(&events_path).expect("reopen log");
    let turn1_events: Vec<serde_json::Value> = log
        .turn("turn-1")
        .expect("turn")
        .iter()
        .map(|e| {
            serde_json::json!({
                "seq": e.seq,
                "kind": e.kind,
                "id": e.id,
                "causation_id": e.causation_id,
                "occurred_at": e.occurred_at,
            })
        })
        .collect();
    report.insert("turn1_events".into(), turn1_events.into());

    // The chain behind whichever decision was actually taken.
    let decision = log
        .turn("turn-1")
        .expect("turn")
        .into_iter()
        .rev()
        .find(|e| e.kind == "MutationAccepted" || e.kind == "MutationRejected");
    match decision {
        Some(decision) => {
            let chain: Vec<serde_json::Value> = log
                .causal_chain(&decision.id)
                .expect("chain")
                .iter()
                .map(|e| {
                    serde_json::json!({
                        "kind": e.kind,
                        "id": e.id,
                        "causation_id": e.causation_id,
                        "payload": e.payload,
                    })
                })
                .collect();
            report.insert("decision_kind".into(), decision.kind.clone().into());
            report.insert("causal_chain".into(), chain.into());
        }
        None => {
            report.insert("causal_chain".into(), serde_json::Value::Null);
        }
    }
    report.insert("total_events".into(), log.len().expect("len").into());

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::Value::Object(report)).expect("report")
    );
}
