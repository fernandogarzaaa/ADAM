//! Does the loop close *if* the evidence ever supports it?
//!
//! # What this file is not
//!
//! It is not evidence of learning. The verdict EVE returns here is a stub, and
//! a stub verdict proves nothing about the world. The real run
//! (`examples/closed_loop.rs`, against the real EVE) returned `NeedsReview` and
//! the organism did not change — that is the experiment's result, and it is
//! reported as such.
//!
//! What this file separates out is a different question, which the real run
//! cannot answer because it never got past governance: supposing a measurement
//! did recommend approval, would acceptance actually reach the environment and
//! change what the organism does next? If the answer were no, the loop would be
//! broken quite apart from what EVE thinks, and every future run would be
//! wasted. So the stub is here to test the wiring, and it is named to make sure
//! nobody mistakes it for a result.

use std::sync::Arc;

use adam_eve::{EveClient, FitnessResult, Measurement, Recommendation, StubProvider};
use adam_eventlog::SqliteEventLog;
use adam_evolution::{EvolutionProposal, ProposalId, ProposalKind};
use adam_organism::Organism;
use adam_pcr::{act_and_record, action_for, Action, Workspace};
use adam_protocol::{BasisPoints, Component, Provenance, SignedBasisPoints};

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

/// A temporary directory that removes itself, so a failing test leaves nothing
/// behind for the next one to find.
struct Scratch(std::path::PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("adam-mech-{label}-{nonce}"));
        std::fs::create_dir_all(&dir).expect("mkdir");
        Self(dir)
    }

    fn join(&self, name: &str) -> std::path::PathBuf {
        self.0.join(name)
    }

    fn path(&self, name: &str) -> String {
        self.join(name).to_str().expect("utf8").to_string()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fresh_workspace(root: &std::path::Path) -> Workspace {
    let _ = std::fs::remove_dir_all(root);
    let ws = Workspace::create(root).expect("create workspace");
    ws.seed(RECORDS).expect("seed");
    ws
}

/// A measurement that says what the test needs it to say.
///
/// Deliberately blunt. Anything subtler would invite the reading that the
/// numbers mean something.
fn stub_eve(proposal_id: ProposalId, recommendation: Recommendation) -> EveClient {
    let measurement = Measurement::experience(
        BasisPoints::from_ratio(0.7),
        BasisPoints::from_ratio(0.7),
        BasisPoints::from_ratio(0.3),
        BasisPoints::from_ratio(0.6),
        BasisPoints::from_ratio(0.4),
        9,
    );
    EveClient::new(Box::new(StubProvider::returning(FitnessResult {
        cp: "cp1".to_string(),
        doc_type: "FitnessResult".to_string(),
        id: "11111111-1111-4111-8111-111111111111".to_string(),
        mutation_id: proposal_id.to_string(),
        seed: 1337,
        scenario_ids: vec!["excellent".to_string()],
        trials: 3,
        baseline: measurement.clone(),
        candidate: measurement,
        delta_bp: SignedBasisPoints::new(700),
        recommendation,
        reason: "stubbed measurement — not a measurement of anything".to_string(),
        provenance: Provenance::now(Component::Eve, "eve:cp1/validate"),
    })))
}

/// The proposal turn 1's experience motivates, in both tests.
fn verify_proposal(memory_id: String, event_id: String) -> EvolutionProposal {
    EvolutionProposal::new(
        ProposalKind::AmendGenome {
            field: "policies.append".to_string(),
            current_value: String::new(),
            suggested_value: POLICY.to_string(),
        },
        "processing stopped on a malformed record that a check would have caught",
        vec![memory_id, event_id],
        0.8,
    )
}

#[test]
fn an_approved_mutation_reaches_the_environment() {
    let scratch = Scratch::new("approve");
    let events_path = scratch.path("events.db");
    let log = Arc::new(SqliteEventLog::open(&events_path).expect("open log"));
    let root = scratch.join("workspace");

    let mut organism = Organism::open(
        "subject",
        "mechanism test",
        ":memory:",
        &scratch.path("genome.json"),
    )
    .expect("organism")
    .with_events(log);

    // Turn 1: naive, and it goes badly, which is the point.
    let ws = fresh_workspace(&root);
    let turn1 = act_and_record(&organism, &ws, "turn-1").expect("turn 1");
    assert_eq!(action_for(organism.genome()), Action::ProcessAll);
    assert_eq!(
        turn1.objective_bp, 5000,
        "processing blindly should stall halfway through the records"
    );

    let proposal_id =
        organism.propose_mutation(verify_proposal(turn1.memory_id.to_string(), turn1.event_id));
    let mut organism = organism.with_eve(stub_eve(proposal_id, Recommendation::Approve));
    organism.announce_proposal(proposal_id, "turn-1");
    organism
        .validate_mutation(proposal_id, "turn-1")
        .expect("stub measurement");
    organism
        .accept_mutation(proposal_id, "turn-1")
        .expect("an approved mutation must be acceptable");

    assert!(
        organism.genome().policies.iter().any(|p| p == POLICY),
        "acceptance must reach the genome, not just the event log"
    );

    // Turn 2: the same world again, so any difference is the organism's.
    let ws = fresh_workspace(&root);
    assert_eq!(
        action_for(organism.genome()),
        Action::VerifyThenProcess,
        "the accepted policy must change what the organism chooses to do"
    );
    let turn2 = act_and_record(&organism, &ws, "turn-2").expect("turn 2");
    assert!(
        turn2.objective_bp > turn1.objective_bp,
        "a changed action must change the environment's own measure: {} then {}",
        turn1.objective_bp,
        turn2.objective_bp
    );
    assert_eq!(turn2.objective_bp, 7500);
}

#[test]
fn an_organism_offered_nothing_does_not_drift() {
    // The control for the test above. Without it, "turn 2 scored higher" has a
    // rival explanation — that a second turn scores higher whatever happens —
    // and the assertion above would be worth nothing.
    let scratch = Scratch::new("control");
    let log = Arc::new(SqliteEventLog::open(":memory:").expect("log"));
    let root = scratch.join("workspace");
    let organism = Organism::new("control", "never adapts", ":memory:")
        .expect("organism")
        .with_events(log);

    let ws = fresh_workspace(&root);
    let first = act_and_record(&organism, &ws, "control-1").expect("turn 1");
    let ws = fresh_workspace(&root);
    let second = act_and_record(&organism, &ws, "control-2").expect("turn 2");

    assert_eq!(
        first.objective_bp, second.objective_bp,
        "repetition alone must not improve anything"
    );
    assert!(organism.genome().policies.is_empty());
}

#[test]
fn a_refused_mutation_leaves_behaviour_alone() {
    // What the real run did. Asserted here so a later change that quietly
    // relaxed the gate would fail a test rather than look like progress.
    let scratch = Scratch::new("refuse");
    let events_path = scratch.path("events.db");
    let log = Arc::new(SqliteEventLog::open(&events_path).expect("open log"));
    let root = scratch.join("workspace");

    let mut organism = Organism::open(
        "subject",
        "mechanism test",
        ":memory:",
        &scratch.path("genome.json"),
    )
    .expect("organism")
    .with_events(log);

    let ws = fresh_workspace(&root);
    let turn1 = act_and_record(&organism, &ws, "turn-1").expect("turn 1");
    let proposal_id = organism.propose_mutation(verify_proposal(
        turn1.memory_id.to_string(),
        turn1.event_id.clone(),
    ));
    let mut organism = organism.with_eve(stub_eve(proposal_id, Recommendation::NeedsReview));
    organism.link_cause(&proposal_id.to_string(), &turn1.event_id);
    organism.announce_proposal(proposal_id, "turn-1");
    organism
        .validate_mutation(proposal_id, "turn-1")
        .expect("stub measurement");

    let refusal = organism
        .accept_mutation(proposal_id, "turn-1")
        .expect_err("an unapproved mutation must not be accepted");
    organism
        .reject_mutation(proposal_id, &refusal.to_string(), "turn-1")
        .expect("the refusal must be recordable");

    assert!(organism.genome().policies.is_empty());
    let ws = fresh_workspace(&root);
    let turn2 = act_and_record(&organism, &ws, "turn-2").expect("turn 2");
    assert_eq!(
        turn1.objective_bp, turn2.objective_bp,
        "a refused proposal must leave the organism exactly as it was"
    );

    // And the refusal is in the log, with the reason, reachable from the
    // observation that started it. An absence would be indistinguishable from
    // a decision never taken.
    drop(organism);
    let log = SqliteEventLog::open(&events_path).expect("reopen");
    let decision = log
        .turn("turn-1")
        .expect("turn")
        .into_iter()
        .find(|e| e.kind == "MutationRejected")
        .expect("the rejection must be durable");
    let chain = log.causal_chain(&decision.id).expect("chain");
    assert_eq!(chain.first().expect("root").id, turn1.event_id);
    assert!(decision.payload.contains("policies.append"));
}
