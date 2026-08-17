//! Adversarial cases against the decision, rather than the store.
//!
//! Each one hands the organism evidence it should refuse to act on, and checks
//! that it refuses for the stated reason and records the refusal. As in
//! `adversarial_store.rs`, tests named `nothing_stops_` or `no_` are findings:
//! they pin down where Phase 0 has no defence, so the limitation is written
//! down in the one place that cannot drift away from the code.
//!
//! Every measurement here is a stub. Stubs settle what the system does with a
//! verdict, never what the verdict should be.

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

struct Scratch(std::path::PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("adam-advl-{label}-{nonce}"));
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

fn measurement(composite: f64) -> Measurement {
    Measurement::experience(
        BasisPoints::from_ratio(composite),
        BasisPoints::from_ratio(composite),
        BasisPoints::from_ratio(0.3),
        BasisPoints::from_ratio(0.6),
        BasisPoints::from_ratio(0.4),
        9,
    )
}

fn verdict(
    proposal_id: ProposalId,
    recommendation: Recommendation,
    delta_bp: i32,
    scenarios: &[&str],
) -> FitnessResult {
    FitnessResult {
        cp: "cp1".to_string(),
        doc_type: "FitnessResult".to_string(),
        id: "11111111-1111-4111-8111-111111111111".to_string(),
        mutation_id: proposal_id.to_string(),
        seed: 1337,
        scenario_ids: scenarios.iter().map(|s| (*s).to_string()).collect(),
        trials: 3,
        baseline: measurement(0.70),
        candidate: measurement(0.70 + f64::from(delta_bp) / 10_000.0),
        delta_bp: SignedBasisPoints::new(delta_bp),
        recommendation,
        reason: "stubbed measurement — not a measurement of anything".to_string(),
        provenance: Provenance::now(Component::Eve, "eve:cp1/validate"),
    }
}

fn proposal(evidence: Vec<String>) -> EvolutionProposal {
    EvolutionProposal::new(
        ProposalKind::AmendGenome {
            field: "policies.append".to_string(),
            current_value: String::new(),
            suggested_value: POLICY.to_string(),
        },
        "processing stopped on a malformed record that a check would have caught",
        evidence,
        0.8,
    )
}

/// One turn up to the point of decision, with whatever EVE the case needs.
struct Turn {
    organism: Organism,
    proposal_id: ProposalId,
    objective_bp: i32,
    observation_event: String,
}

fn up_to_the_decision(scratch: &Scratch, eve: impl FnOnce(ProposalId) -> EveClient) -> Turn {
    let log = Arc::new(SqliteEventLog::open(&scratch.path("events.db")).expect("open"));
    let mut organism = Organism::open(
        "subject",
        "adversarial subject",
        ":memory:",
        &scratch.path("genome.json"),
    )
    .expect("organism")
    .with_events(log);

    let ws = fresh_workspace(&scratch.join("workspace"));
    let turn1 = act_and_record(&organism, &ws, "turn-1").expect("turn 1");
    let proposal_id = organism.propose_mutation(proposal(vec![turn1.event_id.clone()]));
    let organism = organism.with_eve(eve(proposal_id));
    organism.link_cause(&proposal_id.to_string(), &turn1.event_id);
    organism.announce_proposal(proposal_id, "turn-1");

    Turn {
        organism,
        proposal_id,
        objective_bp: turn1.objective_bp,
        observation_event: turn1.event_id,
    }
}

fn approving(proposal_id: ProposalId) -> EveClient {
    EveClient::new(Box::new(StubProvider::returning(verdict(
        proposal_id,
        Recommendation::Approve,
        700,
        &["excellent"],
    ))))
}

// -- verdicts that must not become adaptations ------------------------------

#[test]
fn an_unmeasurable_candidate_is_never_given_a_score() {
    // The failure this guards against is not a crash. It is a system that,
    // finding no measurement, supplies a plausible one and proceeds.
    let scratch = Scratch::new("unmeasurable");
    let mut turn = up_to_the_decision(&scratch, |_| {
        EveClient::new(Box::new(StubProvider::failing(
            "no scenario could exercise this mutation",
        )))
    });

    let measured = turn.organism.validate_mutation(turn.proposal_id, "turn-1");
    let detail = measured.expect_err("an unmeasurable candidate must not yield a result");
    assert!(
        detail.to_string().contains("no scenario could exercise"),
        "the reason must survive to the caller: {detail}"
    );

    let refusal = turn
        .organism
        .accept_mutation(turn.proposal_id, "turn-1")
        .expect_err("without a measurement there is nothing to approve");
    turn.organism
        .reject_mutation(turn.proposal_id, &refusal.to_string(), "turn-1")
        .expect("record the refusal");

    assert!(turn.organism.genome().policies.is_empty());
    assert_eq!(action_for(turn.organism.genome()), Action::ProcessAll);
}

#[test]
fn a_worse_candidate_is_refused() {
    let scratch = Scratch::new("worse");
    let mut turn = up_to_the_decision(&scratch, |id| {
        EveClient::new(Box::new(StubProvider::returning(verdict(
            id,
            Recommendation::Reject,
            -900,
            &["excellent"],
        ))))
    });

    let result = turn
        .organism
        .validate_mutation(turn.proposal_id, "turn-1")
        .expect("a verdict, albeit a negative one");
    assert_eq!(result.delta_bp.raw(), -900);
    assert!(turn
        .organism
        .accept_mutation(turn.proposal_id, "turn-1")
        .is_err());
    assert!(turn.organism.genome().policies.is_empty());
}

#[test]
fn a_candidate_that_changes_nothing_is_refused() {
    // Zero delta is the case most likely to be waved through, because nothing
    // appears to be at stake. But an unmeasurable improvement adopted anyway is
    // exactly the drift the gate exists to prevent.
    let scratch = Scratch::new("identical");
    let mut turn = up_to_the_decision(&scratch, |id| {
        EveClient::new(Box::new(StubProvider::returning(verdict(
            id,
            Recommendation::NeedsReview,
            0,
            &["excellent"],
        ))))
    });

    let result = turn
        .organism
        .validate_mutation(turn.proposal_id, "turn-1")
        .expect("a verdict");
    assert_eq!(result.delta_bp.raw(), 0);
    assert!(turn
        .organism
        .accept_mutation(turn.proposal_id, "turn-1")
        .is_err());
    assert!(turn.organism.genome().policies.is_empty());
}

// -- evidence that should not have counted ----------------------------------

#[test]
fn nothing_stops_a_mutation_justified_by_a_stale_observation() {
    // FINDING. The proposal names an observation from a world that no longer
    // exists — the record it complained about has since been removed — and
    // nothing checks that. Approval still applies the change.
    //
    // Closing this needs the observation to carry a version or state hash of
    // the environment, and the gate to compare it against the environment at
    // decision time. Phase 0 has neither. Consequence: an organism could adapt
    // to a problem that has already gone away.
    let scratch = Scratch::new("stale");
    let root = scratch.join("workspace");
    let log = Arc::new(SqliteEventLog::open(&scratch.path("events.db")).expect("open"));
    let mut organism = Organism::open(
        "subject",
        "stale evidence",
        ":memory:",
        &scratch.path("genome.json"),
    )
    .expect("organism")
    .with_events(log);

    let ws = fresh_workspace(&root);
    let stale = act_and_record(&organism, &ws, "turn-1").expect("turn 1");

    // The world moves on: the offending record is gone.
    std::fs::remove_file(root.join("inbox").join("03-charlie.rec")).expect("remove");

    let proposal_id = organism.propose_mutation(proposal(vec![stale.event_id.clone()]));
    let mut organism = organism.with_eve(approving(proposal_id));
    organism.announce_proposal(proposal_id, "turn-1");
    organism
        .validate_mutation(proposal_id, "turn-1")
        .expect("stub verdict");
    organism
        .accept_mutation(proposal_id, "turn-1")
        .expect("accepted — no staleness check exists to stop it");

    assert!(
        organism.genome().policies.iter().any(|p| p == POLICY),
        "this is the gap: evidence about a vanished problem still changed the organism"
    );
}

#[test]
fn conflicting_observations_are_both_kept_and_neither_is_resolved() {
    // FINDING. Two observations of the same turn contradict each other. The
    // store keeps both, which is right — silently discarding one would destroy
    // the evidence that a conflict happened. What is missing is anything that
    // *notices*. Nothing downstream weighs, dates or reconciles them, so which
    // one motivates a proposal is entirely up to the caller.
    let scratch = Scratch::new("conflict");
    let events_path = scratch.path("events.db");
    let log = Arc::new(SqliteEventLog::open(&events_path).expect("open"));
    let organism = Organism::open(
        "subject",
        "conflicting evidence",
        ":memory:",
        &scratch.path("genome.json"),
    )
    .expect("organism")
    .with_events(log);

    for content in [
        "workspace: acted process_all; processed 2 of 4; stopped on 03-charlie.rec",
        "workspace: acted process_all; processed 4 of 4; completed without stopping",
    ] {
        organism
            .consolidate_memory(
                adam_memory::MemoryKind::Episodic,
                content,
                "pcr:workspace",
                vec![],
                1.0,
                0.0,
                "turn-1",
            )
            .expect("memory");
    }
    drop(organism);

    let log = SqliteEventLog::open(&events_path).expect("reopen");
    let turn = log.turn("turn-1").expect("turn");
    assert_eq!(turn.len(), 2, "both accounts must survive");
    assert!(turn[0].payload.contains("stopped on 03-charlie.rec"));
    assert!(turn[1].payload.contains("completed without stopping"));
}

// -- failures on the way in and out -----------------------------------------

#[test]
fn a_failed_environment_action_records_no_observation() {
    // A connector that invented an observation from a failed action would be
    // the fabricated benchmark the whole experiment is meant to avoid.
    let scratch = Scratch::new("envfail");
    let events_path = scratch.path("events.db");
    let root = scratch.join("workspace");
    let log = Arc::new(SqliteEventLog::open(&events_path).expect("open"));
    let organism = Organism::open(
        "subject",
        "broken environment",
        ":memory:",
        &scratch.path("genome.json"),
    )
    .expect("organism")
    .with_events(log);

    let ws = fresh_workspace(&root);
    std::fs::remove_dir_all(root.join("inbox")).expect("break the environment");

    let outcome = act_and_record(&organism, &ws, "turn-1");
    assert!(outcome.is_err(), "acting on a broken world must fail");
    drop(organism);

    let log = SqliteEventLog::open(&events_path).expect("reopen");
    assert!(
        log.turn("turn-1").expect("turn").is_empty(),
        "a failed action must leave no observation behind"
    );
}

#[test]
fn a_failed_evaluation_leaves_the_organism_where_it_was() {
    let scratch = Scratch::new("evefail");
    let mut turn = up_to_the_decision(&scratch, |_| {
        EveClient::new(Box::new(StubProvider::failing(
            "eve exited before answering",
        )))
    });

    assert!(turn
        .organism
        .validate_mutation(turn.proposal_id, "turn-1")
        .is_err());
    assert!(turn
        .organism
        .accept_mutation(turn.proposal_id, "turn-1")
        .is_err());

    let ws = fresh_workspace(&scratch.join("workspace"));
    let after = act_and_record(&turn.organism, &ws, "turn-2").expect("turn 2");
    assert_eq!(
        after.objective_bp, turn.objective_bp,
        "a failed evaluation must not change behaviour in either direction"
    );
    assert!(!turn.observation_event.is_empty());
}

// -- what the verdict cannot say --------------------------------------------

#[test]
fn no_per_scenario_disagreement_survives_into_the_decision() {
    // FINDING, and a structural one. A candidate can be better in one scenario
    // and worse in another, but a FitnessResult carries scenario *ids* and a
    // single aggregate delta. The disagreement is averaged away before
    // governance ever sees it, so "better overall" cannot be distinguished from
    // "better everywhere".
    //
    // Closing this needs a per-scenario breakdown in the CP/1 FitnessResult and
    // a gate that can refuse on variance. Both are protocol changes, and
    // neither is in Phase 0.
    let scratch = Scratch::new("scenarios");
    let mut turn = up_to_the_decision(&scratch, |id| {
        EveClient::new(Box::new(StubProvider::returning(verdict(
            id,
            Recommendation::Approve,
            50,
            &["excellent", "bad"],
        ))))
    });

    let result = turn
        .organism
        .validate_mutation(turn.proposal_id, "turn-1")
        .expect("a verdict");

    assert_eq!(result.scenario_ids.len(), 2);
    let serialized = serde_json::to_value(&result).expect("json");
    let fields: Vec<&str> = serialized
        .as_object()
        .expect("object")
        .keys()
        .map(String::as_str)
        .collect();
    assert!(
        !fields.iter().any(|f| f.contains("per_scenario")),
        "if this ever fails, the protocol grew the field and this finding is stale: {fields:?}"
    );

    turn.organism
        .accept_mutation(turn.proposal_id, "turn-1")
        .expect("approved on an aggregate that hides the disagreement");
}
