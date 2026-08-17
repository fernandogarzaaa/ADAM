//! The organism's events reaching durable storage, with their causal edges.
//!
//! A unit test can show that `emit` builds an event. It cannot show that a
//! turn's events survive the process and can be walked backwards afterwards,
//! which is the property the whole event log exists for — so this test drives
//! a real `Organism` into a real SQLite file and then re-opens it.

use std::sync::Arc;

use adam_eventlog::SqliteEventLog;
use adam_evolution::{EvolutionProposal, ProposalKind};
use adam_memory::MemoryKind;
use adam_organism::Organism;

const CORRELATION: &str = "turn-observation-1";

struct TempDb {
    dir: std::path::PathBuf,
}

impl TempDb {
    fn new(label: &str) -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("adam-{label}-{nonce}"));
        std::fs::create_dir_all(&dir).expect("mkdir");
        Self { dir }
    }

    fn path(&self, name: &str) -> String {
        self.dir
            .join(name)
            .to_str()
            .expect("temp path is utf8")
            .to_string()
    }
}

impl Drop for TempDb {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Drive one developmental turn and return the id of the event that started it.
///
/// The mutation targets `preferences.*` deliberately: that is the one genome
/// field an organism with no EVE attached is permitted to amend, so this test
/// measures durability and causality without also depending on a subprocess
/// measurement. What it therefore does NOT show is a governed adaptation —
/// that needs EVE, and is a separate test.
fn run_turn(events_path: &str) -> String {
    let log = Arc::new(SqliteEventLog::open(events_path).expect("open log"));
    let mut organism = Organism::new("probe", "durable event probe", ":memory:")
        .expect("organism")
        .with_events(log);

    // Something happened outside the organism, and it noticed.
    let memory_id = organism
        .consolidate_memory(
            MemoryKind::Episodic,
            "cargo test failed: assertion in verify step",
            "test:env",
            vec![],
            0.9,
            0.0,
            CORRELATION,
        )
        .expect("consolidate");
    let observation_event = organism
        .last_event(&memory_id.to_string())
        .expect("the memory it just consolidated was announced");

    let proposal = EvolutionProposal::new(
        ProposalKind::AmendGenome {
            field: "preferences.verify_before_act".to_string(),
            current_value: "false".to_string(),
            suggested_value: "true".to_string(),
        },
        "the observed failure came from acting without verifying",
        vec![memory_id.to_string()],
        0.8,
    );
    let proposal_id = organism.propose_mutation(proposal);

    // The caller — not the organism — knows which observation motivated this,
    // so it says so explicitly rather than letting ordering imply it.
    organism.link_cause(&proposal_id.to_string(), &observation_event);
    organism.announce_proposal(proposal_id, CORRELATION);

    organism
        .accept_mutation(proposal_id, CORRELATION)
        .expect("accept");

    observation_event
}

#[test]
fn a_turns_events_survive_the_organism() {
    let temp = TempDb::new("durable-events");
    let path = temp.path("events.db");

    run_turn(&path);

    // Re-opened, not merely re-read: the organism that wrote these is gone.
    let log = SqliteEventLog::open(&path).expect("reopen");
    let turn = log.turn(CORRELATION).expect("turn");
    let kinds: Vec<&str> = turn.iter().map(|e| e.kind.as_str()).collect();

    assert_eq!(
        kinds,
        vec![
            "MemoryConsolidated",
            "MutationProposed",
            "GenomeCommitted",
            "MutationAccepted",
        ],
        "the whole turn must be on disk, in the order it happened"
    );
}

#[test]
fn an_accepted_mutation_can_be_traced_back_to_what_caused_it() {
    let temp = TempDb::new("causal-chain");
    let path = temp.path("events.db");
    let observation_event = run_turn(&path);

    let log = SqliteEventLog::open(&path).expect("reopen");
    let accepted = log
        .turn(CORRELATION)
        .expect("turn")
        .into_iter()
        .find(|e| e.kind == "MutationAccepted")
        .expect("the turn accepted a mutation");

    let chain = log.causal_chain(&accepted.id).expect("chain");
    let kinds: Vec<&str> = chain.iter().map(|e| e.kind.as_str()).collect();
    assert_eq!(
        kinds,
        vec![
            "MemoryConsolidated",
            "MutationProposed",
            "GenomeCommitted",
            "MutationAccepted",
        ],
        "walking causation_id backwards must reach the observation"
    );

    // The root of the chain is the event carrying the external observation —
    // this is the edge that makes "why did the organism change?" answerable.
    assert_eq!(chain[0].id, observation_event);
    assert!(
        chain[0].causation_id.is_none(),
        "the observation has no cause inside the organism"
    );
    assert!(
        chain[0].document.contains("cargo test failed"),
        "the root must still carry what was actually observed"
    );

    // Every non-root link names its predecessor. Asserted rather than assumed,
    // because a NULL causation_id is exactly the failure this work fixes.
    for pair in chain.windows(2) {
        assert_eq!(
            pair[1].causation_id.as_deref(),
            Some(pair[0].id.as_str()),
            "{} must name {} as its cause",
            pair[1].kind,
            pair[0].kind
        );
    }
}

#[test]
fn an_unrelated_turn_does_not_join_the_chain() {
    let temp = TempDb::new("chain-isolation");
    let path = temp.path("events.db");
    run_turn(&path);

    // A second organism writing to the same log starts its own chains: the
    // causal record must not merge two histories just because they share a
    // file.
    {
        let log = Arc::new(SqliteEventLog::open(&path).expect("open"));
        let organism = Organism::new("other", "unrelated", ":memory:")
            .expect("organism")
            .with_events(log);
        organism
            .consolidate_memory(
                MemoryKind::Semantic,
                "unrelated note",
                "test:env",
                vec![],
                0.5,
                0.0,
                "turn-unrelated",
            )
            .expect("consolidate");
    }

    let log = SqliteEventLog::open(&path).expect("reopen");
    let stray = log.turn("turn-unrelated").expect("turn");
    assert_eq!(stray.len(), 1);
    assert!(
        stray[0].causation_id.is_none(),
        "a new organism's first event has no cause"
    );
    assert_eq!(
        log.turn(CORRELATION).expect("turn").len(),
        4,
        "the first turn is unchanged by the second"
    );
}
