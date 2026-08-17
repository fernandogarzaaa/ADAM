//! Adversarial cases against the event store and the loop's durability.
//!
//! Half of these assert a defence. The other half assert an *absence* of one,
//! because the honest thing to do with a gap is to pin it down where a reader
//! will find it, rather than leave the impression that it was handled. Tests
//! whose names begin `nothing_stops_` are findings, not guarantees, and each
//! says in its body what would have to change.
//!
//! Cases covered here: duplicated observation, replayed event, missing
//! causation_id, corrupted provenance, interrupted loop, restart after
//! persisted state, and event-store corruption. The remaining cases are in
//! `adversarial_loop.rs`, which needs a measurement to argue with.

use std::collections::BTreeMap;
use std::sync::Arc;

use adam_eventlog::{EventLogError, SqliteEventLog};
use adam_evolution::{EvolutionProposal, ProposalKind};
use adam_memory::MemoryKind;
use adam_organism::Organism;
use adam_pcr::{act_and_record, Workspace};
use adam_protocol::{Component, Event, EventKind, PayloadValue, SubjectType};

const RECORDS: &[(&str, &str)] = &[
    ("01-alpha.rec", "name=alpha\ncount=3\n"),
    ("02-bravo.rec", "name=bravo\ncount=1\n"),
    ("03-charlie.rec", "name=charlie\nthis line has no separator\n"),
    ("04-delta.rec", "name=delta\ncount=7\n"),
];

struct Scratch(std::path::PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("adam-adv-{label}-{nonce}"));
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

fn observation(correlation_id: &str, content: &str) -> Event {
    let mut payload = BTreeMap::new();
    payload.insert("content".to_string(), PayloadValue::from(content));
    Event::new(
        Component::Adam,
        EventKind::MemoryConsolidated,
        "22222222-2222-4222-8222-222222222222",
        SubjectType::Memory,
        correlation_id,
        payload,
        "test:adversarial",
    )
}

// -- replay and duplication -------------------------------------------------

#[test]
fn the_same_event_cannot_be_appended_twice() {
    // A replayed event is the cheapest attack there is: re-send yesterday's
    // announcement and the organism counts the same evidence again.
    let scratch = Scratch::new("replay");
    let log = SqliteEventLog::open(&scratch.path("events.db")).expect("open");
    let event = observation("turn-1", "the environment stalled");

    log.append(&event).expect("first append");
    let second = log.append(&event);

    assert!(
        matches!(second, Err(EventLogError::Duplicate { .. })),
        "a replayed event must be refused by id, got {second:?}"
    );
    assert_eq!(log.len().expect("len"), 1);
}

#[test]
fn nothing_stops_the_same_experience_being_recorded_twice_under_new_ids() {
    // FINDING, not a guarantee. Identity is the event id, and two honest
    // observations of the same fact get two ids. So an environment polled
    // twice contributes its evidence twice, and nothing in the store objects.
    //
    // Closing this needs a dedup key over (correlation, origin, content) or a
    // caller that knows it has already looked. Phase 0 has neither, and the
    // consequence is that repeated polling could inflate the apparent weight
    // of a single fact.
    let scratch = Scratch::new("dup");
    let log = SqliteEventLog::open(&scratch.path("events.db")).expect("open");

    log.append(&observation("turn-1", "the environment stalled"))
        .expect("first");
    log.append(&observation("turn-1", "the environment stalled"))
        .expect("second");

    let turn = log.turn("turn-1").expect("turn");
    assert_eq!(turn.len(), 2);
    assert_ne!(turn[0].id, turn[1].id);
    assert_eq!(
        turn[0].payload, turn[1].payload,
        "identical content, two accepted records — this is the gap"
    );
}

// -- causation --------------------------------------------------------------

#[test]
fn an_event_with_no_cause_traces_only_to_itself() {
    // The chain must stop, not guess. An inferred edge would be a fabricated
    // answer to "why did the organism change?", which is worse than no answer.
    let scratch = Scratch::new("nocause");
    let log = SqliteEventLog::open(&scratch.path("events.db")).expect("open");

    let first = observation("turn-1", "something happened");
    let unlinked = observation("turn-1", "something else happened, later");
    log.append(&first).expect("first");
    log.append(&unlinked).expect("second");

    let chain = log.causal_chain(&unlinked.id).expect("chain");
    assert_eq!(chain.len(), 1);
    assert_eq!(chain[0].id, unlinked.id);
    assert!(
        chain[0].causation_id.is_none(),
        "ordering must never be read as causation"
    );
}

#[test]
fn a_chain_naming_an_absent_cause_stops_rather_than_failing() {
    // A partial chain is recoverable evidence; an error here would lose the
    // part that did survive.
    let scratch = Scratch::new("dangling");
    let log = SqliteEventLog::open(&scratch.path("events.db")).expect("open");

    let orphan = observation("turn-1", "caused by an event that was never stored")
        .caused_by("99999999-9999-4999-8999-999999999999");
    log.append(&orphan).expect("append");

    let chain = log.causal_chain(&orphan.id).expect("chain");
    assert_eq!(chain.len(), 1);
    assert_eq!(
        chain[0].causation_id.as_deref(),
        Some("99999999-9999-4999-8999-999999999999"),
        "the dangling reference must stay visible, so the loss is auditable"
    );
}

// -- tampering --------------------------------------------------------------

#[test]
fn stored_events_refuse_to_be_edited_or_deleted() {
    let scratch = Scratch::new("tamper");
    let path = scratch.path("events.db");
    let log = SqliteEventLog::open(&path).expect("open");
    let event = observation("turn-1", "inconvenient truth");
    log.append(&event).expect("append");
    drop(log);

    // Through a second connection, as an attacker with file access would.
    let conn = rusqlite::Connection::open(&path).expect("connect");
    let update = conn.execute("UPDATE events SET payload = '{}' WHERE id = ?1", [&event.id]);
    let delete = conn.execute("DELETE FROM events WHERE id = ?1", [&event.id]);

    assert!(update.is_err(), "UPDATE must be refused");
    assert!(delete.is_err(), "DELETE must be refused");

    let log = SqliteEventLog::open(&path).expect("reopen");
    assert_eq!(log.len().expect("len"), 1);
}

#[test]
fn a_stored_document_still_hashes_to_what_it_claims() {
    // The stored content_hash is recomputed here from the stored document, so
    // a document swapped out from under the row would be detectable.
    //
    // Its limit, stated plainly: the hash is computed by the store itself and
    // stored beside what it covers, so an attacker who can write the file can
    // write both. There is no chained or signed hash in Phase 0. This check
    // catches accidental corruption, not a determined edit.
    let scratch = Scratch::new("hash");
    let log = SqliteEventLog::open(&scratch.path("events.db")).expect("open");
    let event = observation("turn-1", "a fact worth keeping");
    log.append(&event).expect("append");

    let stored = log.by_id(&event.id).expect("query").expect("present");
    let document: serde_json::Value = serde_json::from_str(&stored.document).expect("json");
    let parsed: Event = serde_json::from_value(document).expect("an Event");
    let resealed = parsed.seal().expect("seal");

    assert_eq!(
        resealed["provenance"]["content_hash"].as_str().expect("hash"),
        stored.content_hash,
        "the row's hash must match the document it sits beside"
    );
}

#[test]
fn a_corrupt_store_reports_an_error_instead_of_reporting_nothing() {
    // The dangerous failure is not a crash, it is an empty result: a loop that
    // reads zero events from a broken file would conclude it had no history
    // and carry on as if newly born.
    let scratch = Scratch::new("corrupt");
    let path = scratch.path("events.db");
    let log = SqliteEventLog::open(&path).expect("open");
    log.append(&observation("turn-1", "before the damage"))
        .expect("append");
    drop(log);

    std::fs::write(&path, b"this is not a database").expect("corrupt the file");

    let outcome = SqliteEventLog::open(&path).and_then(|log| log.all());
    assert!(
        outcome.is_err(),
        "a corrupt store must fail loudly, got {outcome:?}"
    );
}

// -- interruption and restart -----------------------------------------------

#[test]
fn an_interrupted_turn_leaves_a_turn_with_no_decision_in_it() {
    // The organism is dropped between proposing and deciding, as a killed
    // process would be. What matters is that the log says so.
    let scratch = Scratch::new("interrupt");
    let events_path = scratch.path("events.db");
    let root = scratch.join("workspace");
    {
        let log = Arc::new(SqliteEventLog::open(&events_path).expect("open"));
        let mut organism = Organism::open(
            "subject",
            "interrupted",
            ":memory:",
            &scratch.path("genome.json"),
        )
        .expect("organism")
        .with_events(log);

        let ws = fresh_workspace(&root);
        let turn1 = act_and_record(&organism, &ws, "turn-1").expect("turn 1");
        let proposal = EvolutionProposal::new(
            ProposalKind::AmendGenome {
                field: "policies.append".to_string(),
                current_value: String::new(),
                suggested_value: "verify records before processing them".to_string(),
            },
            "the run stopped on a malformed record",
            vec![turn1.event_id.clone()],
            0.8,
        );
        let proposal_id = organism.propose_mutation(proposal);
        organism.link_cause(&proposal_id.to_string(), &turn1.event_id);
        organism.announce_proposal(proposal_id, "turn-1");
        // and then the process dies.
    }

    let log = SqliteEventLog::open(&events_path).expect("reopen");
    let kinds: Vec<String> = log
        .turn("turn-1")
        .expect("turn")
        .into_iter()
        .map(|e| e.kind)
        .collect();

    assert_eq!(kinds, vec!["MemoryConsolidated", "MutationProposed"]);
    assert!(
        !kinds.iter().any(|k| k.contains("Accepted")),
        "an interrupted turn must not look decided"
    );
}

#[test]
fn a_restarted_organism_keeps_what_it_had_and_appends_to_the_same_log() {
    let scratch = Scratch::new("restart");
    let events_path = scratch.path("events.db");
    let genome_path = scratch.path("genome.json");
    let root = scratch.join("workspace");

    let first_run_events = {
        let log = Arc::new(SqliteEventLog::open(&events_path).expect("open"));
        let organism = Organism::open("subject", "restarts", ":memory:", &genome_path)
            .expect("organism")
            .with_events(log);
        let ws = fresh_workspace(&root);
        act_and_record(&organism, &ws, "turn-1").expect("turn 1");
        organism
            .consolidate_memory(
                MemoryKind::Episodic,
                "a fact from before the restart",
                "test:adversarial",
                vec![],
                1.0,
                0.0,
                "turn-1",
            )
            .expect("memory");
        SqliteEventLog::open(&events_path)
            .expect("count")
            .len()
            .expect("len")
    };

    let log = Arc::new(SqliteEventLog::open(&events_path).expect("reopen"));
    let organism = Organism::open("subject", "restarts", ":memory:", &genome_path)
        .expect("second organism")
        .with_events(log);
    let ws = fresh_workspace(&root);
    act_and_record(&organism, &ws, "turn-2").expect("turn 2");
    drop(organism);

    let log = SqliteEventLog::open(&events_path).expect("reopen");
    assert!(
        log.len().expect("len") > first_run_events,
        "a restart must continue the log, not start a new one"
    );
    assert_eq!(
        log.turn("turn-1").expect("turn").len(),
        2,
        "the earlier turn must still be readable after the restart"
    );
}
