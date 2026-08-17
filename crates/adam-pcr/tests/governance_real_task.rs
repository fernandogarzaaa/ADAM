//! Does real-environment evidence reach the *same* adaptation gate EVE's does?
//!
//! The question this file answers is not whether the workspace can be measured
//! — `real_task_fitness.rs` settles that — but whether a measurement from a
//! second evaluator is governed identically to a measurement from the first.
//! The answer has to be yes for a structural reason: an organism with two ways
//! to authorise a change has two rate limits, two audit logs, and one of them
//! is the easier one. That is not governance, it is a bypass with paperwork.
//!
//! So there is no `RealTaskGovernance` here and no second gate. The organism is
//! handed the real-task provider through the one provider slot it already has,
//! and everything after that is the existing path, unmodified.

use adam_eve::EveClient;
use adam_evolution::{EvolutionProposal, ProposalKind};
use adam_kernel::Genome;
use adam_organism::Organism;
use adam_pcr::RealTaskFitnessProvider;
use adam_protocol::{Component, EventKind, Recommendation, RecordingSink};

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
const NAME: &str = "subject";
const DESCRIPTION: &str = "governance integration over a real environment";

struct Scratch(std::path::PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is before the epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("adam-realgov-{label}-{nonce}"));
        std::fs::create_dir_all(&dir).expect("could not create the scratch directory");
        Self(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn records() -> Vec<(String, String)> {
    RECORDS
        .iter()
        .map(|(name, body)| ((*name).to_string(), (*body).to_string()))
        .collect()
}

/// An organism wired to the real workspace instead of to EVE.
///
/// The provider's baseline genome is built from the same name and description
/// the organism is built from, so the two agree on what "before" is. If they
/// did not, the request's `genome_before_hash` would not match and the provider
/// would refuse the measurement — the pinning check doing its job, but making
/// this test fail for a reason that has nothing to do with governance.
fn organism_on_the_workspace(scratch: &Scratch, sink: std::sync::Arc<RecordingSink>) -> Organism {
    let provider = RealTaskFitnessProvider::new(
        scratch.0.join("runs"),
        records(),
        Genome::new(NAME, DESCRIPTION),
        "turn-1",
    );
    let memory = scratch.0.join("memory.db");
    Organism::new(NAME, DESCRIPTION, memory.to_str().expect("utf-8 temp path"))
        .expect("organism opens")
        .with_eve(EveClient::new(Box::new(provider)))
        .with_events(sink)
}

fn proposal() -> EvolutionProposal {
    EvolutionProposal::new(
        ProposalKind::AmendGenome {
            field: "policies.append".to_string(),
            current_value: String::new(),
            suggested_value: POLICY.to_string(),
        },
        "processing stopped on a malformed record that a check would have caught",
        vec!["workspace objective stalled at 5000bp".to_string()],
        0.8,
    )
}

#[test]
fn real_evidence_travels_the_existing_path_and_is_accepted_by_the_one_gate() {
    let scratch = Scratch::new("accept");
    let sink = std::sync::Arc::new(RecordingSink::new());
    let mut organism = organism_on_the_workspace(&scratch, sink.clone());

    let id = organism.propose_mutation(proposal());
    organism.announce_proposal(id, "turn-1");

    let measured = organism
        .validate_mutation(id, "turn-1")
        .expect("the workspace should be measurable");

    // The known result, arriving through the organism rather than through a
    // direct call to the provider.
    assert_eq!(measured.delta_bp.raw(), 2500);
    assert_eq!(measured.recommendation, Recommendation::Approve);
    assert_eq!(measured.provenance.authored_by, Component::Pcr);

    organism
        .accept_mutation(id, "turn-1")
        .expect("an approved mutation should pass the rate limit");

    // One audit log, holding this acceptance. The absence of a second log is
    // the actual claim being made here.
    let audit = organism.audit_log();
    assert_eq!(audit.len(), 1, "exactly one governance decision was made");

    // And the change really happened: the organism now carries the policy.
    assert!(organism
        .genome()
        .policies
        .iter()
        .any(|policy| policy == POLICY));
}

#[test]
fn the_fitness_event_names_pcr_as_the_measurer_not_eve() {
    // The organism's provider slot is called `eve`. If the emitted event took
    // its actor from that field, or from the event kind, a PCR measurement
    // would be logged as EVE's — internally consistent, factually wrong, and
    // invisible to anyone auditing the log later.
    let scratch = Scratch::new("actor");
    let sink = std::sync::Arc::new(RecordingSink::new());
    let mut organism = organism_on_the_workspace(&scratch, sink.clone());

    let id = organism.propose_mutation(proposal());
    organism
        .validate_mutation(id, "turn-1")
        .expect("measurable");

    let fitness: Vec<_> = sink
        .events()
        .into_iter()
        .filter(|e| e.kind == EventKind::FitnessMeasured)
        .collect();
    assert_eq!(fitness.len(), 1);
    assert_eq!(fitness[0].actor, Component::Pcr);
}

/// A preference amendment, which the organism may accept without consulting an
/// evaluator at all.
///
/// Used below precisely because it takes the *other* route to acceptance. If
/// the allowance were kept per-evaluator, these would draw on a different
/// budget than the real-task acceptance and the limit would never be reached
/// within one window.
fn preference_proposal(n: usize) -> EvolutionProposal {
    EvolutionProposal::new(
        ProposalKind::AmendGenome {
            field: format!("preferences.tone{n}"),
            current_value: String::new(),
            suggested_value: "terse".to_string(),
        },
        "an unrelated preference, to draw on the same allowance",
        vec![],
        0.9,
    )
}

#[test]
fn real_evidence_draws_on_the_same_allowance_as_every_other_acceptance() {
    // The point of one gate is that a second evaluator does not get a second
    // allowance. The default limit is 5 acceptances per window, so one
    // real-task acceptance plus four unrelated ones must exhaust it — and the
    // sixth must be refused. A per-evaluator budget would not behave this way.
    //
    // The real-task acceptance goes first and only once, deliberately: after it
    // lands, the organism's genome carries the new policy and the provider's
    // pinned baseline no longer describes it. A second real-task measurement
    // would be correctly refused for staleness, which is the pinning check
    // working, not the rate limit.
    let scratch = Scratch::new("limit");
    let sink = std::sync::Arc::new(RecordingSink::new());
    let mut organism = organism_on_the_workspace(&scratch, sink);

    let id = organism.propose_mutation(proposal());
    organism
        .validate_mutation(id, "turn-1")
        .expect("measurable");
    organism
        .accept_mutation(id, "turn-1")
        .expect("the first acceptance is within the allowance");

    let mut accepted = 1usize;
    let mut refusal = None;
    for n in 0..8 {
        let id = organism.propose_mutation(preference_proposal(n));
        match organism.accept_mutation(id, "turn-1") {
            Ok(_) => accepted += 1,
            Err(err) => {
                refusal = Some(err);
                break;
            }
        }
    }

    let err = refusal.expect("the shared allowance must eventually refuse an acceptance");
    assert!(
        err.to_string().to_lowercase().contains("limit"),
        "the refusal should be the evolution rate limit, got: {err}"
    );
    assert_eq!(
        accepted, 5,
        "the real-task acceptance and the preference acceptances share one budget of five"
    );
    assert_eq!(
        organism.audit_log().len(),
        accepted,
        "a refused acceptance must leave no audit entry"
    );
}
