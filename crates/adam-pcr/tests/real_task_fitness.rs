//! Does the real environment produce a measurement, and is it the *known* one?
//!
//! The prior experiment established what this mutation does to the workspace:
//! processing blindly stalls at 5000bp, verifying first reaches 7500bp. Those
//! numbers were measured before this provider existed, by driving the workspace
//! directly (`loop_mechanism.rs`). So they are an independent expectation, not
//! a target this file may choose — if the provider disagrees with them, the
//! provider is wrong, and no number here may be adjusted to hide it.
//!
//! Every measurement is taken through `measure_and_verify`, never by calling
//! `measure` directly. Going around it would check a number while skipping the
//! check that decides whether ADAM is allowed to believe the number.

use adam_eve::{measure_and_verify, FitnessProvider};
use adam_kernel::Genome;
use adam_organism::apply_list_amendment;
use adam_pcr::{action_for, Action, RealTaskFitnessProvider};
use adam_protocol::{
    BasisPoints, Component, EventKind, GenomePair, MeasurementPlan, Mutation, MutationKind,
    MutationStatus, Provenance, Recommendation, RecordingSink, ValidationRequest,
};

/// The same four records the direct experiment used, third one malformed.
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
const MUTATION_ID: &str = "88888888-8888-4888-8888-888888888888";

/// A temp directory that removes itself, so a failing assertion cannot leave
/// run workspaces behind for the next test to trip over.
struct Scratch(std::path::PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is before the epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("adam-realfit-{label}-{nonce}"));
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

fn baseline_genome() -> Genome {
    Genome::new("subject", "real-task fitness test")
}

/// The mutation under test, and a request pinned to it.
///
/// The genome hashes are computed with the same `apply_list_amendment` the
/// provider will use, which is the point of that function being shared: if the
/// two disagreed about what the mutation means, the request would pin one
/// change while the provider measured another.
fn request(trials: u32) -> ValidationRequest {
    let baseline = baseline_genome();
    let mut candidate = baseline.clone();
    apply_list_amendment(&mut candidate, "policies.append", "", POLICY)
        .expect("the amendment should apply to a genome with no policies");

    let mutation = Mutation::new(
        MUTATION_ID,
        MutationKind::AmendGenome,
        "policies.append",
        Some(String::new()),
        Some(POLICY.to_string()),
        "processing stopped on a malformed record that a check would have caught",
        BasisPoints::from_ratio(0.8),
        BasisPoints::from_ratio(0.2),
        MutationStatus::Validating,
        Provenance::now(Component::Adam, "adam:evolution/proposal"),
    );

    ValidationRequest::new(
        "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
        mutation,
        GenomePair::new(baseline.content_hash(), candidate.content_hash()),
        MeasurementPlan {
            scenario_ids: vec!["workspace".to_string()],
            seed: 1337,
            trials,
        },
        Provenance::now(Component::Adam, "adam:evolution/validate"),
    )
}

#[test]
fn the_real_environment_reproduces_the_known_result() {
    let scratch = Scratch::new("known");
    let provider = RealTaskFitnessProvider::new(&scratch.0, records(), baseline_genome(), "turn-1");

    let result = measure_and_verify(&provider, &request(1)).expect("a measurable comparison");

    // The three numbers the direct experiment established. Asserted separately
    // so a failure says which half moved, not merely that the delta changed.
    assert_eq!(
        result.baseline.composite_bp.raw(),
        5000,
        "processing blindly should stall halfway through the records"
    );
    assert_eq!(
        result.candidate.composite_bp.raw(),
        7500,
        "verifying first should set the bad record aside and finish the rest"
    );
    assert_eq!(result.delta_bp.raw(), 2500);
    assert_eq!(result.recommendation, Recommendation::Approve);
}

#[test]
fn the_measurement_is_authored_by_pcr_and_never_by_adam() {
    let scratch = Scratch::new("author");
    let provider = RealTaskFitnessProvider::new(&scratch.0, records(), baseline_genome(), "turn-1");
    let result = measure_and_verify(&provider, &request(1)).expect("a measurable comparison");

    assert_eq!(provider.evaluator(), Component::Pcr);
    assert_eq!(result.provenance.authored_by, Component::Pcr);
    assert!(
        result.is_authentic(MUTATION_ID, Component::Pcr),
        "the result must satisfy the same authenticity check EVE's results face"
    );
    assert!(
        !result.is_authentic(MUTATION_ID, Component::Adam),
        "ADAM must not be able to pass as the author of its own evidence"
    );
}

#[test]
fn it_reports_an_objective_and_invents_no_experience() {
    let scratch = Scratch::new("objective");
    let provider = RealTaskFitnessProvider::new(&scratch.0, records(), baseline_genome(), "turn-1");
    let result = measure_and_verify(&provider, &request(1)).expect("a measurable comparison");

    // The workspace has no simulated human. A frustration number here would be
    // fabricating the other evaluator's objective, which is exactly what the
    // optional members exist to make impossible.
    for side in [&result.baseline, &result.candidate] {
        assert!(side.task_success_bp.is_none());
        assert!(side.frustration_bp.is_none());
        assert!(side.trust_bp.is_none());
        assert!(side.cognitive_load_bp.is_none());
    }
}

#[test]
fn every_run_is_announced_by_pcr_and_chained_to_the_measurement() {
    let scratch = Scratch::new("events");
    let sink = RecordingSink::new();
    let provider = RealTaskFitnessProvider::new(&scratch.0, records(), baseline_genome(), "turn-1")
        .announcing_to(&sink);

    let result = measure_and_verify(&provider, &request(2)).expect("a measurable comparison");

    let kinds = sink.kinds();
    assert_eq!(
        kinds.len(),
        4,
        "two trials on each side of the counterfactual"
    );
    assert!(kinds
        .iter()
        .all(|kind| *kind == EventKind::TaskRunCompleted));
    assert!(sink.events().iter().all(|e| e.actor == Component::Pcr));

    // A measured result must name the runs behind it. Without this the numbers
    // would be unattributable assertions rather than evidence.
    for event in sink.events() {
        assert!(
            result.provenance.derived_from.contains(&event.id),
            "run {} is missing from the measurement's provenance",
            event.id
        );
    }
}

#[test]
fn both_sides_run_the_same_number_of_times() {
    let scratch = Scratch::new("symmetry");
    let provider = RealTaskFitnessProvider::new(&scratch.0, records(), baseline_genome(), "turn-1");
    let result = measure_and_verify(&provider, &request(3)).expect("a measurable comparison");

    assert_eq!(result.baseline.runs, 3);
    assert_eq!(result.candidate.runs, 3);
    assert_eq!(result.trials, 3);
}

#[test]
fn each_run_starts_from_a_clean_workspace() {
    // Repeating a side must not accumulate. Without the reset the second trial
    // would open a workspace whose `done/` already held the first trial's
    // output, and would score higher than the first for no reason at all.
    let scratch = Scratch::new("reset");
    let provider = RealTaskFitnessProvider::new(&scratch.0, records(), baseline_genome(), "turn-1");

    let once = measure_and_verify(&provider, &request(1)).expect("a measurable comparison");
    let thrice = measure_and_verify(&provider, &request(3)).expect("a measurable comparison");

    assert_eq!(
        once.baseline.composite_bp.raw(),
        thrice.baseline.composite_bp.raw()
    );
    assert_eq!(
        once.candidate.composite_bp.raw(),
        thrice.candidate.composite_bp.raw()
    );
}

#[test]
fn a_request_pinned_to_a_different_genome_is_refused() {
    let scratch = Scratch::new("pinning");
    // A provider holding a genome that already carries some other policy.
    // Applying the mutation to it cannot reproduce the request's hashes, so the
    // comparison would be about a different change than the one proposed.
    let mut other = baseline_genome();
    other.policies.push("something else entirely".to_string());

    let provider = RealTaskFitnessProvider::new(&scratch.0, records(), other, "turn-1");
    let err = measure_and_verify(&provider, &request(1))
        .expect_err("a mismatched pinning must not silently measure something else");
    assert!(
        err.to_string().contains("pinned"),
        "the error must name the pinning failure, got: {err}"
    );
}

#[test]
fn a_mutation_that_changes_nothing_is_not_measurable() {
    let scratch = Scratch::new("noop");
    // The policy is already in place, so the "candidate" is the baseline. A
    // delta of zero here would be a lie: nothing was compared.
    let mut already = baseline_genome();
    already.policies.push(POLICY.to_string());

    let provider = RealTaskFitnessProvider::new(&scratch.0, records(), already, "turn-1");
    let err = measure_and_verify(&provider, &request(1))
        .expect_err("an unchanged genome offers nothing to compare");
    let reason = err.to_string();
    assert!(
        reason.contains("nothing to compare") || reason.contains("pinned"),
        "unexpected reason: {reason}"
    );
}

#[test]
fn the_policy_is_what_selects_the_careful_action() {
    // Guards the assumption the whole measurement rests on: that this mutation
    // reaches behaviour at all. If `action_for` stopped reading the policy, the
    // provider would faithfully measure two identical runs and report no
    // difference, and the reproduction test above would fail for a reason
    // nobody could see from its own assertions.
    let baseline = baseline_genome();
    assert_eq!(action_for(&baseline), Action::ProcessAll);

    let mut candidate = baseline.clone();
    apply_list_amendment(&mut candidate, "policies.append", "", POLICY).expect("amendment applies");
    assert_eq!(action_for(&candidate), Action::VerifyThenProcess);
}
