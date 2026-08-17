//! End-to-end: ADAM measures a real mutation by spawning the real EVE.
//!
//! Every other test in this crate uses a stub, which proves ADAM's half of the
//! contract but not that the two halves fit. This one runs the actual
//! transport: ADAM builds a `ValidationRequest`, spawns EVE's `eve-cp1`
//! endpoint, and parses what comes back — exercising canonical encoding,
//! sealing, envelope verification and the line protocol across a language
//! boundary, in both directions.
//!
//! It is skipped unless EVE is available, because ADAM must not grow a build
//! dependency on a TypeScript project — that dependency edge is precisely what
//! CP/1 exists to avoid. Point `ADAM_EVE_CP1` at a built EVE checkout to run it:
//!
//! ```sh
//! cd ../experience-validation-engine && npm ci && npm run build
//! ADAM_EVE_CP1="$PWD/bin/eve-cp1.js" cargo test -p adam-eve --test eve_integration
//! ```
//!
//! CI wires this up in the `eve-integration` job.

use std::path::Path;
use std::time::Duration;

use adam_eve::{Cp1Subprocess, EveClient, ValidationConfig};
use adam_evolution::{EvolutionProposal, ProposalKind};
use adam_protocol::{Component, Recommendation};

/// The EVE endpoint to drive, or `None` when EVE is not available here.
fn endpoint() -> Option<Cp1Subprocess> {
    let script = std::env::var("ADAM_EVE_CP1").ok()?;
    if !Path::new(&script).exists() {
        // An env var pointing at nothing is a misconfiguration worth failing
        // on: silently skipping would let a broken CI job report success.
        panic!("ADAM_EVE_CP1 is set to {script:?}, which does not exist");
    }
    Some(Cp1Subprocess::command("node", vec![script]).with_timeout(Duration::from_secs(600)))
}

fn client(provider: Cp1Subprocess) -> EveClient {
    EveClient::new(Box::new(provider)).with_config(ValidationConfig {
        // One scenario and one trial: this test proves the wire works, not
        // that the measurement is statistically strong. The full suite runs
        // per measurement in production and takes minutes.
        scenario_ids: vec!["excellent".to_string()],
        trials: 1,
        // Derived, as in production when no caller pins one. The seed the
        // request actually ran at is asserted below against `derive_seed`.
        seed: None,
    })
}

/// A preference amendment, which EVE's projection table can measure.
fn measurable_proposal() -> EvolutionProposal {
    EvolutionProposal::new(
        ProposalKind::AmendGenome {
            field: "preferences.thoroughness".to_string(),
            current_value: "low".to_string(),
            suggested_value: "high".to_string(),
        },
        "the organism keeps missing information it had access to",
        vec!["4 recurring conflicts on the same topic".to_string()],
        0.7,
    )
}

/// A goal amendment, which EVE deliberately declines to score.
fn unmeasurable_proposal() -> EvolutionProposal {
    EvolutionProposal::new(
        ProposalKind::AmendGenome {
            field: "goals.append".to_string(),
            current_value: String::new(),
            suggested_value: "survive model replacement".to_string(),
        },
        "the organism has no stated continuity goal",
        vec![],
        0.9,
    )
}

#[test]
fn adam_measures_a_real_mutation_through_the_real_eve() {
    let Some(provider) = endpoint() else {
        eprintln!("skipping: set ADAM_EVE_CP1 to a built EVE checkout's bin/eve-cp1.js");
        return;
    };

    let proposal = measurable_proposal();
    let result = client(provider)
        .validate(&proposal, "genome-before", "genome-after")
        .expect("EVE should answer a well-formed request");

    // Authored by EVE, about this proposal, over a symmetric comparison.
    assert_eq!(result.provenance.authored_by, Component::Eve);
    assert_eq!(result.mutation_id, proposal.id.to_string());
    assert!(result.is_authentic(&proposal.id.to_string(), Component::Eve));

    // A real measurement actually ran.
    assert!(result.baseline.runs >= 1);
    assert_eq!(result.candidate.runs, result.baseline.runs);
    assert_eq!(
        result.delta_bp.raw(),
        i32::from(result.candidate.composite_bp.raw())
            - i32::from(result.baseline.composite_bp.raw()),
        "delta must equal candidate minus baseline"
    );

    // The seed ADAM derived is the seed EVE used, which is what makes the
    // comparison reproducible rather than merely repeated.
    assert_eq!(
        result.seed,
        adam_eve::derive_seed(&proposal.id.to_string(), "genome-before")
    );
    assert!(!result.reason.is_empty());
}

#[test]
fn the_same_proposal_and_genome_reproduce_the_same_measurement() {
    let Some(provider) = endpoint() else {
        eprintln!("skipping: set ADAM_EVE_CP1 to a built EVE checkout's bin/eve-cp1.js");
        return;
    };
    let Some(second) = endpoint() else {
        return;
    };

    let proposal = measurable_proposal();
    let first = client(provider)
        .validate(&proposal, "genome-before", "genome-after")
        .expect("first measurement");
    let repeat = client(second)
        .validate(&proposal, "genome-before", "genome-after")
        .expect("second measurement");

    // Determinism is load-bearing: without it, "we measured it again and got a
    // different answer" is indistinguishable from "the mutation is marginal".
    assert_eq!(repeat.seed, first.seed);
    assert_eq!(repeat.baseline, first.baseline);
    assert_eq!(repeat.candidate, first.candidate);
    assert_eq!(repeat.delta_bp, first.delta_bp);
    assert_eq!(repeat.recommendation, first.recommendation);
}

#[test]
fn eve_declines_to_score_a_mutation_it_cannot_measure() {
    let Some(provider) = endpoint() else {
        eprintln!("skipping: set ADAM_EVE_CP1 to a built EVE checkout's bin/eve-cp1.js");
        return;
    };

    let proposal = unmeasurable_proposal();
    let result = client(provider)
        .validate(&proposal, "genome-before", "genome-after")
        .expect("EVE should answer rather than error");

    // A goal amendment changes what the organism pursues, not how it operates.
    // EVE says so instead of inventing a number — and the escalation is what
    // reaches ADAM's gate, so the amendment stays blocked.
    assert_eq!(result.recommendation, Recommendation::NeedsReview);
    assert!(
        result.reason.contains("not measurable by simulation"),
        "unexpected reason: {}",
        result.reason
    );
    assert_eq!(result.delta_bp.raw(), 0);
    assert_eq!(result.baseline, result.candidate);
}

/// The exact mutation PCR measured in the real workspace, so the two
/// evaluators can be quoted against the same change rather than against two
/// different ones.
fn policy_proposal() -> EvolutionProposal {
    EvolutionProposal::new(
        ProposalKind::AmendGenome {
            field: "policies.append".to_string(),
            current_value: String::new(),
            suggested_value: "verify records before processing them".to_string(),
        },
        "records with no separator were processed as if they were well formed",
        vec!["1 of 4 records was malformed and processed anyway".to_string()],
        0.8,
    )
}

/// Re-measure the policy amendment that PCR scores at +2500 bp.
///
/// This exists because a number quoted from an earlier session is not a
/// measurement — it is a memory of one. The comparison between PCR and EVE is
/// the central Genesis result, so the EVE side of it has to be reproducible on
/// demand rather than carried forward.
///
/// Run with `--nocapture` to read the figures; the assertions here deliberately
/// check only the properties that must hold, not the value itself. Pinning the
/// value would turn a measurement into a regression test and invite somebody to
/// edit the expectation when the engine legitimately changes.
#[test]
fn the_policy_amendment_pcr_approves_is_measured_by_eve_on_demand() {
    let Some(provider) = endpoint() else {
        eprintln!("skipping: set ADAM_EVE_CP1 to a built EVE checkout's bin/eve-cp1.js");
        return;
    };

    let proposal = policy_proposal();
    let result = EveClient::new(Box::new(provider))
        .with_config(ValidationConfig {
            // Empty means EVE's own default suite — the three benchmark apps,
            // which is the configuration the original figure was taken under.
            scenario_ids: vec![],
            trials: 3,
            seed: None,
        })
        .validate(&proposal, "genome-before", "genome-after")
        .expect("EVE should answer a well-formed request");

    eprintln!(
        "EVE re-measurement: baseline={} bp candidate={} bp delta={} bp runs={}/{} rec={:?}\n  reason: {}",
        result.baseline.composite_bp.raw(),
        result.candidate.composite_bp.raw(),
        result.delta_bp.raw(),
        result.baseline.runs,
        result.candidate.runs,
        result.recommendation,
        result.reason,
    );

    // EVE authored it, about this mutation, over a symmetric comparison.
    assert_eq!(result.provenance.authored_by, Component::Eve);
    assert!(result.is_authentic(&proposal.id.to_string(), Component::Eve));
    assert_eq!(result.candidate.runs, result.baseline.runs);
    assert!(result.baseline.runs >= 1, "a real measurement must have run");

    // "verify" is a projectable policy keyword, so EVE must not decline here.
    // If it does, the projection table changed and the comparison in
    // REAL_VS_EVE_COMPARISON.md no longer has an EVE side at all.
    assert!(
        !result.reason.contains("not measurable by simulation"),
        "EVE declined to measure a policy it has a projection for: {}",
        result.reason
    );
    assert_eq!(
        result.delta_bp.raw(),
        i32::from(result.candidate.composite_bp.raw())
            - i32::from(result.baseline.composite_bp.raw()),
        "delta must equal candidate minus baseline"
    );
}
