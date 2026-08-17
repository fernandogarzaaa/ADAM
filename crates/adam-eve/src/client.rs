//! Turning an ADAM proposal into something EVE can measure.
//!
//! This module owns two pieces of policy that are genuinely ADAM's, and defers
//! everything else to EVE:
//!
//! - **Intrinsic risk.** How consequential a proposal is, as a function of what
//!   it changes. ADAM knows which of its own fields are load-bearing; EVE does
//!   not and should not.
//! - **Seed derivation.** Which seed a measurement runs at, derived
//!   deterministically from the mutation and the genome it applies to.
//!
//! What it does *not* own is the verdict. That comes from EVE, and the only
//! thing this module does with it is check that it is authentic before letting
//! it near a governance decision.

use adam_evolution::{EvolutionProposal, ProposalKind};
use adam_protocol::{
    BasisPoints, Component, FitnessResult, GenomePair, MeasurementPlan, Mutation, MutationKind,
    MutationStatus, Provenance, ValidationRequest,
};

use crate::provider::{measure_and_verify, FitnessError, FitnessProvider};

/// How a mutation is measured: against which scenarios, how many times, and at
/// which seed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationConfig {
    /// EVE scenario ids. Empty means EVE's default suite.
    pub scenario_ids: Vec<String>,
    /// Repetitions per persona per scenario.
    pub trials: u32,
    /// The seed to measure at, when the caller has one.
    ///
    /// `None` derives a seed from the mutation id and the genome hash, which is
    /// reproducible for one *proposal object* but not across proposals: an
    /// `EvolutionProposal` takes a fresh uuid at construction, so proposing the
    /// same logical change twice yields two ids and therefore two seeds. A
    /// repeated experiment then quietly runs at a different seed, and the
    /// difference between the two answers cannot be attributed — it may be the
    /// mutation, or it may be the seed.
    ///
    /// Set this whenever two measurements are meant to be comparable. The value
    /// actually used is recorded on the request either way
    /// (`MeasurementPlan::seed`), so which seed a measurement ran at is never a
    /// matter of reconstruction.
    pub seed: Option<u32>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            // EVE's three construct-validated reference apps: the instrument
            // whose ability to separate good software from bad is what makes a
            // fitness delta mean anything.
            scenario_ids: vec![
                "excellent".to_string(),
                "average".to_string(),
                "bad".to_string(),
            ],
            // Three trials per persona per scenario. One would let a single
            // unlucky run decide a genome amendment; more would multiply a
            // measurement that already runs a full browser simulation for every
            // repetition.
            trials: 3,
            // Derive by default: a caller who has not thought about seeds still
            // gets a reproducible measurement for the proposal in hand.
            seed: None,
        }
    }
}

/// Intrinsic risk of a proposal, as a function of what it changes.
///
/// Deliberately independent of the proposal's own `confidence`. Factoring in
/// self-reported confidence would let a caller lower a proposal's risk simply
/// by asserting more confidence in it, which defeats the point of scoring
/// proposals against external evidence.
///
/// Genome amendments carry the highest baseline because they touch identity and
/// are only reversible by a forward commit, never a silent undo. Skill
/// retirement is lowest: it narrows available behavior without changing what
/// the organism is.
pub fn intrinsic_risk(kind: &ProposalKind) -> BasisPoints {
    let raw = match kind {
        ProposalKind::RetireSkill { .. } => 0.1,
        ProposalKind::InvestigateConflict { .. } => 0.2,
        ProposalKind::ReconcileBelief { .. } => 0.3,
        ProposalKind::AmendGenome { field, .. } => {
            // Preferences are low-stakes and reversible; the rest of the genome
            // is identity. Collapsing both into one number would either
            // over-gate preferences or under-gate values.
            if field.starts_with("preferences.") {
                0.25
            } else {
                0.6
            }
        }
    };
    BasisPoints::from_ratio(raw)
}

/// Project an [`EvolutionProposal`] onto the CP/1 [`Mutation`] EVE reads.
///
/// The proposal's own id becomes the mutation's id, so a `FitnessResult` can be
/// matched back to the proposal it measured without a side table.
pub fn to_mutation(proposal: &EvolutionProposal) -> Mutation {
    let (kind, target, current, proposed) = match &proposal.kind {
        ProposalKind::RetireSkill { skill_name } => {
            (MutationKind::RetireSkill, skill_name.clone(), None, None)
        }
        ProposalKind::ReconcileBelief { statement } => {
            (MutationKind::ReconcileBelief, statement.clone(), None, None)
        }
        ProposalKind::InvestigateConflict { topic } => {
            (MutationKind::InvestigateConflict, topic.clone(), None, None)
        }
        ProposalKind::AmendGenome {
            field,
            current_value,
            suggested_value,
        } => (
            MutationKind::AmendGenome,
            field.clone(),
            Some(current_value.clone()),
            Some(suggested_value.clone()),
        ),
    };

    Mutation::new(
        proposal.id.to_string(),
        kind,
        target,
        current,
        proposed,
        proposal.rationale.clone(),
        BasisPoints::from_f32(proposal.confidence),
        intrinsic_risk(&proposal.kind),
        MutationStatus::Validating,
        Provenance::now(Component::Adam, "adam:evolution/proposal")
            .with_evidence(proposal.evidence.clone()),
    )
}

/// Derive the seed a measurement runs at.
///
/// Deterministic in the mutation and the genome it applies to, so re-validating
/// the same proposal against the same genome reproduces the same measurement
/// exactly. A random seed would make every re-validation a different
/// experiment, and "we measured it again and got a different answer" is
/// indistinguishable from "the mutation is marginal".
///
/// FNV-1a over the two identifiers: not cryptographic, and does not need to be
/// — the requirement is reproducibility, not unpredictability.
pub fn derive_seed(mutation_id: &str, genome_before_hash: &str) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for byte in mutation_id
        .bytes()
        .chain(b"@".iter().copied())
        .chain(genome_before_hash.bytes())
    {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

/// ADAM's client for EVE's fitness measurement.
///
/// Holds a [`FitnessProvider`] and the configuration a measurement runs under.
/// It has no opinion about the verdict — its job is to ask a well-formed
/// question and confirm the answer came from EVE.
pub struct EveClient {
    provider: Box<dyn FitnessProvider>,
    config: ValidationConfig,
}

impl EveClient {
    pub fn new(provider: Box<dyn FitnessProvider>) -> Self {
        Self {
            provider,
            config: ValidationConfig::default(),
        }
    }

    pub fn with_config(mut self, config: ValidationConfig) -> Self {
        self.config = config;
        self
    }

    pub fn config(&self) -> &ValidationConfig {
        &self.config
    }

    /// The provider's identity, for audit entries.
    pub fn provider(&self) -> String {
        self.provider.describe()
    }

    /// Build the request that would measure `proposal`.
    ///
    /// Separate from [`EveClient::validate`] so the exact question ADAM asks is
    /// inspectable and testable without a provider running.
    ///
    /// The seed comes from [`ValidationConfig::seed`] when the caller supplied
    /// one, and is derived otherwise. Either way it is written to the request's
    /// [`MeasurementPlan`] *and* named in provenance evidence, so a reader of
    /// the request never has to recompute which experiment was run — and two
    /// requests that claim to be the same experiment can be compared on the
    /// point that decides whether they are.
    pub fn request_for(
        &self,
        proposal: &EvolutionProposal,
        genome_before_hash: &str,
        genome_after_hash: &str,
    ) -> ValidationRequest {
        let mutation = to_mutation(proposal);
        let (seed, origin) = match self.config.seed {
            Some(seed) => (seed, "supplied"),
            None => (derive_seed(&mutation.id, genome_before_hash), "derived"),
        };
        ValidationRequest::new(
            uuid::Uuid::new_v4().to_string(),
            mutation,
            GenomePair::new(genome_before_hash, genome_after_hash),
            MeasurementPlan {
                scenario_ids: self.config.scenario_ids.clone(),
                seed,
                trials: self.config.trials,
            },
            Provenance::now(Component::Adam, "adam:evolution/validate")
                .derived_from([proposal.id.to_string()])
                .with_evidence([format!("seed={seed} ({origin})")]),
        )
    }

    /// Measure `proposal` and return EVE's verdict.
    ///
    /// The result is guaranteed authentic for this proposal — authored by EVE,
    /// about this mutation, with a symmetric comparison — or an error. There is
    /// no path that returns an unverified result.
    pub fn validate(
        &self,
        proposal: &EvolutionProposal,
        genome_before_hash: &str,
        genome_after_hash: &str,
    ) -> Result<FitnessResult, FitnessError> {
        let request = self.request_for(proposal, genome_before_hash, genome_after_hash);
        measure_and_verify(self.provider.as_ref(), &request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adam_protocol::{Measurement, Recommendation, SignedBasisPoints};

    fn amend(field: &str) -> EvolutionProposal {
        EvolutionProposal::new(
            ProposalKind::AmendGenome {
                field: field.to_string(),
                current_value: "low".to_string(),
                suggested_value: "high".to_string(),
            },
            "evidence suggests a change",
            vec!["observed 4 times".to_string()],
            0.7,
        )
    }

    fn retire() -> EvolutionProposal {
        EvolutionProposal::new(
            ProposalKind::RetireSkill {
                skill_name: "flaky".to_string(),
            },
            "chronically failing",
            vec!["timeout".to_string()],
            0.9,
        )
    }

    fn client_seeded(seed: Option<u32>) -> EveClient {
        EveClient::new(Box::new(crate::provider::StubProvider::failing("unused"))).with_config(
            ValidationConfig {
                seed,
                ..ValidationConfig::default()
            },
        )
    }

    /// The confound itself, demonstrated rather than asserted.
    ///
    /// Two proposals of the *same logical change* are two objects with two
    /// uuids, so a derived seed differs between them. Nothing warns the caller:
    /// the two measurements simply run different experiments, and the gap
    /// between their answers is unattributable.
    #[test]
    fn a_derived_seed_differs_between_two_proposals_of_the_same_change() {
        let a = client_seeded(None).request_for(&amend("preferences.x"), "before", "after");
        let b = client_seeded(None).request_for(&amend("preferences.x"), "before", "after");
        assert_ne!(a.seed, b.seed);
    }

    #[test]
    fn an_explicit_seed_makes_two_proposals_of_the_same_change_comparable() {
        let a = client_seeded(Some(4242)).request_for(&amend("preferences.x"), "before", "after");
        let b = client_seeded(Some(4242)).request_for(&amend("preferences.x"), "before", "after");
        assert_eq!(a.seed, 4242);
        assert_eq!(a.seed, b.seed);
    }

    /// Recorded, not reconstructable: a reader must be able to see which seed
    /// ran and whether anyone chose it, without recomputing anything.
    #[test]
    fn the_seed_and_its_origin_are_persisted_in_provenance() {
        let supplied = client_seeded(Some(7)).request_for(&retire(), "before", "after");
        assert!(supplied
            .provenance
            .evidence
            .contains(&"seed=7 (supplied)".to_string()));

        let derived = client_seeded(None).request_for(&retire(), "before", "after");
        let expected = format!("seed={} (derived)", derived.seed);
        assert!(derived.provenance.evidence.contains(&expected));
    }

    fn measurement(runs: u32) -> Measurement {
        Measurement::experience(
            BasisPoints::from_ratio(0.64),
            BasisPoints::from_ratio(0.7),
            BasisPoints::from_ratio(0.3),
            BasisPoints::from_ratio(0.6),
            BasisPoints::from_ratio(0.4),
            runs,
        )
    }

    pub(crate) fn fitness_for(mutation_id: &str, author: Component) -> FitnessResult {
        FitnessResult {
            cp: "cp1".to_string(),
            doc_type: "FitnessResult".to_string(),
            id: uuid::Uuid::new_v4().to_string(),
            mutation_id: mutation_id.to_string(),
            seed: 1337,
            scenario_ids: vec!["excellent".to_string()],
            trials: 3,
            baseline: measurement(9),
            candidate: measurement(9),
            delta_bp: SignedBasisPoints::new(700),
            recommendation: Recommendation::Approve,
            reason: "improved".to_string(),
            provenance: Provenance::now(author, "eve:cp1/validate"),
        }
    }

    #[test]
    fn genome_amendments_outrank_preferences_which_outrank_skill_retirement() {
        let values = intrinsic_risk(&amend("values.append").kind);
        let preferences = intrinsic_risk(&amend("preferences.tone").kind);
        let skill = intrinsic_risk(&retire().kind);
        assert!(values > preferences);
        assert!(preferences > skill);
    }

    #[test]
    fn risk_ignores_a_proposals_self_reported_confidence() {
        // Otherwise a caller could lower a proposal's risk by asserting more
        // confidence in it, which is the whole thing external measurement
        // exists to prevent.
        let mut timid = amend("values.append");
        timid.confidence = 0.01;
        let mut bold = amend("values.append");
        bold.confidence = 0.99;
        assert_eq!(intrinsic_risk(&timid.kind), intrinsic_risk(&bold.kind));
    }

    #[test]
    fn a_proposal_projects_onto_a_mutation_carrying_its_id() {
        let proposal = amend("preferences.thoroughness");
        let mutation = to_mutation(&proposal);
        assert_eq!(mutation.id, proposal.id.to_string());
        assert_eq!(mutation.kind, MutationKind::AmendGenome);
        assert_eq!(mutation.target, "preferences.thoroughness");
        assert_eq!(mutation.current_value.as_deref(), Some("low"));
        assert_eq!(mutation.proposed_value.as_deref(), Some("high"));
        assert_eq!(mutation.confidence_bp.raw(), 7_000);
        assert_eq!(mutation.status, MutationStatus::Validating);
        assert_eq!(mutation.provenance.authored_by, Component::Adam);
    }

    #[test]
    fn a_projected_mutation_carries_the_proposals_evidence() {
        let mutation = to_mutation(&retire());
        assert_eq!(mutation.provenance.evidence, vec!["timeout".to_string()]);
    }

    #[test]
    fn kinds_without_values_project_without_them() {
        for proposal in [
            retire(),
            EvolutionProposal::new(
                ProposalKind::ReconcileBelief {
                    statement: "s".to_string(),
                },
                "r",
                vec![],
                0.5,
            ),
        ] {
            let mutation = to_mutation(&proposal);
            assert!(mutation.current_value.is_none());
            assert!(mutation.proposed_value.is_none());
        }
    }

    #[test]
    fn the_seed_is_reproducible_for_the_same_mutation_and_genome() {
        let a = derive_seed("m1", "genome-hash");
        let b = derive_seed("m1", "genome-hash");
        assert_eq!(a, b);
    }

    #[test]
    fn the_seed_changes_with_either_input() {
        let base = derive_seed("m1", "genome-a");
        assert_ne!(base, derive_seed("m2", "genome-a"));
        assert_ne!(base, derive_seed("m1", "genome-b"));
    }

    #[test]
    fn the_seed_separator_prevents_boundary_collisions() {
        // Without a separator, ("ab", "c") and ("a", "bc") would hash alike.
        assert_ne!(derive_seed("ab", "c"), derive_seed("a", "bc"));
    }

    #[test]
    fn the_default_suite_is_eves_construct_validated_instrument() {
        let config = ValidationConfig::default();
        assert_eq!(config.scenario_ids, vec!["excellent", "average", "bad"]);
        assert!(config.trials >= 1);
    }

    #[test]
    fn a_request_pins_the_genome_pair_and_a_derived_seed() {
        let client = EveClient::new(Box::new(crate::provider::StubProvider::failing("unused")));
        let proposal = amend("goals.append");
        let request = client.request_for(&proposal, "before-hash", "after-hash");

        assert_eq!(request.genome_before_hash, "before-hash");
        assert_eq!(request.genome_after_hash, "after-hash");
        assert_eq!(
            request.seed,
            derive_seed(&proposal.id.to_string(), "before-hash")
        );
        assert_eq!(request.mutation.id, proposal.id.to_string());
        assert!(request.trials >= 1);
    }

    #[test]
    fn a_request_seals_and_verifies() {
        let client = EveClient::new(Box::new(crate::provider::StubProvider::failing("unused")));
        let request = client.request_for(&amend("goals.append"), "b", "a");
        let sealed = request.seal().unwrap();
        assert!(adam_protocol::verify_seal(&sealed).unwrap());
        assert_eq!(sealed["type"], serde_json::json!("ValidationRequest"));
    }

    #[test]
    fn validate_returns_an_authentic_result() {
        let proposal = amend("goals.append");
        let client = EveClient::new(Box::new(crate::provider::StubProvider::returning(
            fitness_for(&proposal.id.to_string(), Component::Eve),
        )));
        let result = client.validate(&proposal, "b", "a").unwrap();
        assert_eq!(result.recommendation, Recommendation::Approve);
    }

    #[test]
    fn validate_refuses_a_result_adam_authored() {
        // A provider that returned an ADAM-authored result would be
        // reintroducing self-scored evidence. The client refuses it regardless
        // of what the provider claims.
        let proposal = amend("goals.append");
        let client = EveClient::new(Box::new(crate::provider::StubProvider::returning(
            fitness_for(&proposal.id.to_string(), Component::Adam),
        )));
        let err = client.validate(&proposal, "b", "a").unwrap_err();
        assert!(matches!(err, FitnessError::Inauthentic { .. }));
        // Rejected for *being ADAM*, not for failing to be EVE: the rule now
        // admits a second evaluator, and this test must keep failing for the
        // reason that still matters after that generalization.
        assert!(
            err.to_string().contains("is not an evaluator"),
            "unexpected reason: {err}"
        );
    }

    #[test]
    fn validate_refuses_a_result_about_a_different_mutation() {
        let client = EveClient::new(Box::new(crate::provider::StubProvider::returning(
            fitness_for("some-other-mutation", Component::Eve),
        )));
        let err = client
            .validate(&amend("goals.append"), "b", "a")
            .unwrap_err();
        assert!(matches!(err, FitnessError::Inauthentic { .. }));
    }

    #[test]
    fn validate_propagates_a_provider_failure_rather_than_defaulting() {
        // There is no "measurement failed, proceed anyway" path.
        let client = EveClient::new(Box::new(crate::provider::StubProvider::failing(
            "scenario suite unavailable",
        )));
        let err = client
            .validate(&amend("goals.append"), "b", "a")
            .unwrap_err();
        assert!(err.to_string().contains("scenario suite unavailable"));
    }
}
