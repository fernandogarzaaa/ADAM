//! ADAM Belief System.
//!
//! Beliefs are epistemic claims with confidence derived from an explicit
//! evidence trail (provenance-first, matching `adam-memory`'s pattern).
//! Confidence is updated, never overwritten; contradicting beliefs compete
//! and the loser is superseded, not deleted.

mod belief;
mod registry;

pub use belief::{Belief, BeliefError, BeliefId, BeliefStatus, Evidence, EvidenceOrigin};
pub use registry::BeliefRegistry;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn belief_forms_with_initial_confidence_from_evidence() {
        let belief = Belief::form(
            "the build passes on main",
            EvidenceOrigin::Observation,
            "CI ran green",
            0.7,
        )
        .unwrap();
        assert_eq!(belief.confidence, 0.7);
        assert_eq!(belief.evidence.len(), 1);
        assert!(belief.is_active());
    }

    #[test]
    fn supporting_evidence_increases_confidence_toward_one() {
        let mut belief = Belief::form(
            "rust is memory safe",
            EvidenceOrigin::Reasoning,
            "type system",
            0.5,
        )
        .unwrap();
        belief
            .add_evidence(
                EvidenceOrigin::Observation,
                "no segfaults in 1000 runs",
                true,
                0.5,
            )
            .unwrap();
        assert!(belief.confidence > 0.5);
        assert_eq!(belief.evidence.len(), 2);
    }

    #[test]
    fn contradicting_evidence_decreases_confidence_and_can_retract() {
        let mut belief = Belief::form(
            "the api is stable",
            EvidenceOrigin::Observation,
            "no errors seen",
            0.6,
        )
        .unwrap();
        belief
            .add_evidence(
                EvidenceOrigin::Observation,
                "endpoint changed shape",
                false,
                1.0,
            )
            .unwrap();
        assert_eq!(belief.confidence, 0.0);
        assert_eq!(belief.status, BeliefStatus::Retracted);
    }

    #[test]
    fn adding_evidence_to_inactive_belief_errors() {
        let mut belief = Belief::form("x", EvidenceOrigin::Reasoning, "seed", 1.0).unwrap();
        belief
            .add_evidence(EvidenceOrigin::Observation, "kill it", false, 1.0)
            .unwrap();
        assert_eq!(belief.status, BeliefStatus::Retracted);

        let err = belief
            .add_evidence(EvidenceOrigin::Observation, "too late", true, 0.5)
            .unwrap_err();
        assert_eq!(err, BeliefError::NotActive(belief.id));
    }

    #[test]
    fn out_of_range_weight_is_rejected() {
        let err = Belief::form("x", EvidenceOrigin::Reasoning, "seed", 1.5).unwrap_err();
        assert!(matches!(err, BeliefError::WeightOutOfRange(_)));
    }

    #[test]
    fn competing_beliefs_resolve_by_confidence_and_supersede_the_loser() {
        let mut registry = BeliefRegistry::new();

        let strong = Belief::form(
            "the outage was caused by a bad deploy",
            EvidenceOrigin::Observation,
            "deploy timestamp matches outage start",
            0.9,
        )
        .unwrap();
        let weak = Belief::form(
            "the outage was caused by network issues",
            EvidenceOrigin::UserAssertion,
            "on-call guessed",
            0.3,
        )
        .unwrap();

        let strong_id = strong.id;
        let weak_id = weak.id;
        registry.upsert(strong);
        registry.upsert(weak);

        registry.mark_contradicting(strong_id, weak_id);
        let winner = registry.resolve_conflict(strong_id, weak_id).unwrap();

        assert_eq!(winner, strong_id);
        assert!(registry.get(strong_id).unwrap().is_active());
        assert_eq!(
            registry.get(weak_id).unwrap().status,
            BeliefStatus::Superseded { by: strong_id }
        );
    }

    #[test]
    fn resolving_non_contradicting_beliefs_errors() {
        let mut registry = BeliefRegistry::new();
        let a = Belief::form("a", EvidenceOrigin::Reasoning, "seed", 0.5).unwrap();
        let b = Belief::form("b", EvidenceOrigin::Reasoning, "seed", 0.5).unwrap();
        let (id_a, id_b) = (a.id, b.id);
        registry.upsert(a);
        registry.upsert(b);

        let err = registry.resolve_conflict(id_a, id_b).unwrap_err();
        assert_eq!(err, BeliefError::NotContradicting(id_a, id_b));
    }

    #[test]
    fn find_about_matches_active_beliefs_by_statement_substring() {
        let mut registry = BeliefRegistry::new();
        registry.upsert(
            Belief::form(
                "Rust prevents data races",
                EvidenceOrigin::Reasoning,
                "seed",
                0.8,
            )
            .unwrap(),
        );
        registry.upsert(
            Belief::form(
                "Python is dynamically typed",
                EvidenceOrigin::Reasoning,
                "seed",
                0.8,
            )
            .unwrap(),
        );

        let matches = registry.find_about("rust");
        assert_eq!(matches.len(), 1);
        assert!(matches[0].statement.contains("Rust"));
    }
}
