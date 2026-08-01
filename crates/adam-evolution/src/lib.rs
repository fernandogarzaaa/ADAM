//! ADAM Evolution Engine.
//!
//! Analyzes signals distilled from an organism's history (skill failures,
//! belief instability, recurring conflicts, genome drift) and produces
//! [`EvolutionProposal`]s. The engine never applies a mutation itself —
//! proposals sit `Proposed` until an external governance decision accepts
//! or rejects them, and applying an accepted proposal is the job of
//! whatever owns the genome/skill/belief state being changed.

mod engine;
mod proposal;
mod signal;
mod store;

pub use engine::{EvolutionEngine, EvolutionThresholds};
pub use proposal::{EvolutionProposal, ProposalError, ProposalId, ProposalKind, ProposalStatus};
pub use signal::{
    BeliefInstabilitySignal, EvolutionSignals, GenomeDriftSignal, RecurringConflictSignal,
    SkillFailureSignal,
};
pub use store::ProposalStore;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chronically_failing_skill_produces_retire_proposal() {
        let signals = EvolutionSignals {
            skill_failures: vec![SkillFailureSignal {
                skill_name: "flaky-parser".to_string(),
                fitness_score: 0.2,
                failure_count: 4,
                failures: vec!["timeout".to_string(), "panic".to_string()],
            }],
            ..Default::default()
        };

        let engine = EvolutionEngine::new(EvolutionThresholds::default());
        let proposals = engine.analyze(&signals);

        assert_eq!(proposals.len(), 1);
        match &proposals[0].kind {
            ProposalKind::RetireSkill { skill_name } => assert_eq!(skill_name, "flaky-parser"),
            other => panic!("expected RetireSkill, got {other:?}"),
        }
        assert_eq!(proposals[0].status, ProposalStatus::Proposed);
    }

    #[test]
    fn healthy_skill_produces_no_proposal() {
        let signals = EvolutionSignals {
            skill_failures: vec![SkillFailureSignal {
                skill_name: "reliable-skill".to_string(),
                fitness_score: 0.9,
                failure_count: 1,
                failures: vec![],
            }],
            ..Default::default()
        };

        let engine = EvolutionEngine::new(EvolutionThresholds::default());
        assert!(engine.analyze(&signals).is_empty());
    }

    #[test]
    fn unstable_belief_produces_reconcile_proposal() {
        let signals = EvolutionSignals {
            belief_instabilities: vec![BeliefInstabilitySignal {
                statement: "the cache is always warm".to_string(),
                confidence: 0.1,
                retraction_count: 3,
            }],
            ..Default::default()
        };

        let engine = EvolutionEngine::new(EvolutionThresholds::default());
        let proposals = engine.analyze(&signals);
        assert_eq!(proposals.len(), 1);
        assert!(matches!(proposals[0].kind, ProposalKind::ReconcileBelief { .. }));
    }

    #[test]
    fn recurring_conflict_and_genome_drift_produce_proposals() {
        let signals = EvolutionSignals {
            recurring_conflicts: vec![RecurringConflictSignal {
                topic: "deployment ownership".to_string(),
                occurrences: 5,
            }],
            genome_drifts: vec![GenomeDriftSignal {
                field: "preferences.verbosity".to_string(),
                current_value: "verbose".to_string(),
                suggested_value: "concise".to_string(),
                supporting_observations: 8,
            }],
            ..Default::default()
        };

        let engine = EvolutionEngine::new(EvolutionThresholds::default());
        let proposals = engine.analyze(&signals);
        assert_eq!(proposals.len(), 2);
        assert!(proposals
            .iter()
            .any(|p| matches!(p.kind, ProposalKind::InvestigateConflict { .. })));
        assert!(proposals
            .iter()
            .any(|p| matches!(p.kind, ProposalKind::AmendGenome { .. })));
    }

    #[test]
    fn proposals_never_auto_apply_and_require_explicit_decision() {
        let mut proposal = EvolutionProposal::new(
            ProposalKind::RetireSkill {
                skill_name: "x".to_string(),
            },
            "test rationale",
            vec![],
            0.9,
        );
        assert_eq!(proposal.status, ProposalStatus::Proposed);
        assert!(proposal.decided_at.is_none());

        proposal.accept().unwrap();
        assert_eq!(proposal.status, ProposalStatus::Accepted);
        assert!(proposal.decided_at.is_some());

        let err = proposal.reject().unwrap_err();
        assert_eq!(
            err,
            ProposalError::AlreadyDecided(proposal.id, ProposalStatus::Accepted)
        );
    }

    #[test]
    fn store_tracks_pending_and_accepted_proposals_separately() {
        let mut store = ProposalStore::new();
        let signals = EvolutionSignals {
            skill_failures: vec![SkillFailureSignal {
                skill_name: "a".to_string(),
                fitness_score: 0.1,
                failure_count: 5,
                failures: vec![],
            }],
            belief_instabilities: vec![BeliefInstabilitySignal {
                statement: "b".to_string(),
                confidence: 0.1,
                retraction_count: 4,
            }],
            ..Default::default()
        };
        let engine = EvolutionEngine::new(EvolutionThresholds::default());
        let ids = store.record_all(engine.analyze(&signals));
        assert_eq!(store.len(), 2);
        assert_eq!(store.pending().len(), 2);

        store.get_mut(ids[0]).unwrap().accept().unwrap();
        assert_eq!(store.pending().len(), 1);
        assert_eq!(store.accepted().len(), 1);
    }
}
