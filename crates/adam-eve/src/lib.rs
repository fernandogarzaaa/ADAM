//! ADAM EVE Integration.
//!
//! Adapter that scores evolution proposals via sandboxed simulation
//! trials rather than trusting a proposal's self-reported confidence.
//! EVE only scores — it never accepts or rejects a proposal itself; that
//! decision belongs to governance (Phase 8), which may use EVE's
//! recommendation as one input among others.

mod evaluator;

pub use evaluator::{
    evaluate_from_trials, EvaluationResult, EvaluationThresholds, Recommendation,
    SimulationEvaluator, TrialFn, TrialOutcome,
};

#[cfg(test)]
mod tests {
    use super::*;
    use adam_evolution::{EvolutionProposal, ProposalKind};

    fn retire_skill_proposal(confidence: f32) -> EvolutionProposal {
        EvolutionProposal::new(
            ProposalKind::RetireSkill {
                skill_name: "flaky".to_string(),
            },
            "chronically failing",
            vec!["timeout".to_string()],
            confidence,
        )
    }

    #[test]
    fn always_succeeding_trials_yield_full_fitness_and_approval() {
        let evaluator = SimulationEvaluator::new(EvaluationThresholds::default(), 10);
        let proposal = retire_skill_proposal(0.9);

        let result = evaluator.evaluate(&proposal, &|_| TrialOutcome {
            succeeded: true,
            detail: "sandbox replay succeeded".to_string(),
        });

        assert_eq!(result.fitness, 1.0);
        assert_eq!(result.trials.len(), 10);
        assert_eq!(result.recommendation, Recommendation::Approve);
    }

    #[test]
    fn always_failing_trials_yield_zero_fitness_and_rejection() {
        let evaluator = SimulationEvaluator::new(EvaluationThresholds::default(), 10);
        let proposal = retire_skill_proposal(0.9);

        let result = evaluator.evaluate(&proposal, &|_| TrialOutcome {
            succeeded: false,
            detail: "sandbox replay failed".to_string(),
        });

        assert_eq!(result.fitness, 0.0);
        assert_eq!(result.recommendation, Recommendation::Reject);
    }

    #[test]
    fn mixed_trials_yield_partial_fitness() {
        let evaluator = SimulationEvaluator::new(EvaluationThresholds::default(), 4);
        let proposal = retire_skill_proposal(0.9);

        let call_count = std::cell::Cell::new(0);
        let result = evaluator.evaluate(&proposal, &|_| {
            let n = call_count.get();
            call_count.set(n + 1);
            TrialOutcome {
                succeeded: n % 2 == 0,
                detail: format!("trial {n}"),
            }
        });

        assert_eq!(result.fitness, 0.5);
        assert_eq!(result.recommendation, Recommendation::NeedsReview);
    }

    #[test]
    fn genome_amendments_carry_higher_intrinsic_risk_than_skill_retirement() {
        let evaluator = SimulationEvaluator::new(EvaluationThresholds::default(), 5);

        let low_risk = retire_skill_proposal(0.9);
        let high_risk = EvolutionProposal::new(
            ProposalKind::AmendGenome {
                field: "values.honesty".to_string(),
                current_value: "high".to_string(),
                suggested_value: "medium".to_string(),
            },
            "drift observed",
            vec![],
            0.9,
        );

        let always_pass = |_: &EvolutionProposal| TrialOutcome {
            succeeded: true,
            detail: "ok".to_string(),
        };

        let low_result = evaluator.evaluate(&low_risk, &always_pass);
        let high_result = evaluator.evaluate(&high_risk, &always_pass);

        assert!(high_result.risk > low_result.risk);
        // High fitness but high risk on a genome amendment forces review
        // rather than a silent approve, even though trials all passed.
        assert_eq!(high_result.recommendation, Recommendation::NeedsReview);
        assert_eq!(low_result.recommendation, Recommendation::Approve);
    }
}
