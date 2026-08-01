//! EVE ("Evaluate Via Experiment") integration: the adapter that scores an
//! [`EvolutionProposal`] before it reaches governance, by running it
//! through sandboxed trials rather than trusting the proposal's own
//! self-reported confidence.

use adam_evolution::{EvolutionProposal, ProposalId};
use serde::{Deserialize, Serialize};

/// The outcome of a single sandboxed trial run against a proposal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrialOutcome {
    pub succeeded: bool,
    pub detail: String,
}

/// What EVE recommends doing with a proposal, independent of whatever
/// governance ultimately decides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Recommendation {
    Approve,
    NeedsReview,
    Reject,
}

/// A fitness-scored evaluation of one proposal, backed by real trial
/// evidence rather than the proposal's own confidence field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationResult {
    pub proposal_id: ProposalId,
    /// Fraction of trials that succeeded, in `[0, 1]`.
    pub fitness: f32,
    /// How much this evaluation should be discounted for risk (destructive
    /// proposal kinds carry higher risk even at high fitness).
    pub risk: f32,
    pub recommendation: Recommendation,
    pub trials: Vec<TrialOutcome>,
}

/// A function that runs one sandboxed trial of a proposal and reports
/// whether it succeeded. Callers supply this — it is the actual
/// simulation/test harness (unit test sandbox, staging replay, etc.);
/// this crate owns the aggregation and scoring policy, not the trial
/// mechanics themselves.
pub type TrialFn<'a> = dyn Fn(&EvolutionProposal) -> TrialOutcome + 'a;

/// Thresholds controlling how fitness + risk map to a recommendation.
#[derive(Debug, Clone)]
pub struct EvaluationThresholds {
    pub approve_fitness_floor: f32,
    pub reject_fitness_ceiling: f32,
    pub max_acceptable_risk: f32,
}

impl Default for EvaluationThresholds {
    fn default() -> Self {
        Self {
            approve_fitness_floor: 0.75,
            reject_fitness_ceiling: 0.3,
            max_acceptable_risk: 0.6,
        }
    }
}

/// Runs a proposal through `trial_count` sandboxed trials via the supplied
/// `trial_fn`, aggregates the pass rate into a fitness score, and derives
/// a risk-adjusted recommendation.
pub struct SimulationEvaluator {
    thresholds: EvaluationThresholds,
    trial_count: u32,
}

impl SimulationEvaluator {
    pub fn new(thresholds: EvaluationThresholds, trial_count: u32) -> Self {
        Self {
            thresholds,
            trial_count: trial_count.max(1),
        }
    }

    pub fn evaluate(&self, proposal: &EvolutionProposal, trial_fn: &TrialFn) -> EvaluationResult {
        let trials: Vec<TrialOutcome> = (0..self.trial_count).map(|_| trial_fn(proposal)).collect();
        let passed = trials.iter().filter(|t| t.succeeded).count() as f32;
        let fitness = passed / trials.len() as f32;
        let risk = risk_for(proposal);

        let recommendation = if risk > self.thresholds.max_acceptable_risk {
            Recommendation::NeedsReview
        } else if fitness >= self.thresholds.approve_fitness_floor {
            Recommendation::Approve
        } else if fitness <= self.thresholds.reject_fitness_ceiling {
            Recommendation::Reject
        } else {
            Recommendation::NeedsReview
        };

        EvaluationResult {
            proposal_id: proposal.id,
            fitness,
            risk,
            recommendation,
            trials,
        }
    }
}

/// Intrinsic risk of a proposal kind, independent of trial outcomes.
/// Genome amendments carry the highest baseline risk since they touch
/// core identity; skill retirement is lowest since it only narrows
/// available behavior.
fn risk_for(proposal: &EvolutionProposal) -> f32 {
    use adam_evolution::ProposalKind::*;
    let base = match &proposal.kind {
        RetireSkill { .. } => 0.1,
        ReconcileBelief { .. } => 0.3,
        InvestigateConflict { .. } => 0.2,
        AmendGenome { .. } => 0.6,
    };
    (base + (1.0 - proposal.confidence) * 0.2).clamp(0.0, 1.0)
}
