//! Threshold-driven analysis that turns signals into proposals.
//!
//! The rules here are intentionally simple and explainable — each rule
//! traces directly from a threshold to a proposal kind, so a human
//! reviewing `EvolutionProposal::rationale` can verify the engine's
//! reasoning without trusting a black box.

use crate::proposal::{EvolutionProposal, ProposalKind};
use crate::signal::EvolutionSignals;

/// Configurable thresholds controlling when a signal becomes a proposal.
#[derive(Debug, Clone)]
pub struct EvolutionThresholds {
    pub skill_fitness_floor: f32,
    pub skill_failure_count_floor: u32,
    pub belief_retraction_floor: u32,
    pub conflict_occurrence_floor: u32,
    pub genome_drift_observation_floor: u32,
}

impl Default for EvolutionThresholds {
    fn default() -> Self {
        Self {
            skill_fitness_floor: 0.4,
            skill_failure_count_floor: 3,
            belief_retraction_floor: 2,
            conflict_occurrence_floor: 3,
            genome_drift_observation_floor: 5,
        }
    }
}

/// Stateless analyzer: consumes a snapshot of signals, produces proposals.
/// It never touches genome, skill, memory, or belief state directly.
pub struct EvolutionEngine {
    thresholds: EvolutionThresholds,
}

impl EvolutionEngine {
    pub fn new(thresholds: EvolutionThresholds) -> Self {
        Self { thresholds }
    }

    pub fn analyze(&self, signals: &EvolutionSignals) -> Vec<EvolutionProposal> {
        let mut proposals = Vec::new();

        for signal in &signals.skill_failures {
            if signal.fitness_score < self.thresholds.skill_fitness_floor
                && signal.failure_count >= self.thresholds.skill_failure_count_floor
            {
                proposals.push(EvolutionProposal::new(
                    ProposalKind::RetireSkill {
                        skill_name: signal.skill_name.clone(),
                    },
                    format!(
                        "skill '{}' has fitness {:.2} (below floor {:.2}) across {} failures",
                        signal.skill_name,
                        signal.fitness_score,
                        self.thresholds.skill_fitness_floor,
                        signal.failure_count
                    ),
                    signal.failures.clone(),
                    1.0 - signal.fitness_score,
                ));
            }
        }

        for signal in &signals.belief_instabilities {
            if signal.retraction_count >= self.thresholds.belief_retraction_floor {
                proposals.push(EvolutionProposal::new(
                    ProposalKind::ReconcileBelief {
                        statement: signal.statement.clone(),
                    },
                    format!(
                        "belief '{}' has been retracted/superseded {} times (confidence now {:.2})",
                        signal.statement, signal.retraction_count, signal.confidence
                    ),
                    vec![format!("current confidence: {:.2}", signal.confidence)],
                    (signal.retraction_count as f32 / (signal.retraction_count as f32 + 1.0))
                        .clamp(0.0, 1.0),
                ));
            }
        }

        for signal in &signals.recurring_conflicts {
            if signal.occurrences >= self.thresholds.conflict_occurrence_floor {
                proposals.push(EvolutionProposal::new(
                    ProposalKind::InvestigateConflict {
                        topic: signal.topic.clone(),
                    },
                    format!(
                        "topic '{}' has produced {} recurring conflicts",
                        signal.topic, signal.occurrences
                    ),
                    vec![format!("occurrences: {}", signal.occurrences)],
                    (signal.occurrences as f32 / (signal.occurrences as f32 + 2.0)).clamp(0.0, 1.0),
                ));
            }
        }

        for signal in &signals.genome_drifts {
            if signal.supporting_observations >= self.thresholds.genome_drift_observation_floor {
                proposals.push(EvolutionProposal::new(
                    ProposalKind::AmendGenome {
                        field: signal.field.clone(),
                        current_value: signal.current_value.clone(),
                        suggested_value: signal.suggested_value.clone(),
                    },
                    format!(
                        "genome field '{}' has {} supporting observations suggesting '{}' over '{}'",
                        signal.field,
                        signal.supporting_observations,
                        signal.suggested_value,
                        signal.current_value
                    ),
                    vec![format!(
                        "supporting observations: {}",
                        signal.supporting_observations
                    )],
                    (signal.supporting_observations as f32 / 10.0).clamp(0.0, 1.0),
                ));
            }
        }

        proposals
    }
}
