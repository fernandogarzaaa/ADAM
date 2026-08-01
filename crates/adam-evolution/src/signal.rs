//! Input signals the evolution engine analyzes.
//!
//! These are intentionally decoupled from `adam-skills`/`adam-beliefs`/
//! `adam-memory` types: the calling orchestrator (the future MCP layer)
//! translates real skill/belief/memory state into these lightweight
//! signals, keeping the evolution engine testable in isolation and free
//! of a dependency cycle back onto the crates whose history it studies.

use serde::{Deserialize, Serialize};

/// A skill that has repeatedly underperformed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillFailureSignal {
    pub skill_name: String,
    pub fitness_score: f32,
    pub failure_count: u32,
    pub failures: Vec<String>,
}

/// A belief that keeps losing confidence or competitions on the same topic.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeliefInstabilitySignal {
    pub statement: String,
    pub confidence: f32,
    pub retraction_count: u32,
}

/// A memory-conflict pattern: the same kind of contradiction keeps
/// recurring, suggesting a systemic gap rather than one-off noise.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecurringConflictSignal {
    pub topic: String,
    pub occurrences: u32,
}

/// A genome capability or policy that evidence suggests is stale.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenomeDriftSignal {
    pub field: String,
    pub current_value: String,
    pub suggested_value: String,
    pub supporting_observations: u32,
}

/// All signals gathered for one analysis pass.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvolutionSignals {
    pub skill_failures: Vec<SkillFailureSignal>,
    pub belief_instabilities: Vec<BeliefInstabilitySignal>,
    pub recurring_conflicts: Vec<RecurringConflictSignal>,
    pub genome_drifts: Vec<GenomeDriftSignal>,
}
