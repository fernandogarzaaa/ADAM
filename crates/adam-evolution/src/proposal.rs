//! Evolution proposals: the only artifact the evolution engine produces.
//!
//! A proposal is never self-executing. It sits in `Proposed` until an
//! external governance decision (Phase 8) explicitly `accept`s or
//! `reject`s it — the engine that analyzes history is deliberately kept
//! separate from the authority to change the organism.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

pub type ProposalId = Uuid;

/// The category of change a proposal recommends.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ProposalKind {
    /// Retire or force re-evolution of a chronically failing skill.
    RetireSkill { skill_name: String },
    /// Re-examine a belief whose confidence keeps collapsing.
    ReconcileBelief { statement: String },
    /// Investigate a systemic, recurring point of contradiction.
    InvestigateConflict { topic: String },
    /// Amend a specific genome field to a new suggested value.
    AmendGenome {
        field: String,
        current_value: String,
        suggested_value: String,
    },
}

/// Lifecycle of a proposal. Set only by explicit governance action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalStatus {
    Proposed,
    Accepted,
    Rejected,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ProposalError {
    #[error("proposal {0} has already been decided ({1:?}) and cannot be re-decided")]
    AlreadyDecided(ProposalId, ProposalStatus),
}

/// A single, auditable recommendation to change some part of the
/// organism, backed by the evidence that triggered it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvolutionProposal {
    pub id: ProposalId,
    pub kind: ProposalKind,
    pub rationale: String,
    pub evidence: Vec<String>,
    /// How strongly the engine recommends this, in `[0, 1]`.
    pub confidence: f32,
    pub status: ProposalStatus,
    pub created_at: DateTime<Utc>,
    pub decided_at: Option<DateTime<Utc>>,
}

impl EvolutionProposal {
    pub fn new(
        kind: ProposalKind,
        rationale: impl Into<String>,
        evidence: Vec<String>,
        confidence: f32,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            kind,
            rationale: rationale.into(),
            evidence,
            confidence: confidence.clamp(0.0, 1.0),
            status: ProposalStatus::Proposed,
            created_at: Utc::now(),
            decided_at: None,
        }
    }

    /// Explicitly accept the proposal. Does not apply it — applying a
    /// mutation is the responsibility of the governance/safety layer that
    /// owns the genome, skill registry, or belief registry being changed.
    pub fn accept(&mut self) -> Result<(), ProposalError> {
        self.decide(ProposalStatus::Accepted)
    }

    pub fn reject(&mut self) -> Result<(), ProposalError> {
        self.decide(ProposalStatus::Rejected)
    }

    fn decide(&mut self, status: ProposalStatus) -> Result<(), ProposalError> {
        if self.status != ProposalStatus::Proposed {
            return Err(ProposalError::AlreadyDecided(self.id, self.status));
        }
        self.status = status;
        self.decided_at = Some(Utc::now());
        Ok(())
    }
}
