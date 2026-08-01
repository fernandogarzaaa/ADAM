//! Epistemic state: beliefs with confidence, evidence, and provenance.
//!
//! A [`Belief`] is never overwritten in place — new evidence adjusts its
//! confidence via an explicit update rule, and when two beliefs compete
//! (contradict each other), the loser is `Retracted`/`Superseded` rather
//! than deleted, preserving the full epistemic history.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

pub type BeliefId = Uuid;

/// Where a piece of evidence came from — mirrors the provenance pattern
/// used by `adam-memory`, so beliefs can be traced back to the experiences
/// or reasoning that produced them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceOrigin {
    Observation,
    Memory,
    Reasoning,
    ExternalSource,
    UserAssertion,
}

/// One observation that pushes a belief's confidence up (`supports = true`)
/// or down (`supports = false`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    pub origin: EvidenceOrigin,
    pub description: String,
    pub supports: bool,
    /// How strongly this evidence should move confidence, in `[0, 1]`.
    pub weight: f32,
    pub recorded_at: DateTime<Utc>,
}

/// Lifecycle state of a belief.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BeliefStatus {
    /// Currently held and available to influence behavior.
    Active,
    /// Explicitly withdrawn (e.g. evidence dropped confidence to zero).
    Retracted,
    /// Lost a competition against a higher-confidence contradicting belief.
    Superseded { by: BeliefId },
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum BeliefError {
    #[error("belief {0} is not active and cannot be updated")]
    NotActive(BeliefId),
    #[error("evidence weight {0} out of range [0, 1]")]
    WeightOutOfRange(String),
    #[error("cannot resolve competition: belief {0} does not contradict {1}")]
    NotContradicting(BeliefId, BeliefId),
}

/// A single epistemic claim held by the organism, together with the
/// evidence trail that justifies its current confidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Belief {
    pub id: BeliefId,
    pub statement: String,
    pub confidence: f32,
    pub evidence: Vec<Evidence>,
    pub contradicts: Vec<BeliefId>,
    pub status: BeliefStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Belief {
    /// Form a new belief from an initial piece of evidence. Confidence
    /// starts at the evidence's weight.
    pub fn form(
        statement: impl Into<String>,
        origin: EvidenceOrigin,
        description: impl Into<String>,
        weight: f32,
    ) -> Result<Self, BeliefError> {
        let weight = clamp_weight(weight)?;
        let now = Utc::now();
        let evidence = Evidence {
            origin,
            description: description.into(),
            supports: true,
            weight,
            recorded_at: now,
        };
        Ok(Self {
            id: Uuid::new_v4(),
            statement: statement.into(),
            confidence: weight,
            evidence: vec![evidence],
            contradicts: Vec::new(),
            status: BeliefStatus::Active,
            created_at: now,
            updated_at: now,
        })
    }

    /// Fold in one more piece of evidence, adjusting confidence with a
    /// simple bounded update rule: supporting evidence moves confidence
    /// toward 1 proportionally to `weight`, contradicting evidence moves
    /// it toward 0. A belief whose confidence reaches zero is retracted.
    pub fn add_evidence(
        &mut self,
        origin: EvidenceOrigin,
        description: impl Into<String>,
        supports: bool,
        weight: f32,
    ) -> Result<(), BeliefError> {
        if self.status != BeliefStatus::Active {
            return Err(BeliefError::NotActive(self.id));
        }
        let weight = clamp_weight(weight)?;
        let now = Utc::now();

        self.confidence = if supports {
            self.confidence + weight * (1.0 - self.confidence)
        } else {
            self.confidence - weight * self.confidence
        };
        self.confidence = self.confidence.clamp(0.0, 1.0);

        self.evidence.push(Evidence {
            origin,
            description: description.into(),
            supports,
            weight,
            recorded_at: now,
        });
        self.updated_at = now;

        if self.confidence <= f32::EPSILON {
            self.status = BeliefStatus::Retracted;
        }
        Ok(())
    }

    /// Record that this belief contradicts another (mutual bookkeeping is
    /// the caller's responsibility via `BeliefRegistry::mark_contradicting`).
    pub fn mark_contradicts(&mut self, other: BeliefId) {
        if !self.contradicts.contains(&other) {
            self.contradicts.push(other);
        }
    }

    /// Resolve a competition against a contradicting belief: whichever has
    /// higher confidence stays `Active`; the other becomes `Superseded`.
    /// Returns `true` if `self` won.
    pub fn resolve_against(&mut self, other: &mut Belief) -> Result<bool, BeliefError> {
        if !self.contradicts.contains(&other.id) && !other.contradicts.contains(&self.id) {
            return Err(BeliefError::NotContradicting(self.id, other.id));
        }
        let now = Utc::now();
        if self.confidence >= other.confidence {
            other.status = BeliefStatus::Superseded { by: self.id };
            other.updated_at = now;
            Ok(true)
        } else {
            self.status = BeliefStatus::Superseded { by: other.id };
            self.updated_at = now;
            Ok(false)
        }
    }

    pub fn is_active(&self) -> bool {
        self.status == BeliefStatus::Active
    }
}

fn clamp_weight(weight: f32) -> Result<f32, BeliefError> {
    if !(0.0..=1.0).contains(&weight) {
        return Err(BeliefError::WeightOutOfRange(format!("{weight}")));
    }
    Ok(weight)
}
