//! Append-only audit log: every governance-relevant action the organism
//! takes (accepting/rejecting a mutation, rolling back the genome) is
//! recorded here and never removed or rewritten.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type AuditEntryId = Uuid;

/// A governance-relevant action worth recording permanently.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "action")]
pub enum AuditAction {
    ProposalAccepted {
        proposal_id: Uuid,
        effect_summary: String,
    },
    ProposalRejected {
        proposal_id: Uuid,
    },
    RollbackPerformed {
        target: Uuid,
        new_version: Uuid,
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditEntry {
    pub id: AuditEntryId,
    pub recorded_at: DateTime<Utc>,
    pub action: AuditAction,
}

/// Append-only log. There is deliberately no method to remove or modify an
/// entry once recorded — the audit trail is the organism's only record of
/// what it has done to itself.
#[derive(Debug, Default)]
pub struct AuditLog {
    entries: Vec<AuditEntry>,
}

impl AuditLog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, action: AuditAction) -> AuditEntryId {
        let id = Uuid::new_v4();
        self.entries.push(AuditEntry {
            id,
            recorded_at: Utc::now(),
            action,
        });
        id
    }

    pub fn entries(&self) -> &[AuditEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
