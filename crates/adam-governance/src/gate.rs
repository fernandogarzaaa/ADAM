//! The governance gate: the single choke point every mutation acceptance,
//! rejection, and rollback must pass through. It enforces evolution rate
//! limits and writes every decision to the append-only audit log — no
//! path exists to change the organism's genome or skill set without
//! leaving a record here.

use uuid::Uuid;

use crate::audit::{AuditAction, AuditEntry, AuditLog};
use crate::limits::{EvolutionLimits, GovernanceError, RateLimiter};

pub struct GovernanceGate {
    audit: AuditLog,
    limiter: RateLimiter,
}

impl GovernanceGate {
    pub fn new(limits: EvolutionLimits) -> Self {
        Self {
            audit: AuditLog::new(),
            limiter: RateLimiter::new(limits),
        }
    }

    /// Call before applying an accepted proposal's effect. Returns an
    /// error (and records nothing) if the evolution rate limit would be
    /// exceeded.
    pub fn authorize_acceptance(&mut self) -> Result<(), GovernanceError> {
        self.limiter.check_and_record(chrono::Utc::now())
    }

    /// Call after successfully applying an accepted proposal's effect.
    pub fn log_acceptance(&mut self, proposal_id: Uuid, effect_summary: impl Into<String>) {
        self.audit.record(AuditAction::ProposalAccepted {
            proposal_id,
            effect_summary: effect_summary.into(),
        });
    }

    pub fn log_rejection(&mut self, proposal_id: Uuid) {
        self.audit.record(AuditAction::ProposalRejected { proposal_id });
    }

    pub fn log_rollback(&mut self, target: Uuid, new_version: Uuid, reason: impl Into<String>) {
        self.audit.record(AuditAction::RollbackPerformed {
            target,
            new_version,
            reason: reason.into(),
        });
    }

    pub fn audit_log(&self) -> &[AuditEntry] {
        self.audit.entries()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn acceptance_beyond_the_limit_is_refused_and_unlogged() {
        let mut gate = GovernanceGate::new(EvolutionLimits {
            max_accepted_per_window: 1,
            window: Duration::hours(1),
        });
        assert!(gate.authorize_acceptance().is_ok());
        gate.log_acceptance(Uuid::new_v4(), "skill retired");

        let err = gate.authorize_acceptance().unwrap_err();
        assert!(matches!(err, GovernanceError::LimitExceeded { .. }));
        assert_eq!(gate.audit_log().len(), 1);
    }

    #[test]
    fn every_decision_is_recorded_in_the_audit_log() {
        let mut gate = GovernanceGate::new(EvolutionLimits::default());
        let accepted_id = Uuid::new_v4();
        let rejected_id = Uuid::new_v4();
        let target = Uuid::new_v4();
        let new_version = Uuid::new_v4();

        gate.authorize_acceptance().unwrap();
        gate.log_acceptance(accepted_id, "genome amended");
        gate.log_rejection(rejected_id);
        gate.log_rollback(target, new_version, "regression");

        assert_eq!(gate.audit_log().len(), 3);
        assert!(matches!(
            gate.audit_log()[0].action,
            AuditAction::ProposalAccepted { .. }
        ));
        assert!(matches!(
            gate.audit_log()[1].action,
            AuditAction::ProposalRejected { .. }
        ));
        assert!(matches!(
            gate.audit_log()[2].action,
            AuditAction::RollbackPerformed { .. }
        ));
    }
}
