//! ADAM Safety & Governance.
//!
//! Every mutation acceptance, rejection, and rollback passes through a
//! single [`GovernanceGate`] that enforces evolution rate limits and
//! writes an immutable audit trail — the organism cannot change itself
//! without leaving a record, and cannot change itself arbitrarily fast.

mod audit;
mod gate;
mod limits;

pub use audit::{AuditAction, AuditEntry, AuditEntryId, AuditLog};
pub use gate::GovernanceGate;
pub use limits::{EvolutionLimits, GovernanceError, RateLimiter};
