//! Where the organism meets something that is not itself.
//!
//! Two halves, deliberately separate:
//!
//! - [`workspace`] is the world — a directory of records with its own state,
//!   its own failure modes, and its own measure of how much work is done. It
//!   knows nothing about ADAM.
//! - [`connector`] is the boundary — it chooses an action from the organism's
//!   genome, performs it, and records what came back as an ADAM memory. It is
//!   the only part that knows about both.
//!
//! Keeping them apart is what makes the experiment falsifiable. If the
//! environment could see the organism, an improvement could come from the
//! environment being accommodating rather than from the organism behaving
//! differently, and no measurement could tell those apart.

pub mod connector;
pub mod fitness;
pub mod workspace;

pub use connector::{act_and_record, action_for, ConnectorError, Observed, ORIGIN, VERIFY_POLICY};
// Aliased because `connector::ORIGIN` already owns the bare name at the crate
// root, and two different provenance origins sharing one name at a call site is
// exactly how a document ends up claiming it came from somewhere it did not.
pub use fitness::{RealTaskFitnessProvider, ORIGIN as MEASURE_ORIGIN};
pub use workspace::{Action, ActionOutcome, Workspace, WorkspaceError, WorkspaceState};
