//! Turning what happened in the environment into something ADAM can hold.
//!
//! # Why this is not an `ObservationRecorded` event
//!
//! CP/1 gives each canonical type exactly one author, and `Observation` —
//! along with the `ObservationRecorded` event — belongs to EVE
//! (`EventKind::emitter`). ADAM emitting one would be ADAM signing EVE's name,
//! which is the property the protocol exists to prevent. So an external fact
//! enters the organism the way any other fact does: as an ADAM episodic
//! memory, announced as `MemoryConsolidated`, authored by ADAM, carrying where
//! it came from.
//!
//! That is a constraint discovered rather than designed around, and it narrows
//! what Phase 0 can claim: the loop closes through ADAM's own memory, not
//! through EVE's perception, because EVE does not perceive anything yet.
//!
//! # Choosing what to do
//!
//! [`action_for`] reads the action out of the genome's policies. This is the
//! load-bearing detail of the whole experiment: if the action were chosen any
//! other way, an accepted mutation could not change behaviour, and a loop that
//! cannot change behaviour is a pipeline.

use adam_kernel::Genome;
use adam_memory::{MemoryId, MemoryKind};
use adam_organism::{Organism, OrganismError};
use thiserror::Error;

use crate::workspace::{Action, ActionOutcome, Workspace, WorkspaceError, WorkspaceState};

/// The policy substring that selects the careful action.
///
/// A substring rather than an exact string because a policy is written for a
/// person to read, and `"verify records before processing them"` should count.
pub const VERIFY_POLICY: &str = "verify";

/// Where observations from this connector say they came from.
pub const ORIGIN: &str = "pcr:workspace";

#[derive(Debug, Error)]
pub enum ConnectorError {
    #[error(transparent)]
    Workspace(#[from] WorkspaceError),
    #[error(transparent)]
    Organism(#[from] OrganismError),
    #[error("the organism announced no event for memory {0}")]
    Unannounced(String),
}

/// What the organism did, and how to find the record of it.
#[derive(Debug, Clone)]
pub struct Observed {
    pub memory_id: MemoryId,
    /// The `MemoryConsolidated` event id, so a later proposal can name this
    /// observation as its cause.
    pub event_id: String,
    /// The environment's objective after acting, in basis points.
    pub objective_bp: i32,
}

/// Choose what to do, from what the organism currently believes it should do.
///
/// An organism with no relevant policy acts naively. That is not a penalty
/// imposed on it — it is what having no policy means.
pub fn action_for(genome: &Genome) -> Action {
    let careful = genome
        .policies
        .iter()
        .any(|policy| policy.to_lowercase().contains(VERIFY_POLICY));
    if careful {
        Action::VerifyThenProcess
    } else {
        Action::ProcessAll
    }
}

/// State the environment's result in a sentence the organism can store.
///
/// Deliberately concrete: an observation reading "an error occurred" tells a
/// later reader nothing about why the organism changed, which is the question
/// the whole event log is for.
pub fn describe(state: &WorkspaceState, outcome: &ActionOutcome) -> String {
    let mut text = format!(
        "workspace: acted {}; processed {} of {}",
        outcome.action.as_str(),
        outcome.processed,
        state.total()
    );
    if outcome.quarantined > 0 {
        text.push_str(&format!("; set aside {}", outcome.quarantined));
    }
    match (&outcome.failed_on, &outcome.error) {
        (Some(name), Some(problem)) => {
            text.push_str(&format!("; stopped on {name}: {problem}"));
            text.push_str(&format!(
                "; {} records left unprocessed",
                state.pending.len()
            ));
        }
        _ => text.push_str("; completed without stopping"),
    }
    text
}

/// Act in the environment, then record what happened as an ADAM memory.
///
/// Observe *after* acting rather than before: the state the description is
/// written from is the one the organism now has to live with, and a
/// description taken from before would report a world that no longer exists.
pub fn act_and_record(
    organism: &Organism,
    workspace: &Workspace,
    correlation_id: &str,
) -> Result<Observed, ConnectorError> {
    let action = action_for(organism.genome());
    let outcome = workspace.act(action)?;
    let state = workspace.observe()?;
    let objective_bp = workspace.objective_bp()?;
    let content = describe(&state, &outcome);

    // Confidence 1.0 and decay 0.0: this is not a guess about the world, it is
    // a record of what the organism did and what came back. A first-hand
    // outcome that faded would make the organism forget its own experience.
    let memory_id = organism.consolidate_memory(
        MemoryKind::Episodic,
        &content,
        ORIGIN,
        vec![format!("objective_bp={objective_bp}")],
        1.0,
        0.0,
        correlation_id,
    )?;

    let event_id = organism
        .last_event(&memory_id.to_string())
        .ok_or_else(|| ConnectorError::Unannounced(memory_id.to_string()))?;

    Ok(Observed {
        memory_id,
        event_id,
        objective_bp,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_organism_with_no_policy_acts_naively() {
        let genome = Genome::new("probe", "no policies");
        assert_eq!(action_for(&genome), Action::ProcessAll);
    }

    #[test]
    fn a_verify_policy_selects_the_careful_action() {
        let mut genome = Genome::new("probe", "with a policy");
        genome
            .policies
            .push("verify records before processing them".to_string());
        assert_eq!(action_for(&genome), Action::VerifyThenProcess);
    }

    #[test]
    fn an_unrelated_policy_changes_nothing() {
        let mut genome = Genome::new("probe", "with a policy");
        genome
            .policies
            .push("prefer short explanations".to_string());
        assert_eq!(
            action_for(&genome),
            Action::ProcessAll,
            "only a policy about verifying may select the verifying action"
        );
    }

    #[test]
    fn a_failure_is_described_specifically_enough_to_act_on() {
        let state = WorkspaceState {
            pending: vec!["03-charlie.rec".into(), "04-delta.rec".into()],
            processed: vec!["01-alpha.rec".into(), "02-bravo.rec".into()],
            quarantined: vec![],
        };
        let outcome = ActionOutcome {
            action: Action::ProcessAll,
            processed: 2,
            quarantined: 0,
            failed_on: Some("03-charlie.rec".into()),
            error: Some("line 2 of 03-charlie.rec is not key=value".into()),
        };
        let text = describe(&state, &outcome);

        assert!(text.contains("process_all"));
        assert!(text.contains("processed 2 of 4"));
        assert!(text.contains("03-charlie.rec"));
        assert!(text.contains("not key=value"));
        assert!(text.contains("2 records left unprocessed"));
    }

    #[test]
    fn a_clean_run_says_so() {
        let state = WorkspaceState {
            pending: vec![],
            processed: vec!["01-alpha.rec".into()],
            quarantined: vec!["03-charlie.rec".into()],
        };
        let outcome = ActionOutcome {
            action: Action::VerifyThenProcess,
            processed: 1,
            quarantined: 1,
            failed_on: None,
            error: None,
        };
        let text = describe(&state, &outcome);
        assert!(text.contains("completed without stopping"));
        assert!(text.contains("set aside 1"));
    }
}
