//! CP/1 events — the organism's nervous system.
//!
//! Subsystems announce facts; nothing calls anything directly. Before this,
//! ADAM's only observable trace was the governance audit log, which recorded
//! acceptances, rejections and rollbacks — three of the fifteen things worth
//! knowing. Everything else the organism did happened silently.
//!
//! An event names the canonical document it concerns (`subject_id` and
//! `subject_type`) and carries only enough payload to be readable in a log; the
//! full document is fetched by id when detail is needed. Keeping payloads to
//! scalars is what lets an event log stay cheap enough to always be on.
//!
//! The set is closed: consumers switch exhaustively over [`EventKind`], so
//! adding one is a CP/1 version change.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::canonical::{self, CanonicalError};
use crate::types::{Component, Provenance, Timestamp};

/// Every event the organism can emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EventKind {
    /// EVE perceived an environment.
    ObservationRecorded,
    /// EVE situated an observation in goal, action and outcome.
    ExperienceCreated,
    /// AXIOM compressed a working set.
    ContextCompressed,
    /// AXIOM could not support a claim from the supplied evidence.
    GroundingFailed,
    /// ADAM distilled experiences into a durable memory.
    MemoryConsolidated,
    /// ADAM formed, reinforced, weakened or retracted a belief.
    BeliefUpdated,
    /// ADAM promoted a skill.
    SkillLearned,
    /// ADAM produced a self-assessment across subsystems.
    ReflectionCompleted,
    /// ADAM proposed a change to genome, skills or beliefs.
    MutationProposed,
    /// EVE finished a deterministic scenario run.
    SimulationCompleted,
    /// The real-task evaluator finished a run against the actual workspace.
    ///
    /// The counterpart of `SimulationCompleted`, and separate from it for the
    /// same reason `Component::Pcr` is separate from `Component::Eve`: one
    /// announces that a simulation happened, the other that real work happened.
    /// A measured `FitnessResult` chains back to whichever produced its runs.
    TaskRunCompleted,
    /// An evaluator scored a mutation against baseline and candidate runs.
    FitnessMeasured,
    /// ADAM applied a proposal that passed governance.
    MutationAccepted,
    /// ADAM refused a proposal.
    MutationRejected,
    /// ADAM appended a new immutable genome version.
    GenomeCommitted,
}

impl EventKind {
    pub fn as_str(self) -> &'static str {
        match self {
            EventKind::ObservationRecorded => "ObservationRecorded",
            EventKind::ExperienceCreated => "ExperienceCreated",
            EventKind::ContextCompressed => "ContextCompressed",
            EventKind::GroundingFailed => "GroundingFailed",
            EventKind::MemoryConsolidated => "MemoryConsolidated",
            EventKind::BeliefUpdated => "BeliefUpdated",
            EventKind::SkillLearned => "SkillLearned",
            EventKind::ReflectionCompleted => "ReflectionCompleted",
            EventKind::MutationProposed => "MutationProposed",
            EventKind::SimulationCompleted => "SimulationCompleted",
            EventKind::TaskRunCompleted => "TaskRunCompleted",
            EventKind::FitnessMeasured => "FitnessMeasured",
            EventKind::MutationAccepted => "MutationAccepted",
            EventKind::MutationRejected => "MutationRejected",
            EventKind::GenomeCommitted => "GenomeCommitted",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|k| k.as_str() == value)
    }

    /// The components permitted to emit this event.
    ///
    /// Ownership of an event follows ownership of the concept it announces, so
    /// this is checkable: an `ObservationRecorded` from ADAM would mean ADAM
    /// minted an EVE-owned fact.
    ///
    /// Thirteen of the fifteen kinds have exactly one owner. `FitnessMeasured`
    /// has two, because two independent evaluators can honestly score a
    /// mutation — EVE against simulated experience, PCR against the real
    /// objective — and a function that answered with one `Component` could only
    /// do so by naming the wrong one for the other. The permission is still
    /// closed: two named components, not "anyone".
    pub fn emitters(self) -> &'static [Component] {
        match self {
            EventKind::ObservationRecorded
            | EventKind::ExperienceCreated
            | EventKind::SimulationCompleted => &[Component::Eve],
            EventKind::TaskRunCompleted => &[Component::Pcr],
            EventKind::FitnessMeasured => &[Component::Eve, Component::Pcr],
            EventKind::ContextCompressed | EventKind::GroundingFailed => &[Component::Axiom],
            EventKind::MemoryConsolidated
            | EventKind::BeliefUpdated
            | EventKind::SkillLearned
            | EventKind::ReflectionCompleted
            | EventKind::MutationProposed
            | EventKind::MutationAccepted
            | EventKind::MutationRejected
            | EventKind::GenomeCommitted => &[Component::Adam],
        }
    }

    /// Whether `actor` may emit this kind.
    pub fn permits(self, actor: Component) -> bool {
        self.emitters().contains(&actor)
    }

    pub const ALL: [EventKind; 15] = [
        EventKind::ObservationRecorded,
        EventKind::ExperienceCreated,
        EventKind::ContextCompressed,
        EventKind::GroundingFailed,
        EventKind::MemoryConsolidated,
        EventKind::BeliefUpdated,
        EventKind::SkillLearned,
        EventKind::ReflectionCompleted,
        EventKind::MutationProposed,
        EventKind::SimulationCompleted,
        EventKind::TaskRunCompleted,
        EventKind::FitnessMeasured,
        EventKind::MutationAccepted,
        EventKind::MutationRejected,
        EventKind::GenomeCommitted,
    ];
}

/// The canonical type an event is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubjectType {
    Identity,
    Genome,
    Capability,
    Belief,
    Memory,
    Skill,
    Mutation,
    Reflection,
    Observation,
    Experience,
    FitnessResult,
    Context,
}

/// A payload member. Scalars only, by design.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PayloadValue {
    Bool(bool),
    Int(i64),
    Text(String),
}

impl From<bool> for PayloadValue {
    fn from(v: bool) -> Self {
        PayloadValue::Bool(v)
    }
}
impl From<i64> for PayloadValue {
    fn from(v: i64) -> Self {
        PayloadValue::Int(v)
    }
}
impl From<u32> for PayloadValue {
    fn from(v: u32) -> Self {
        PayloadValue::Int(i64::from(v))
    }
}
impl From<usize> for PayloadValue {
    fn from(v: usize) -> Self {
        PayloadValue::Int(v as i64)
    }
}
impl From<&str> for PayloadValue {
    fn from(v: &str) -> Self {
        PayloadValue::Text(v.to_string())
    }
}
impl From<String> for PayloadValue {
    fn from(v: String) -> Self {
        PayloadValue::Text(v)
    }
}

/// One announced fact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Event {
    pub cp: String,
    #[serde(rename = "type")]
    pub kind: EventKind,
    pub id: String,
    pub occurred_at: Timestamp,
    pub actor: Component,
    pub subject_id: String,
    pub subject_type: SubjectType,
    /// Shared by every event of one developmental turn, which is what makes a
    /// full Observe-through-Commit cycle reconstructible from the log alone.
    pub correlation_id: String,
    /// The event that caused this one. With `correlation_id`, a turn becomes a
    /// tree rather than a bag.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub causation_id: Option<String>,
    /// `BTreeMap` so serialization is already key-sorted, matching canonical
    /// form without a second pass.
    pub payload: BTreeMap<String, PayloadValue>,
    pub provenance: Provenance,
}

impl Event {
    /// Build an event.
    ///
    /// `actor` is supplied rather than derived, because `FitnessMeasured` has
    /// two legitimate emitters and only the caller knows which one it is. It is
    /// checked against [`EventKind::emitters`], so the guarantee that an event
    /// cannot claim an authorship it is not entitled to is unchanged — it moved
    /// from "impossible to express" to "rejected when expressed".
    ///
    /// # Panics
    ///
    /// If `actor` is not permitted to emit `kind`. That is a programming error
    /// in the emitting subsystem, not a malformed input: every call site names
    /// a literal kind, and an event log that silently recorded the wrong actor
    /// would corrupt every audit built on it afterwards.
    pub fn new(
        actor: Component,
        kind: EventKind,
        subject_id: impl Into<String>,
        subject_type: SubjectType,
        correlation_id: impl Into<String>,
        payload: BTreeMap<String, PayloadValue>,
        origin: impl Into<String>,
    ) -> Self {
        assert!(
            kind.permits(actor),
            "{} may not emit {}; permitted: {:?}",
            actor.as_str(),
            kind.as_str(),
            kind.emitters()
        );
        Self {
            cp: crate::CP.to_string(),
            kind,
            id: uuid::Uuid::new_v4().to_string(),
            occurred_at: Timestamp::now(),
            actor,
            subject_id: subject_id.into(),
            subject_type,
            correlation_id: correlation_id.into(),
            causation_id: None,
            payload,
            provenance: Provenance::now(actor, origin),
        }
    }

    /// Record which event triggered this one.
    pub fn caused_by(mut self, event_id: impl Into<String>) -> Self {
        self.causation_id = Some(event_id.into());
        self
    }

    /// Record the documents this event's subject was derived from.
    pub fn derived_from(mut self, ids: impl IntoIterator<Item = String>) -> Self {
        self.provenance.derived_from.extend(ids);
        self
    }

    pub fn with_evidence(mut self, evidence: impl IntoIterator<Item = String>) -> Self {
        self.provenance.evidence.extend(evidence);
        self
    }

    /// Serialize to a sealed `serde_json::Value` ready for transport.
    pub fn seal(&self) -> Result<serde_json::Value, CanonicalError> {
        let mut value = serde_json::to_value(self).expect("Event always serializes");
        canonical::seal(&mut value)?;
        Ok(value)
    }
}

/// Anything that accepts emitted events.
///
/// A trait rather than a concrete bus, so an emitting subsystem never learns
/// where its events go — the property that keeps the event system a nervous
/// system rather than another call graph.
pub trait EventSink: Send + Sync {
    fn emit(&self, event: &Event);
}

/// An [`EventSink`] that discards everything.
///
/// The default for an organism nobody has wired a sink to. Making the sink
/// mandatory would force every caller and every test to supply one; making it
/// optional would put an `if let Some(..)` around every emission site.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullSink;

impl EventSink for NullSink {
    fn emit(&self, _event: &Event) {}
}

/// An [`EventSink`] that retains events in memory, for tests and for
/// short-lived processes that report their event log on exit.
#[derive(Debug, Default)]
pub struct RecordingSink {
    events: std::sync::Mutex<Vec<Event>>,
}

impl RecordingSink {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn events(&self) -> Vec<Event> {
        self.events.lock().expect("sink mutex poisoned").clone()
    }

    pub fn kinds(&self) -> Vec<EventKind> {
        self.events().iter().map(|e| e.kind).collect()
    }

    pub fn len(&self) -> usize {
        self.events.lock().expect("sink mutex poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Every event sharing `correlation_id`, in emission order — one full
    /// developmental turn.
    pub fn turn(&self, correlation_id: &str) -> Vec<Event> {
        self.events()
            .into_iter()
            .filter(|e| e.correlation_id == correlation_id)
            .collect()
    }
}

impl EventSink for RecordingSink {
    fn emit(&self, event: &Event) {
        self.events
            .lock()
            .expect("sink mutex poisoned")
            .push(event.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(pairs: &[(&str, PayloadValue)]) -> BTreeMap<String, PayloadValue> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    fn event(kind: EventKind, correlation: &str) -> Event {
        Event::new(
            kind.emitters()[0],
            kind,
            "33333333-3333-4333-8333-333333333333",
            SubjectType::Genome,
            correlation,
            payload(&[("version_label", PayloadValue::Text("1.1".into()))]),
            "adam:test",
        )
    }

    #[test]
    fn every_event_name_round_trips() {
        for kind in EventKind::ALL {
            assert_eq!(EventKind::parse(kind.as_str()), Some(kind));
        }
    }

    #[test]
    fn event_names_serialize_as_bare_strings() {
        assert_eq!(
            serde_json::to_string(&EventKind::GenomeCommitted).unwrap(),
            "\"GenomeCommitted\""
        );
    }

    #[test]
    fn unknown_event_names_are_rejected() {
        assert_eq!(EventKind::parse("GenomeDeleted"), None);
    }

    #[test]
    fn actor_is_checked_against_the_kind_and_covers_all_four_components() {
        assert_eq!(
            event(EventKind::GenomeCommitted, "c").actor,
            Component::Adam
        );
        let emitters: std::collections::BTreeSet<&str> = EventKind::ALL
            .iter()
            .flat_map(|k| k.emitters())
            .map(|c| c.as_str())
            .collect();
        assert_eq!(
            emitters,
            ["adam", "axiom", "eve", "pcr"]
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>()
        );
    }

    #[test]
    fn fitness_may_be_measured_by_either_evaluator_and_by_nobody_else() {
        assert!(EventKind::FitnessMeasured.permits(Component::Eve));
        assert!(EventKind::FitnessMeasured.permits(Component::Pcr));
        assert!(!EventKind::FitnessMeasured.permits(Component::Adam));
        assert!(!EventKind::FitnessMeasured.permits(Component::Axiom));
    }

    #[test]
    fn a_run_event_belongs_to_exactly_one_evaluator() {
        // The two run events are what a measurement chains back to, so
        // confusing them would let a simulated run stand as evidence that real
        // work happened.
        assert_eq!(
            EventKind::SimulationCompleted.emitters(),
            &[Component::Eve]
        );
        assert_eq!(EventKind::TaskRunCompleted.emitters(), &[Component::Pcr]);
    }

    #[test]
    #[should_panic(expected = "may not emit")]
    fn an_event_cannot_claim_an_authorship_it_is_not_entitled_to() {
        Event::new(
            Component::Adam,
            EventKind::FitnessMeasured,
            "33333333-3333-4333-8333-333333333333",
            SubjectType::Mutation,
            "c",
            BTreeMap::new(),
            "adam:test",
        );
    }

    #[test]
    fn a_sealed_event_verifies_and_omits_absent_causation() {
        let sealed = event(EventKind::GenomeCommitted, "c").seal().unwrap();
        assert!(canonical::verify_seal(&sealed).unwrap());
        assert!(
            sealed.get("causation_id").is_none(),
            "an absent causation must be an absent key, never a null"
        );
    }

    #[test]
    fn causation_and_derivation_are_recorded_when_set() {
        let sealed = event(EventKind::GenomeCommitted, "c")
            .caused_by("1f1f1f1f-1f1f-4f1f-8f1f-1f1f1f1f1f1f")
            .derived_from(["cccccccc-cccc-4ccc-8ccc-cccccccccccc".to_string()])
            .seal()
            .unwrap();
        assert_eq!(
            sealed["causation_id"].as_str(),
            Some("1f1f1f1f-1f1f-4f1f-8f1f-1f1f1f1f1f1f")
        );
        assert_eq!(
            sealed["provenance"]["derived_from"][0].as_str(),
            Some("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
        );
    }

    #[test]
    fn recording_sink_preserves_emission_order() {
        let sink = RecordingSink::new();
        for kind in [EventKind::MutationProposed, EventKind::GenomeCommitted] {
            sink.emit(&event(kind, "c"));
        }
        assert_eq!(
            sink.kinds(),
            vec![EventKind::MutationProposed, EventKind::GenomeCommitted]
        );
    }

    #[test]
    fn a_turn_is_reconstructible_from_its_correlation_id() {
        let sink = RecordingSink::new();
        sink.emit(&event(EventKind::MutationProposed, "turn-a"));
        sink.emit(&event(EventKind::MutationProposed, "turn-b"));
        sink.emit(&event(EventKind::GenomeCommitted, "turn-a"));

        assert_eq!(sink.turn("turn-a").len(), 2);
        assert_eq!(sink.turn("turn-b").len(), 1);
        assert_eq!(sink.len(), 3);
    }

    #[test]
    fn the_null_sink_accepts_everything_and_keeps_nothing() {
        let sink = NullSink;
        sink.emit(&event(EventKind::GenomeCommitted, "c"));
        // Nothing to assert but that it did not panic — which is the contract.
    }
}
