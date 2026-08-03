//! The developmental lifecycle: the organism's metabolism.
//!
//! Before this module, every stage of the lifecycle existed as a method
//! somewhere and nothing executed the sequence. `reflect` returned a summary
//! and consolidated nothing; there was no `observe`; consolidation from
//! repeated episodes into semantic memory was described in `adam-memory`'s
//! documentation and driven by nothing. The organism had organs and no
//! metabolism.
//!
//! [`LifecycleDriver`] executes the whole sequence, in order, with no
//! shortcuts:
//!
//! ```text
//! Observe → Experience → Reflect → Consolidate memory → Update beliefs
//!   → Generate mutations → Validate inside EVE → Measure fitness
//!   → Commit genome → continue operating
//! ```
//!
//! Every stage emits its CP/1 event, and every event of one turn shares a
//! correlation id — so a completed turn is reconstructible from the event log
//! alone, without joining against the organism's internal state.
//!
//! # What a turn does and does not do
//!
//! A turn never applies a change the organism cannot justify. Concretely:
//!
//! - A mutation touching the genome beyond `preferences.*` is validated in EVE
//!   and accepted only on an approving measurement. If EVE is unreachable, or
//!   declines to measure, or measures a regression, the proposal stays pending
//!   and the turn reports why.
//! - Advisory proposals (belief reconciliation, conflict investigation) are
//!   never auto-accepted, because accepting them applies nothing — they exist
//!   to be read by something that can act on them.
//! - A turn that proposes nothing is not a failure. Most turns should propose
//!   nothing; an organism that mutates on every observation is not learning,
//!   it is thrashing.

use adam_beliefs::Belief;
use adam_evolution::ProposalId;
use adam_memory::MemoryKind;
use adam_protocol::Recommendation;

use crate::organism::{
    new_correlation_id, AppliedEffect, CorrelationId, Organism, OrganismError, ReflectionSummary,
};

/// One thing the organism perceived, as it enters the lifecycle.
///
/// Deliberately minimal and stringly-typed. A rich observation type here would
/// duplicate EVE's `Observation`, which is EVE's to author — this is the shape
/// an ADAM-side caller (a tool result, a user statement, a build failure) can
/// actually produce.
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    /// What was perceived.
    pub content: String,
    /// Where it came from, e.g. `"tool:cargo_build"` or `"user:conversation"`.
    pub origin: String,
    /// Supporting detail: log excerpts, quoted text, references.
    pub evidence: Vec<String>,
    /// How much the organism should trust this, in `[0, 1]`.
    pub confidence: f32,
}

impl Observation {
    pub fn new(content: impl Into<String>, origin: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            origin: origin.into(),
            evidence: Vec::new(),
            confidence: 0.8,
        }
    }

    pub fn with_evidence(mut self, evidence: Vec<String>) -> Self {
        self.evidence = evidence;
        self
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

/// What one proposal's journey through validation and governance came to.
#[derive(Debug, Clone, PartialEq)]
pub enum MutationOutcome {
    /// Measured, approved, applied.
    Accepted {
        id: ProposalId,
        effect_label: String,
        delta_bp: i32,
    },
    /// Measured and refused, with EVE's reason.
    Rejected { id: ProposalId, reason: String },
    /// Left pending for a decision the organism is not entitled to make alone.
    Escalated { id: ProposalId, reason: String },
}

impl MutationOutcome {
    pub fn id(&self) -> ProposalId {
        match self {
            MutationOutcome::Accepted { id, .. }
            | MutationOutcome::Rejected { id, .. }
            | MutationOutcome::Escalated { id, .. } => *id,
        }
    }
}

/// Everything one developmental turn did.
#[derive(Debug, Clone)]
pub struct TurnReport {
    /// Groups every event this turn emitted.
    pub correlation_id: CorrelationId,
    /// Memories consolidated from the turn's observations.
    pub memories: Vec<adam_memory::MemoryId>,
    /// The self-assessment taken before mutations were generated.
    pub reflection: ReflectionSummary,
    /// Beliefs formed or reinforced this turn.
    pub beliefs: Vec<adam_beliefs::BeliefId>,
    /// Every proposal generated, and what became of it.
    pub mutations: Vec<MutationOutcome>,
}

impl TurnReport {
    /// Proposals that were applied.
    pub fn accepted(&self) -> impl Iterator<Item = &MutationOutcome> {
        self.mutations
            .iter()
            .filter(|m| matches!(m, MutationOutcome::Accepted { .. }))
    }

    /// Whether the turn changed the organism at all. Most turns should not.
    pub fn changed_anything(&self) -> bool {
        self.accepted().next().is_some()
    }
}

/// How a turn behaves.
#[derive(Debug, Clone)]
pub struct LifecycleConfig {
    /// Confidence at or above which an observation also becomes a belief.
    ///
    /// Not every observation is a claim about the world. A low-confidence
    /// observation is recorded as an episodic memory and left there; promoting
    /// everything to a belief would fill the registry with things the organism
    /// does not actually hold.
    pub belief_confidence_floor: f32,
    /// Decay rate applied to memories consolidated from observations.
    pub memory_decay_rate: f32,
    /// Whether a turn may accept mutations, or only propose and measure them.
    ///
    /// Setting this false gives a fully observable dry run: proposals are
    /// generated and measured in EVE, every event is emitted, and nothing is
    /// applied. That is the right mode for the first deployment of a new
    /// scenario suite, where the measurements themselves are what need review.
    pub apply_accepted_mutations: bool,
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        Self {
            belief_confidence_floor: 0.7,
            // Slow: a memory consolidated from a real observation should
            // outlive the session that produced it.
            memory_decay_rate: 0.02,
            apply_accepted_mutations: true,
        }
    }
}

/// Executes the developmental lifecycle over an organism.
///
/// Holds no state of its own beyond configuration. The organism is the state;
/// this is the sequence.
pub struct LifecycleDriver {
    config: LifecycleConfig,
}

impl Default for LifecycleDriver {
    fn default() -> Self {
        Self::new(LifecycleConfig::default())
    }
}

impl LifecycleDriver {
    pub fn new(config: LifecycleConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &LifecycleConfig {
        &self.config
    }

    /// Run one full developmental turn.
    ///
    /// Returns what the turn did. Errors only when the organism's own state
    /// could not be read or written — a mutation that could not be measured,
    /// or was measured and refused, is a normal outcome recorded in the report,
    /// not an error. Treating "EVE said no" as a failure would make the caller
    /// unable to distinguish it from "the database is gone".
    pub fn run_turn(
        &self,
        organism: &mut Organism,
        observations: &[Observation],
    ) -> Result<TurnReport, OrganismError> {
        let correlation_id = new_correlation_id();

        // Observe → Experience → Consolidate memory.
        //
        // An observation becomes an episodic memory: a specific thing that
        // happened. Generalizing it into semantic knowledge is the job of
        // repetition, not of a single turn.
        let mut memories = Vec::with_capacity(observations.len());
        for observation in observations {
            memories.push(organism.consolidate_memory(
                MemoryKind::Episodic,
                &observation.content,
                &observation.origin,
                observation.evidence.clone(),
                observation.confidence,
                self.config.memory_decay_rate,
                &correlation_id,
            )?);
        }

        // Reflect. Taken *before* mutations are generated, so the proposals
        // are a response to the organism's state as it actually is, and the
        // reflection in the report is the one the proposals were made against.
        let reflection = organism.reflect_and_announce(&correlation_id)?;

        // Update beliefs. Only sufficiently-confident observations become
        // claims the organism holds.
        let mut beliefs = Vec::new();
        for observation in observations {
            if observation.confidence < self.config.belief_confidence_floor {
                continue;
            }
            let belief = Belief::form(
                observation.content.clone(),
                adam_beliefs::EvidenceOrigin::Observation,
                observation.origin.clone(),
                observation.confidence,
            )?;
            beliefs.push(organism.update_belief(belief, &correlation_id));
        }

        // Generate mutations from the organism's own accumulated signals.
        let proposal_ids = organism.evolve_auto()?;
        for id in &proposal_ids {
            organism.announce_proposal(*id, &correlation_id);
        }

        // Validate in EVE → measure fitness → commit genome.
        let mut mutations = Vec::with_capacity(proposal_ids.len());
        for id in proposal_ids {
            mutations.push(self.decide(organism, id, &correlation_id));
        }

        Ok(TurnReport {
            correlation_id,
            memories,
            reflection,
            beliefs,
            mutations,
        })
    }

    /// Measure one proposal and act on the measurement.
    ///
    /// Every path out of here is recorded. There is no path that applies a
    /// change without a measurement behind it, and none that silently drops a
    /// proposal.
    fn decide(
        &self,
        organism: &mut Organism,
        id: ProposalId,
        correlation_id: &str,
    ) -> MutationOutcome {
        let fitness = match organism.validate_mutation(id, correlation_id) {
            Ok(fitness) => fitness,
            Err(err) => {
                // Unmeasurable is not the same as bad. The proposal stays
                // pending, and the reason travels with it so a human sees
                // whether EVE was unreachable or the mutation had nothing to
                // measure.
                return MutationOutcome::Escalated {
                    id,
                    reason: format!("not validated: {err}"),
                };
            }
        };

        match fitness.recommendation {
            Recommendation::Reject => {
                let reason = fitness.reason.clone();
                match organism.reject_mutation(id, &reason, correlation_id) {
                    Ok(()) => MutationOutcome::Rejected { id, reason },
                    Err(err) => MutationOutcome::Escalated {
                        id,
                        reason: format!(
                            "EVE recommended rejection but it could not be recorded: {err}"
                        ),
                    },
                }
            }

            Recommendation::NeedsReview => MutationOutcome::Escalated {
                id,
                reason: fitness.reason.clone(),
            },

            Recommendation::Approve => {
                if !self.config.apply_accepted_mutations {
                    return MutationOutcome::Escalated {
                        id,
                        reason: format!(
                            "approved (+{}bp) but this driver does not apply mutations: {}",
                            fitness.delta_bp.raw(),
                            fitness.reason
                        ),
                    };
                }
                match organism.accept_mutation(id, correlation_id) {
                    Ok(effect) => MutationOutcome::Accepted {
                        id,
                        effect_label: describe(&effect),
                        delta_bp: fitness.delta_bp.raw(),
                    },
                    // The governance rate limit lives here: an approved
                    // mutation that exceeds the acceptance budget stays
                    // pending for a later turn rather than being lost.
                    Err(err) => MutationOutcome::Escalated {
                        id,
                        reason: format!("approved by EVE but not applied: {err}"),
                    },
                }
            }
        }
    }
}

fn describe(effect: &AppliedEffect) -> String {
    match effect {
        AppliedEffect::SkillRetired { skill_name } => format!("retired skill '{skill_name}'"),
        AppliedEffect::GenomeAmended { label, .. } => format!("committed genome {label}"),
        AppliedEffect::AdvisoryOnly { note } => format!("advisory: {note}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adam_protocol::{
        BasisPoints, Component, EventKind, Measurement, Provenance, RecordingSink,
        SignedBasisPoints,
    };
    use adam_skills::Skill;
    use std::sync::Arc;

    /// A stubbed EVE returning `recommendation` for `proposal_id`.
    ///
    /// The lifecycle is what is under test here, not the measurement; using a
    /// real EVE would make these tests depend on a Node toolchain and on the
    /// simulation's outcome, neither of which is what a lifecycle test should
    /// be sensitive to.
    fn stub_eve(proposal_id: ProposalId, recommendation: Recommendation) -> adam_eve::EveClient {
        let measurement = Measurement {
            composite_bp: BasisPoints::from_ratio(0.7),
            task_success_bp: BasisPoints::from_ratio(0.7),
            frustration_bp: BasisPoints::from_ratio(0.3),
            trust_bp: BasisPoints::from_ratio(0.6),
            cognitive_load_bp: BasisPoints::from_ratio(0.4),
            runs: 9,
        };
        adam_eve::EveClient::new(Box::new(adam_eve::StubProvider::returning(
            adam_eve::FitnessResult {
                cp: "cp1".to_string(),
                doc_type: "FitnessResult".to_string(),
                id: uuid::Uuid::new_v4().to_string(),
                mutation_id: proposal_id.to_string(),
                seed: 1337,
                scenario_ids: vec!["excellent".to_string()],
                trials: 3,
                baseline: measurement.clone(),
                candidate: measurement,
                delta_bp: SignedBasisPoints::new(700),
                recommendation,
                reason: "stubbed measurement".to_string(),
                provenance: Provenance::now(Component::Eve, "eve:cp1/validate"),
            },
        )))
    }

    fn organism() -> Organism {
        Organism::new("ADAM", "lifecycle test organism", ":memory:").unwrap()
    }

    /// A skill whose recorded failures make `evolve_auto` propose retiring it,
    /// giving a turn something real to measure.
    fn failing_skill() -> Skill {
        let mut skill = Skill::discover("flaky", "unreliable", vec![]);
        skill.define_procedure("try thing", vec![]).unwrap();
        for _ in 0..4 {
            skill.record_test(false, "failed").unwrap();
        }
        skill.evaluate(0.5).unwrap();
        skill
    }

    fn observation(content: &str, confidence: f32) -> Observation {
        Observation::new(content, "tool:cargo_build")
            .with_evidence(vec!["exit code 101".to_string()])
            .with_confidence(confidence)
    }

    #[test]
    fn a_turn_with_no_observations_still_reflects_and_reports() {
        let mut organism = organism();
        let report = LifecycleDriver::default()
            .run_turn(&mut organism, &[])
            .unwrap();

        assert!(report.memories.is_empty());
        assert!(report.beliefs.is_empty());
        assert!(report.mutations.is_empty());
        assert!(!report.changed_anything());
        assert_eq!(report.reflection.genome_version, "1.0");
    }

    #[test]
    fn observations_become_episodic_memories() {
        let mut organism = organism();
        let report = LifecycleDriver::default()
            .run_turn(
                &mut organism,
                &[observation("cargo build failed: missing dependency", 0.9)],
            )
            .unwrap();

        assert_eq!(report.memories.len(), 1);
        let hits = organism
            .memory_query("cargo build failure", None, 5)
            .unwrap();
        assert!(hits
            .iter()
            .any(|(record, _)| record.id == report.memories[0]));
    }

    #[test]
    fn only_confident_observations_become_beliefs() {
        let mut organism = organism();
        let driver = LifecycleDriver::default();
        let report = driver
            .run_turn(
                &mut organism,
                &[
                    observation("the build is reproducible", 0.95),
                    // Below the floor: recorded as a memory, not held as a claim.
                    observation("the flake might be a race", 0.3),
                ],
            )
            .unwrap();

        assert_eq!(report.memories.len(), 2);
        assert_eq!(report.beliefs.len(), 1);
        assert_eq!(organism.beliefs().all_active().len(), 1);
    }

    #[test]
    fn a_turn_emits_one_correlated_event_per_stage() {
        let sink = Arc::new(RecordingSink::new());
        let mut organism = organism().with_events(sink.clone());

        let report = LifecycleDriver::default()
            .run_turn(&mut organism, &[observation("a thing happened", 0.9)])
            .unwrap();

        let kinds = sink.kinds();
        assert!(kinds.contains(&EventKind::MemoryConsolidated));
        assert!(kinds.contains(&EventKind::ReflectionCompleted));
        assert!(kinds.contains(&EventKind::BeliefUpdated));

        // The turn is reconstructible from its correlation id alone, which is
        // the property that makes an event log an audit trail rather than a
        // stream of disconnected facts.
        assert_eq!(sink.turn(&report.correlation_id).len(), sink.len());
        assert!(sink.turn("some-other-turn").is_empty());
    }

    #[test]
    fn an_approved_mutation_is_measured_then_applied() {
        let mut organism = organism();
        organism.register_skill(failing_skill());

        // Discover the proposal id the way a turn will, so the stub can be
        // pinned to it. `evolve_auto` is deterministic given the same state.
        let ids = organism.evolve_auto().unwrap();
        assert_eq!(ids.len(), 1);
        for id in &ids {
            organism
                .reject_mutation(*id, "setup", "setup-turn")
                .unwrap();
        }

        let sink = Arc::new(RecordingSink::new());
        let mut organism = organism.with_events(sink.clone());
        let pending = organism.evolve_auto().unwrap();
        let proposal_id = pending[0];
        // Re-propose through a fresh organism wired to a stub pinned to this id.
        let mut organism = organism.with_eve(stub_eve(proposal_id, Recommendation::Approve));

        let outcome = LifecycleDriver::default().decide(&mut organism, proposal_id, "turn");

        match outcome {
            MutationOutcome::Accepted { delta_bp, .. } => assert_eq!(delta_bp, 700),
            other => panic!("expected Accepted, got {other:?}"),
        }
        assert!(organism.skills().is_empty(), "the skill should be retired");

        let kinds = sink.kinds();
        assert!(kinds.contains(&EventKind::FitnessMeasured));
        assert!(kinds.contains(&EventKind::MutationAccepted));
    }

    #[test]
    fn a_rejected_measurement_refuses_the_mutation_and_changes_nothing() {
        let mut organism = organism();
        organism.register_skill(failing_skill());
        let ids = organism.evolve_auto().unwrap();
        let proposal_id = ids[0];

        let sink = Arc::new(RecordingSink::new());
        let mut organism = organism
            .with_events(sink.clone())
            .with_eve(stub_eve(proposal_id, Recommendation::Reject));

        let outcome = LifecycleDriver::default().decide(&mut organism, proposal_id, "turn");
        assert!(matches!(outcome, MutationOutcome::Rejected { .. }));
        assert_eq!(
            organism.skills().len(),
            1,
            "nothing should have been retired"
        );
        assert!(sink.kinds().contains(&EventKind::MutationRejected));
    }

    #[test]
    fn a_needs_review_measurement_escalates_and_leaves_the_proposal_pending() {
        let mut organism = organism();
        organism.register_skill(failing_skill());
        let proposal_id = organism.evolve_auto().unwrap()[0];

        let mut organism = organism.with_eve(stub_eve(proposal_id, Recommendation::NeedsReview));
        let outcome = LifecycleDriver::default().decide(&mut organism, proposal_id, "turn");

        assert!(matches!(outcome, MutationOutcome::Escalated { .. }));
        assert_eq!(organism.proposals().pending().len(), 1);
        assert_eq!(organism.skills().len(), 1);
    }

    #[test]
    fn an_unreachable_eve_escalates_rather_than_applying_blind() {
        // The property that matters most: no measurement means no change.
        let mut organism = organism();
        organism.register_skill(failing_skill());
        let proposal_id = organism.evolve_auto().unwrap()[0];

        // No `with_eve` at all.
        let outcome = LifecycleDriver::default().decide(&mut organism, proposal_id, "turn");
        match outcome {
            MutationOutcome::Escalated { reason, .. } => {
                assert!(reason.contains("not validated"), "{reason}");
            }
            other => panic!("expected Escalated, got {other:?}"),
        }
        assert_eq!(organism.skills().len(), 1);
        assert_eq!(organism.proposals().pending().len(), 1);
    }

    #[test]
    fn a_dry_run_driver_measures_everything_and_applies_nothing() {
        let mut organism = organism();
        organism.register_skill(failing_skill());
        let proposal_id = organism.evolve_auto().unwrap()[0];

        let sink = Arc::new(RecordingSink::new());
        let mut organism = organism
            .with_events(sink.clone())
            .with_eve(stub_eve(proposal_id, Recommendation::Approve));

        let driver = LifecycleDriver::new(LifecycleConfig {
            apply_accepted_mutations: false,
            ..LifecycleConfig::default()
        });
        let outcome = driver.decide(&mut organism, proposal_id, "turn");

        match outcome {
            MutationOutcome::Escalated { reason, .. } => {
                assert!(reason.contains("does not apply mutations"), "{reason}");
            }
            other => panic!("expected Escalated, got {other:?}"),
        }
        // Measured, but untouched.
        assert!(sink.kinds().contains(&EventKind::FitnessMeasured));
        assert!(!sink.kinds().contains(&EventKind::MutationAccepted));
        assert_eq!(organism.skills().len(), 1);
    }

    #[test]
    fn a_full_turn_runs_every_stage_in_order() {
        let sink = Arc::new(RecordingSink::new());
        let mut organism = organism();
        organism.register_skill(failing_skill());
        let proposal_id = organism.evolve_auto().unwrap()[0];
        organism
            .reject_mutation(proposal_id, "setup", "setup")
            .unwrap();

        let mut organism = organism.with_events(sink.clone());
        // The turn's own evolve_auto will regenerate an equivalent proposal
        // with a fresh id, which no stub can be pinned to — so this turn
        // exercises the unreachable-EVE path, and asserts the stages before it
        // all ran.
        let report = LifecycleDriver::default()
            .run_turn(&mut organism, &[observation("the build broke again", 0.9)])
            .unwrap();

        let kinds = sink
            .turn(&report.correlation_id)
            .into_iter()
            .map(|e| e.kind)
            .collect::<Vec<_>>();

        let position = |kind: EventKind| kinds.iter().position(|k| *k == kind);
        let memory = position(EventKind::MemoryConsolidated).expect("consolidation ran");
        let reflection = position(EventKind::ReflectionCompleted).expect("reflection ran");
        let belief = position(EventKind::BeliefUpdated).expect("belief update ran");
        let proposed = position(EventKind::MutationProposed).expect("a mutation was proposed");

        // Observe → Consolidate → Reflect → Beliefs → Mutations. Reflection is
        // taken before mutations are generated so the proposals respond to the
        // state the reflection describes.
        assert!(memory < reflection, "consolidation precedes reflection");
        assert!(reflection < belief, "reflection precedes belief update");
        assert!(belief < proposed, "beliefs precede mutation generation");

        assert_eq!(report.memories.len(), 1);
        assert_eq!(report.beliefs.len(), 1);
        assert_eq!(report.mutations.len(), 1);
        assert!(!report.changed_anything());
    }
}
