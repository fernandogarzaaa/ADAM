//! The organism: composes genome history, memory, skills, beliefs, and the
//! evolution engine into one coherent, stateful unit. This is the surface
//! the MCP server (Phase 7) wraps — every `adam_*` tool maps to one method
//! here.

use adam_beliefs::{Belief, BeliefError, BeliefId, BeliefRegistry};
use adam_eve::{EveClient, FitnessError, FitnessResult, Recommendation};
use adam_evolution::{
    BeliefInstabilitySignal, EvolutionEngine, EvolutionProposal, EvolutionSignals,
    EvolutionThresholds, ProposalError, ProposalId, ProposalKind, ProposalStore,
    RecurringConflictSignal, SkillFailureSignal,
};
use adam_governance::{AuditEntry, EvolutionLimits, GovernanceError, GovernanceGate};
use adam_kernel::{Genome, GenomeDiff, GenomeError, GenomeHistory, GenomeVersion, VersionId};
use adam_memory::{MemoryError, MemoryId, MemoryKind, MemoryRecord, MemoryStore, Provenance};
use adam_protocol::{
    Component, Event, EventKind, EventSink, NullSink, PayloadValue, SubjectType,
};
use adam_skills::{Skill, SkillRegistry};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

use crate::embedding::embed;

/// Groups every event emitted while processing one developmental turn.
///
/// A turn is the unit that makes an event log reconstructible: given a
/// correlation id, the whole Observe-through-Commit sequence can be replayed
/// from the log without joining against anything else.
pub type CorrelationId = String;

/// A fresh correlation id, for a caller starting a turn of its own.
pub fn new_correlation_id() -> CorrelationId {
    uuid::Uuid::new_v4().to_string()
}

#[derive(Debug, Error)]
pub enum OrganismError {
    #[error(transparent)]
    Genome(#[from] GenomeError),
    #[error(transparent)]
    Memory(#[from] MemoryError),
    #[error(transparent)]
    Belief(#[from] BeliefError),
    #[error(transparent)]
    Proposal(#[from] ProposalError),
    #[error(transparent)]
    Governance(#[from] GovernanceError),
    #[error("proposal {0} not found")]
    ProposalNotFound(ProposalId),
    #[error("skill '{0}' not found")]
    SkillNotFound(String),
    #[error(
        "genome field '{0}' is not a supported amendment target (expected preferences.<key>, or <values|goals|capabilities|policies>.<append|remove>)"
    )]
    UnsupportedGenomeField(String),
    #[error(
        "genome field '{field}' requires a fitness measurement from EVE recommending approval before it can be accepted (found: {found})"
    )]
    EveApprovalRequired { field: String, found: String },
    #[error(
        "no fitness provider is configured, so genome field '{field}' cannot be validated; \
         construct the organism with `with_eve` before accepting amendments beyond preferences.*"
    )]
    NoFitnessProvider { field: String },
    #[error("fitness measurement failed: {0}")]
    Fitness(String),
    #[error("failed to persist genome to '{path}': {source}")]
    GenomePersistence {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to (de)serialize genome history: {0}")]
    GenomeSerialization(#[from] serde_json::Error),
}

/// The concrete, auditable outcome of applying an accepted proposal.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "effect")]
pub enum AppliedEffect {
    SkillRetired {
        skill_name: String,
    },
    GenomeAmended {
        new_version: VersionId,
        label: String,
    },
    AdvisoryOnly {
        note: String,
    },
}

/// A point-in-time self-assessment across every subsystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionSummary {
    pub genome_version: String,
    pub genome_version_id: VersionId,
    pub total_memories: usize,
    pub active_beliefs: usize,
    pub promoted_skills: usize,
    pub rejected_skills: usize,
    pub pending_proposals: usize,
    pub accepted_proposals: usize,
}

pub struct Organism {
    history: GenomeHistory,
    memory: MemoryStore,
    skills: SkillRegistry,
    beliefs: BeliefRegistry,
    proposals: ProposalStore,
    engine: EvolutionEngine,
    governance: GovernanceGate,
    /// The link to EVE. `None` until [`Organism::with_eve`] supplies one, in
    /// which case genome amendments beyond `preferences.*` cannot be accepted
    /// at all — refusing is the safe failure, since the alternative is
    /// accepting an identity change with no evidence behind it.
    eve: Option<EveClient>,
    /// Fitness measurements obtained for pending proposals, consumed by
    /// [`Organism::accept_mutation`] and dropped once a proposal is decided.
    fitness: std::collections::HashMap<ProposalId, FitnessResult>,
    /// Where this organism announces what it does. Defaults to discarding.
    events: std::sync::Arc<dyn EventSink>,
    /// The most recent event id per causal chain, so a later event can name
    /// the earlier one that caused it.
    ///
    /// Keyed by the thing the chain is *about* rather than by any single
    /// event's subject: a proposal's lifecycle runs from `MutationProposed`
    /// through `GenomeCommitted`, and that last event's subject is a genome
    /// version. Only edges the organism actually knows are recorded — nothing
    /// is inferred from the order events happen to arrive in, because a
    /// sequence is not a cause.
    ///
    /// A `Mutex` because several announcing methods take `&self`.
    causes: std::sync::Mutex<std::collections::HashMap<String, String>>,
    genome_path: Option<String>,
}

impl Organism {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        memory_path: &str,
    ) -> Result<Self, OrganismError> {
        let genome = Genome::new(name, description);
        let history = GenomeHistory::init(genome, "genesis");
        let memory = MemoryStore::open(memory_path)?;
        Ok(Self {
            history,
            memory,
            skills: SkillRegistry::new(),
            beliefs: BeliefRegistry::new(),
            proposals: ProposalStore::new(),
            engine: EvolutionEngine::new(EvolutionThresholds::default()),
            governance: GovernanceGate::new(EvolutionLimits::default()),
            eve: None,
            fitness: std::collections::HashMap::new(),
            events: std::sync::Arc::new(NullSink),
            causes: std::sync::Mutex::new(std::collections::HashMap::new()),
            genome_path: None,
        })
    }

    /// Like [`Organism::new`], but persists genome history to `genome_path`
    /// as JSON: if the file already exists, its history is loaded instead
    /// of starting a fresh genesis, and every subsequent commit/rollback
    /// is written back — giving identity continuity across process
    /// restarts (and, since the file format is plain JSON, across LLM
    /// provider backends running the same MCP server).
    pub fn open(
        name: impl Into<String>,
        description: impl Into<String>,
        memory_path: &str,
        genome_path: &str,
    ) -> Result<Self, OrganismError> {
        let memory = MemoryStore::open(memory_path)?;
        let history = match std::fs::read_to_string(genome_path) {
            Ok(json) => serde_json::from_str(&json)?,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                GenomeHistory::init(Genome::new(name, description), "genesis")
            }
            Err(source) => {
                return Err(OrganismError::GenomePersistence {
                    path: genome_path.to_string(),
                    source,
                })
            }
        };
        let organism = Self {
            history,
            memory,
            skills: SkillRegistry::new(),
            beliefs: BeliefRegistry::new(),
            proposals: ProposalStore::new(),
            engine: EvolutionEngine::new(EvolutionThresholds::default()),
            governance: GovernanceGate::new(EvolutionLimits::default()),
            eve: None,
            fitness: std::collections::HashMap::new(),
            events: std::sync::Arc::new(NullSink),
            causes: std::sync::Mutex::new(std::collections::HashMap::new()),
            genome_path: Some(genome_path.to_string()),
        };
        organism.persist_genome()?;
        Ok(organism)
    }

    /// Attach the link to EVE.
    ///
    /// Without one, [`Organism::validate_mutation`] and any acceptance that
    /// requires a fitness measurement fail with
    /// [`OrganismError::NoFitnessProvider`]. That is deliberate: an organism
    /// with no way to measure a change must refuse to make identity-level
    /// changes, not make them unmeasured.
    pub fn with_eve(mut self, client: EveClient) -> Self {
        self.eve = Some(client);
        self
    }

    /// Attach an event sink.
    ///
    /// Emission is fire-and-forget by design — a subsystem announcing a fact
    /// must not be able to fail because of what a listener does with it, or
    /// the nervous system becomes another way for the organism to break.
    pub fn with_events(mut self, sink: std::sync::Arc<dyn EventSink>) -> Self {
        self.events = sink;
        self
    }

    /// Announce a fact. Never fails, and never blocks the caller's work.
    ///
    /// Returns the id of the emitted event so a caller driving a turn can
    /// name it as the cause of something later.
    fn emit(
        &self,
        kind: EventKind,
        subject_id: impl Into<String>,
        subject_type: SubjectType,
        correlation_id: &str,
        payload: &[(&str, PayloadValue)],
    ) -> String {
        self.emit_as(
            Component::Adam,
            kind,
            subject_id,
            subject_type,
            correlation_id,
            payload,
        )
    }

    /// Announce a fact that some *other* component performed.
    ///
    /// Separate from [`Organism::emit`] so that relaying is never accidental.
    /// ADAM writes this log line, but ADAM did not do the thing: a measurement
    /// is performed by an evaluator, over a boundary ADAM does not control, and
    /// an event claiming otherwise would misattribute the one act the whole
    /// governance chain rests on.
    ///
    /// `actor` must come from a document ADAM has *verified*, never from a
    /// document ADAM composed — the distinction between copying an attribution
    /// and asserting one. [`EventKind::emitters`] then rejects any actor the
    /// kind does not permit, so a relay cannot launder an attribution either.
    fn emit_as(
        &self,
        actor: Component,
        kind: EventKind,
        subject_id: impl Into<String>,
        subject_type: SubjectType,
        correlation_id: &str,
        payload: &[(&str, PayloadValue)],
    ) -> String {
        let subject_id = subject_id.into();
        let chain = subject_id.clone();
        self.emit_in_chain(
            actor,
            kind,
            subject_id,
            subject_type,
            correlation_id,
            payload,
            &chain,
        )
    }

    /// Announce a fact that continues the causal chain tracked under `chain`.
    ///
    /// The emitted event names the previous event in that chain as its cause,
    /// and then becomes the chain's newest link. An unknown chain simply
    /// produces an event with no `causation_id` — a missing edge is honest,
    /// whereas a guessed one would make the audit trail worthless.
    fn emit_in_chain(
        &self,
        actor: Component,
        kind: EventKind,
        subject_id: impl Into<String>,
        subject_type: SubjectType,
        correlation_id: &str,
        payload: &[(&str, PayloadValue)],
        chain: &str,
    ) -> String {
        let payload: BTreeMap<String, PayloadValue> = payload
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect();
        let mut event = Event::new(
            actor,
            kind,
            subject_id,
            subject_type,
            correlation_id,
            payload,
            "adam:organism",
        );
        let id = event.id.clone();
        {
            let mut causes = self.causes.lock().expect("causes mutex poisoned");
            if let Some(cause) = causes.get(chain) {
                event = event.caused_by(cause.clone());
            }
            causes.insert(chain.to_string(), id.clone());
        }
        self.events.emit(&event);
        id
    }

    /// The newest event id in a causal chain, or `None` if nothing has been
    /// announced under that key yet.
    ///
    /// A caller driving a turn uses this to find the event it should attribute
    /// a later decision to — for example the memory event carrying an external
    /// observation, before the proposal that observation motivated exists.
    pub fn last_event(&self, chain: &str) -> Option<String> {
        self.causes
            .lock()
            .expect("causes mutex poisoned")
            .get(chain)
            .cloned()
    }

    /// Declare that the next event announced under `chain` was caused by
    /// `cause_event_id`.
    ///
    /// This is how evidence from outside the organism enters the causal
    /// record: only the caller knows that a particular proposal exists because
    /// of a particular observation, and asserting it explicitly keeps that
    /// claim traceable to whoever made it rather than inferred here.
    pub fn link_cause(&self, chain: &str, cause_event_id: impl Into<String>) {
        self.causes
            .lock()
            .expect("causes mutex poisoned")
            .insert(chain.to_string(), cause_event_id.into());
    }

    fn persist_genome(&self) -> Result<(), OrganismError> {
        let Some(path) = &self.genome_path else {
            return Ok(());
        };
        let json = serde_json::to_string_pretty(&self.history)?;
        std::fs::write(path, json).map_err(|source| OrganismError::GenomePersistence {
            path: path.clone(),
            source,
        })
    }

    // -- identity / genome --------------------------------------------

    pub fn identity(&self) -> &GenomeVersion {
        self.history.head()
    }

    pub fn genome(&self) -> &Genome {
        &self.history.head().genome
    }

    pub fn history(&self) -> &[GenomeVersion] {
        self.history.all()
    }

    pub fn rollback(
        &mut self,
        target: VersionId,
        reason: impl Into<String>,
    ) -> Result<VersionId, OrganismError> {
        let reason = reason.into();
        let new_version = self.history.rollback(target, reason.clone())?;
        self.persist_genome()?;
        self.governance.log_rollback(target, new_version, reason);
        Ok(new_version)
    }

    pub fn audit_log(&self) -> &[AuditEntry] {
        self.governance.audit_log()
    }

    pub fn diff(&self, from: VersionId, to: VersionId) -> Result<GenomeDiff, OrganismError> {
        Ok(self.history.diff(from, to)?)
    }

    // -- memory ----------------------------------------------------------

    pub fn memory_store(
        &self,
        kind: MemoryKind,
        content: &str,
        origin: &str,
        evidence: Vec<String>,
        confidence: f32,
        decay_rate: f32,
    ) -> Result<MemoryId, OrganismError> {
        let record = MemoryRecord::new(
            kind,
            content,
            embed(content),
            confidence,
            Provenance {
                origin: origin.to_string(),
                evidence,
            },
            decay_rate,
        );
        self.memory.store(&record)?;
        Ok(record.id)
    }

    /// Store a memory and announce it, for callers driving a developmental
    /// turn.
    ///
    /// Distinct from [`Organism::memory_store`] because that method takes
    /// `&self` — the store is internally synchronized — and a caller that
    /// merely records a memory should not be forced to hold the correlation id
    /// of a turn it is not part of.
    #[allow(clippy::too_many_arguments)]
    pub fn consolidate_memory(
        &self,
        kind: MemoryKind,
        content: &str,
        origin: &str,
        evidence: Vec<String>,
        confidence: f32,
        decay_rate: f32,
        correlation_id: &str,
    ) -> Result<MemoryId, OrganismError> {
        let id = self.memory_store(kind, content, origin, evidence, confidence, decay_rate)?;
        self.emit(
            EventKind::MemoryConsolidated,
            id.to_string(),
            SubjectType::Memory,
            correlation_id,
            &[
                ("kind", PayloadValue::from(kind.as_str())),
                ("origin", PayloadValue::from(origin)),
                // The observation itself, not just a pointer to it. Without
                // this the event log can say a memory was consolidated but not
                // what was noticed, and a causal chain that ends at an opaque
                // id cannot answer why the organism changed — it only says
                // where to look, in a store that may not have been kept.
                ("content", PayloadValue::from(content)),
                (
                    "confidence_bp",
                    PayloadValue::from(i64::from(
                        adam_protocol::BasisPoints::from_f32(confidence).raw(),
                    )),
                ),
            ],
        );
        Ok(id)
    }

    pub fn memory_query(
        &self,
        query: &str,
        kind: Option<MemoryKind>,
        top_k: usize,
    ) -> Result<Vec<(MemoryRecord, f32)>, OrganismError> {
        Ok(self.memory.query_similar(&embed(query), kind, top_k)?)
    }

    /// Approximate-nearest-neighbor counterpart to [`Organism::memory_query`],
    /// for memory volumes large enough that the exact O(n) scan becomes a
    /// bottleneck. Builds a fresh [`adam_memory::AnnIndex`] snapshot per
    /// call, so it trades a build cost for approximate results — callers
    /// with very high query volume relative to write volume may prefer to
    /// build once via `self.memory().build_ann_index()` and query that
    /// snapshot repeatedly instead. Building per call also means a single
    /// MCP tool invocation pays the full O(n log n) index-build cost
    /// before it can answer one query, so for organism-scale memory
    /// volumes (see DESIGN.md) `Organism::memory_query`'s exact scan is
    /// typically faster in practice; this path exists for callers that
    /// have already grown past that point.
    pub fn memory_query_ann(
        &self,
        query: &str,
        kind: Option<MemoryKind>,
        top_k: usize,
    ) -> Result<Vec<(MemoryRecord, f32)>, OrganismError> {
        let index = self.memory.build_ann_index(kind)?;
        let hits = index.query(&embed(query), top_k);
        let mut records = Vec::with_capacity(hits.len());
        for (id, score) in hits {
            if let Some(record) = self.memory.get(id)? {
                records.push((record, score));
            }
        }
        Ok(records)
    }

    pub fn memory(&self) -> &MemoryStore {
        &self.memory
    }

    // -- beliefs -----------------------------------------------------------

    pub fn beliefs(&self) -> &BeliefRegistry {
        &self.beliefs
    }

    pub fn beliefs_mut(&mut self) -> &mut BeliefRegistry {
        &mut self.beliefs
    }

    pub fn form_belief(&mut self, belief: Belief) -> BeliefId {
        self.beliefs.upsert(belief)
    }

    /// Form or update a belief and announce it.
    pub fn update_belief(&mut self, belief: Belief, correlation_id: &str) -> BeliefId {
        let statement = belief.statement.clone();
        let confidence = belief.confidence;
        let active = belief.is_active();
        let id = self.beliefs.upsert(belief);
        self.emit(
            EventKind::BeliefUpdated,
            id.to_string(),
            SubjectType::Belief,
            correlation_id,
            &[
                ("statement", PayloadValue::from(statement)),
                (
                    "confidence_bp",
                    PayloadValue::from(i64::from(
                        adam_protocol::BasisPoints::from_f32(confidence).raw(),
                    )),
                ),
                ("active", PayloadValue::from(active)),
            ],
        );
        id
    }

    // -- skills ------------------------------------------------------------

    pub fn skills(&self) -> &SkillRegistry {
        &self.skills
    }

    pub fn skills_mut(&mut self) -> &mut SkillRegistry {
        &mut self.skills
    }

    pub fn register_skill(&mut self, skill: Skill) -> adam_skills::SkillId {
        self.skills.upsert(skill)
    }

    /// Register a skill and, if it has reached the promoted stage, announce
    /// that the organism learned it.
    ///
    /// Only promotion is announced. A skill in an earlier stage is a
    /// hypothesis, not a capability, and emitting `SkillLearned` for one would
    /// make the event mean something weaker than its name.
    pub fn learn_skill(&mut self, skill: Skill, correlation_id: &str) -> adam_skills::SkillId {
        let name = skill.name.clone();
        let promoted = skill.stage == adam_skills::SkillStage::Promoted;
        let fitness = skill.fitness_score;
        let id = self.skills.upsert(skill);
        if promoted {
            self.emit(
                EventKind::SkillLearned,
                id.to_string(),
                SubjectType::Skill,
                correlation_id,
                &[
                    ("name", PayloadValue::from(name)),
                    (
                        "fitness_bp",
                        PayloadValue::from(i64::from(
                            adam_protocol::BasisPoints::from_f32(fitness).raw(),
                        )),
                    ),
                ],
            );
        }
        id
    }

    /// Derive [`EvolutionSignals`] from the organism's own current state
    /// instead of requiring a caller to assemble them: chronically failing
    /// skills, beliefs that keep getting retracted or superseded on the
    /// same statement, and memory topics that keep losing contradictions.
    /// Genome drift signals still require external judgment (there is no
    /// structural indicator for "this policy is stale") and are left
    /// empty here; callers may still merge their own into the result.
    pub fn collect_signals(&self) -> Result<EvolutionSignals, OrganismError> {
        let skill_failures = self
            .skills
            .all()
            .into_iter()
            .filter(|s| !s.test_results.is_empty() && s.fitness_score < 1.0)
            .map(|s| SkillFailureSignal {
                skill_name: s.name.clone(),
                fitness_score: s.fitness_score,
                failure_count: s.test_results.iter().filter(|r| !r.passed).count() as u32,
                failures: s.failures.clone(),
            })
            .collect();

        let mut retractions: std::collections::HashMap<
            String,
            (f32, chrono::DateTime<chrono::Utc>, u32),
        > = std::collections::HashMap::new();
        for belief in self.beliefs.all() {
            if !belief.is_active() {
                retractions
                    .entry(belief.statement.clone())
                    .and_modify(|(confidence, updated_at, count)| {
                        // `self.beliefs.all()` iterates a HashMap, so
                        // insertion order is nondeterministic — keep the
                        // most-recently-updated belief's confidence
                        // rather than whichever happened to be visited
                        // last, so the signal is reproducible.
                        if belief.updated_at > *updated_at {
                            *confidence = belief.confidence;
                            *updated_at = belief.updated_at;
                        }
                        *count += 1;
                    })
                    .or_insert((belief.confidence, belief.updated_at, 1));
            }
        }
        let mut belief_instabilities: Vec<BeliefInstabilitySignal> = retractions
            .into_iter()
            .map(
                |(statement, (confidence, _, retraction_count))| BeliefInstabilitySignal {
                    statement,
                    confidence,
                    retraction_count,
                },
            )
            .collect();
        belief_instabilities.sort_by(|a, b| a.statement.cmp(&b.statement));

        let recurring_conflicts = self
            .memory
            .conflict_topics()?
            .into_iter()
            .map(|(topic, occurrences)| RecurringConflictSignal { topic, occurrences })
            .collect();

        Ok(EvolutionSignals {
            skill_failures,
            belief_instabilities,
            recurring_conflicts,
            genome_drifts: Vec::new(),
        })
    }

    /// Analyze automatically-collected signals (see [`Organism::collect_signals`])
    /// and record every generated proposal, returning their ids. This is
    /// the proactive counterpart to [`Organism::evolve`], which requires a
    /// caller-supplied [`EvolutionSignals`].
    pub fn evolve_auto(&mut self) -> Result<Vec<ProposalId>, OrganismError> {
        let signals = self.collect_signals()?;
        Ok(self.evolve(&signals))
    }

    // -- evolution -----------------------------------------------------

    pub fn proposals(&self) -> &ProposalStore {
        &self.proposals
    }

    /// Analyze signals and record every generated proposal, returning
    /// their ids. Proposals sit `Proposed` until explicitly decided.
    pub fn evolve(&mut self, signals: &EvolutionSignals) -> Vec<ProposalId> {
        let generated = self.engine.analyze(signals);
        self.proposals.record_all(generated)
    }

    /// Manually record a proposal (e.g. organism- or user-initiated,
    /// outside the automatic threshold analysis in [`Organism::evolve`]).
    pub fn propose_mutation(&mut self, proposal: EvolutionProposal) -> ProposalId {
        self.proposals.record(proposal)
    }

    /// Emit a `MutationProposed` event for a recorded proposal.
    ///
    /// Separate from [`Organism::propose_mutation`] because a caller driving a
    /// developmental turn owns the correlation id that groups the turn's
    /// events, and the recording call itself does not know it.
    pub fn announce_proposal(&self, id: ProposalId, correlation_id: &str) {
        let Some(proposal) = self.proposals.get(id) else {
            return;
        };
        self.emit(
            EventKind::MutationProposed,
            id.to_string(),
            SubjectType::Mutation,
            correlation_id,
            &[
                ("kind", PayloadValue::from(kind_label(&proposal.kind))),
                ("target", PayloadValue::from(target_of(&proposal.kind))),
                (
                    "confidence_bp",
                    PayloadValue::from(
                        adam_protocol::BasisPoints::from_f32(proposal.confidence).raw() as i64,
                    ),
                ),
                (
                    "risk_bp",
                    PayloadValue::from(adam_eve::intrinsic_risk(&proposal.kind).raw() as i64),
                ),
            ],
        );
    }

    /// Measure a pending proposal in EVE, storing the result so
    /// [`Organism::accept_mutation`] can consult it.
    ///
    /// This is the "validate inside EVE" step of the developmental lifecycle,
    /// and it is the one the organism cannot perform for itself: the
    /// measurement is produced by a separate component, over a process
    /// boundary, and is rejected here unless EVE authored it. The organism
    /// supplies the question — which mutation, against which genome — and
    /// nothing else.
    ///
    /// Fails rather than degrading when no provider is configured. An organism
    /// that cannot measure a change must refuse to make it, not make it blind.
    pub fn validate_mutation(
        &mut self,
        id: ProposalId,
        correlation_id: &str,
    ) -> Result<FitnessResult, OrganismError> {
        let proposal = self
            .proposals
            .get(id)
            .ok_or(OrganismError::ProposalNotFound(id))?
            .clone();

        let eve = self
            .eve
            .as_ref()
            .ok_or_else(|| OrganismError::NoFitnessProvider {
                field: target_of(&proposal.kind),
            })?;

        // The genome the measurement is pinned to. `genome_after_hash` is the
        // hash the genome *would* have if the proposal were applied, computed
        // without applying it — so the measurement is attributable to a
        // specific before/after pair even though nothing has changed yet.
        let before_hash = self.history.head().genome.content_hash();
        let after_hash = self.prospective_genome_hash(&proposal.kind);

        let result = eve
            .validate(&proposal, &before_hash, &after_hash)
            .map_err(|err: FitnessError| OrganismError::Fitness(err.to_string()))?;

        // Relayed, not authored. The evaluator named here is copied from the
        // result's provenance, which `measure_and_verify` has already checked
        // is a real evaluator and is the one this request was dispatched to.
        // Before this the emitter was derived from the event kind, so an
        // ADAM-written FitnessMeasured was recorded as EVE's — internally
        // consistent and factually wrong, and undetectable in the log.
        self.emit_as(
            result.provenance.authored_by,
            EventKind::FitnessMeasured,
            id.to_string(),
            SubjectType::Mutation,
            correlation_id,
            &[
                (
                    "recommendation",
                    PayloadValue::from(format!("{:?}", result.recommendation)),
                ),
                (
                    "delta_bp",
                    PayloadValue::from(i64::from(result.delta_bp.raw())),
                ),
                ("runs", PayloadValue::from(result.baseline.runs)),
                ("seed", PayloadValue::from(result.seed)),
                ("reason", PayloadValue::from(result.reason.clone())),
            ],
        );

        self.fitness.insert(id, result.clone());
        Ok(result)
    }

    /// The hash the genome would have if `kind` were applied, without applying
    /// it.
    ///
    /// Returns the current hash for proposals that change no genome field,
    /// which is honest: for those, before and after genomes really are the
    /// same document.
    fn prospective_genome_hash(&self, kind: &ProposalKind) -> String {
        let mut genome = self.history.head().genome.clone();
        if let ProposalKind::AmendGenome {
            field,
            current_value,
            suggested_value,
        } = kind
        {
            if let Some(key) = field.strip_prefix("preferences.") {
                genome
                    .preferences
                    .insert(key.to_string(), suggested_value.clone());
            } else {
                // A field this organism cannot amend yields the unchanged
                // hash rather than an error: the acceptance path is where an
                // unsupported field is rejected, and duplicating that
                // judgment here would let the two disagree.
                let _ = apply_list_amendment(&mut genome, field, current_value, suggested_value);
            }
        }
        genome.content_hash()
    }

    /// The fitness measurement recorded for a proposal, if one has been taken.
    pub fn fitness_for(&self, id: ProposalId) -> Option<&FitnessResult> {
        self.fitness.get(&id)
    }

    /// Accept a pending proposal and apply its concrete effect. This is
    /// the one place mutation actually happens — everywhere else, changes
    /// require this explicit, auditable step. Gated by the evolution rate
    /// limit: if the organism has already accepted too many mutations in
    /// the current window, this fails and the proposal remains `Proposed`
    /// (untouched) for later retry rather than being silently dropped.
    pub fn accept_mutation(
        &mut self,
        id: ProposalId,
        correlation_id: &str,
    ) -> Result<AppliedEffect, OrganismError> {
        self.governance.authorize_acceptance()?;

        let kind = {
            let proposal = self
                .proposals
                .get(id)
                .ok_or(OrganismError::ProposalNotFound(id))?;
            proposal.kind.clone()
        };

        // Validate before mutating the proposal's state: if this were
        // done after `proposal.accept()`, a validation failure (missing
        // EVE approval, an already-removed skill, ...) would leave the
        // proposal permanently stuck `Accepted` with no effect applied
        // and no audit entry — no way to retry or reject it.
        self.validate_applicable(id, &kind)?;

        {
            let proposal = self
                .proposals
                .get_mut(id)
                .ok_or(OrganismError::ProposalNotFound(id))?;
            proposal.accept()?;
        }

        let effect = self.apply(id, &kind, correlation_id)?;
        self.governance.log_acceptance(id, format!("{effect:?}"));

        let delta_bp = self
            .fitness
            .get(&id)
            .map(|f| i64::from(f.delta_bp.raw()))
            .unwrap_or(0);
        self.emit(
            EventKind::MutationAccepted,
            id.to_string(),
            SubjectType::Mutation,
            correlation_id,
            &[
                ("kind", PayloadValue::from(kind_label(&kind))),
                ("target", PayloadValue::from(target_of(&kind))),
                ("effect", PayloadValue::from(effect_label(&effect))),
                ("fitness_delta_bp", PayloadValue::from(delta_bp)),
                (
                    "validated",
                    PayloadValue::from(self.fitness.contains_key(&id)),
                ),
            ],
        );

        // The measurement recorded against this proposal has served its
        // purpose (gating this exact acceptance); dropping it keeps `fitness`
        // from growing without bound as proposals cycle through
        // validate -> accept/reject.
        self.fitness.remove(&id);
        Ok(effect)
    }

    /// Read-only precondition checks for [`Organism::apply`], run before
    /// the proposal is marked `Accepted` so a failing precondition leaves
    /// the proposal untouched instead of stuck.
    fn validate_applicable(
        &self,
        id: ProposalId,
        kind: &ProposalKind,
    ) -> Result<(), OrganismError> {
        match kind {
            ProposalKind::RetireSkill { skill_name } => self
                .skills
                .find_by_name(skill_name)
                .map(|_| ())
                .ok_or_else(|| OrganismError::SkillNotFound(skill_name.clone())),
            ProposalKind::AmendGenome { field, .. } => {
                if field.strip_prefix("preferences.").is_none() {
                    self.require_eve_approval(id, field)?;
                }
                Ok(())
            }
            ProposalKind::ReconcileBelief { .. } | ProposalKind::InvestigateConflict { .. } => {
                Ok(())
            }
        }
    }

    /// Refuse a pending proposal.
    ///
    /// `reason` is recorded on the emitted event rather than inferred, because
    /// the interesting rejections are the ones a human made for a reason no
    /// measurement captured.
    pub fn reject_mutation(
        &mut self,
        id: ProposalId,
        reason: &str,
        correlation_id: &str,
    ) -> Result<(), OrganismError> {
        let kind = {
            let proposal = self
                .proposals
                .get_mut(id)
                .ok_or(OrganismError::ProposalNotFound(id))?;
            proposal.reject()?;
            proposal.kind.clone()
        };
        self.governance.log_rejection(id);

        self.emit(
            EventKind::MutationRejected,
            id.to_string(),
            SubjectType::Mutation,
            correlation_id,
            &[
                ("kind", PayloadValue::from(kind_label(&kind))),
                ("target", PayloadValue::from(target_of(&kind))),
                ("reason", PayloadValue::from(reason)),
            ],
        );

        // See the matching cleanup in `accept_mutation`: a rejected
        // proposal's measurement has no further use.
        self.fitness.remove(&id);
        Ok(())
    }

    fn apply(
        &mut self,
        id: ProposalId,
        kind: &ProposalKind,
        correlation_id: &str,
    ) -> Result<AppliedEffect, OrganismError> {
        match kind {
            ProposalKind::RetireSkill { skill_name } => {
                let removed = self
                    .skills
                    .remove_by_name(skill_name)
                    .ok_or_else(|| OrganismError::SkillNotFound(skill_name.clone()))?;
                Ok(AppliedEffect::SkillRetired {
                    skill_name: removed.name,
                })
            }
            ProposalKind::AmendGenome {
                field,
                current_value,
                suggested_value,
            } => {
                let mut genome = self.history.head().genome.clone();
                let changed = if let Some(key) = field.strip_prefix("preferences.") {
                    // Preferences are low-stakes and reversible (see
                    // DESIGN.md), so they remain ungated by EVE.
                    let previous = genome.preferences.get(key).cloned();
                    genome
                        .preferences
                        .insert(key.to_string(), suggested_value.clone());
                    previous.as_deref() != Some(suggested_value.as_str())
                } else {
                    self.require_eve_approval(id, field)?;
                    apply_list_amendment(&mut genome, field, current_value, suggested_value)?
                };

                if !changed {
                    // Nothing actually changed (e.g. appending a value
                    // already present, or removing one that's absent) —
                    // committing anyway would create a genome version
                    // with an empty diff, polluting `adam_history`'s
                    // rollback/diff chain with no-op entries.
                    return Ok(AppliedEffect::AdvisoryOnly {
                        note: format!(
                            "amend {field}: suggested value already reflected in the genome; no new version created"
                        ),
                    });
                }

                let reason = format!("accepted mutation: amend {field}");
                let new_version = self.history.commit(genome, reason.clone());
                self.persist_genome()?;
                let label = self.history.get(new_version)?.label.clone();

                // Chained under the mutation rather than the genome version:
                // this commit exists because that proposal was accepted, and
                // the version it produced has no earlier event of its own.
                self.emit_in_chain(
                    Component::Adam,
                    EventKind::GenomeCommitted,
                    new_version.to_string(),
                    SubjectType::Genome,
                    correlation_id,
                    &[
                        ("version_label", PayloadValue::from(label.clone())),
                        ("reason", PayloadValue::from(reason)),
                        ("mutation_id", PayloadValue::from(id.to_string())),
                    ],
                    &id.to_string(),
                );

                Ok(AppliedEffect::GenomeAmended { new_version, label })
            }
            ProposalKind::ReconcileBelief { statement } => Ok(AppliedEffect::AdvisoryOnly {
                note: format!(
                    "belief '{statement}' flagged for reconciliation; requires manual evidence review, not auto-applied"
                ),
            }),
            ProposalKind::InvestigateConflict { topic } => Ok(AppliedEffect::AdvisoryOnly {
                note: format!(
                    "recurring conflict on '{topic}' flagged for investigation; requires manual review, not auto-applied"
                ),
            }),
        }
    }

    /// The acceptance gate for identity-level change.
    ///
    /// Genome fields beyond `preferences.*` (values, goals, capabilities,
    /// policies) touch what the organism *is*, and are only reversible by a
    /// forward commit rather than a silent undo. Accepting one requires a
    /// fitness measurement on this exact proposal that recommends approval.
    ///
    /// The measurement must additionally be authentic — authored by a real
    /// evaluator (never by ADAM), about this mutation, with baseline and
    /// candidate compared over the same number of runs. `EveClient` already
    /// refuses anything else, so a measurement in `self.fitness` has passed that
    /// check; re-asserting it here is defence in depth against a future path
    /// that populates the map some other way, and costs one comparison.
    ///
    /// The expected evaluator is taken from the result itself, which looks
    /// circular and is not: the comparison that catches a substituted evaluator
    /// is *who was asked versus who answered*, and only `measure_and_verify`
    /// knows who was asked. What remains checkable here is the property that
    /// does not need that knowledge — that the author is an evaluator at all —
    /// and it is the property that stops ADAM scoring itself.
    fn require_eve_approval(&self, id: ProposalId, field: &str) -> Result<(), OrganismError> {
        let Some(result) = self.fitness.get(&id) else {
            return Err(if self.eve.is_none() {
                OrganismError::NoFitnessProvider {
                    field: field.to_string(),
                }
            } else {
                OrganismError::EveApprovalRequired {
                    field: field.to_string(),
                    found: "no measurement recorded; call validate_mutation first".to_string(),
                }
            });
        };

        if let Some(detail) =
            result.authenticity_failure(&id.to_string(), result.provenance.authored_by)
        {
            return Err(OrganismError::EveApprovalRequired {
                field: field.to_string(),
                found: detail,
            });
        }

        if result.recommendation == Recommendation::Approve {
            Ok(())
        } else {
            Err(OrganismError::EveApprovalRequired {
                field: field.to_string(),
                found: format!("{:?} — {}", result.recommendation, result.reason),
            })
        }
    }

    // -- reflection ----------------------------------------------------

    /// Produce a self-assessment and announce it.
    ///
    /// The announcing variant of [`Organism::reflect`], for callers driving a
    /// developmental turn.
    pub fn reflect_and_announce(
        &self,
        correlation_id: &str,
    ) -> Result<ReflectionSummary, OrganismError> {
        let summary = self.reflect()?;
        self.emit(
            EventKind::ReflectionCompleted,
            summary.genome_version_id.to_string(),
            SubjectType::Reflection,
            correlation_id,
            &[
                (
                    "genome_version",
                    PayloadValue::from(summary.genome_version.clone()),
                ),
                ("total_memories", PayloadValue::from(summary.total_memories)),
                ("active_beliefs", PayloadValue::from(summary.active_beliefs)),
                (
                    "promoted_skills",
                    PayloadValue::from(summary.promoted_skills),
                ),
                (
                    "pending_proposals",
                    PayloadValue::from(summary.pending_proposals),
                ),
            ],
        );
        Ok(summary)
    }

    pub fn reflect(&self) -> Result<ReflectionSummary, OrganismError> {
        let head = self.history.head();
        Ok(ReflectionSummary {
            genome_version: head.label.clone(),
            genome_version_id: head.id,
            total_memories: self.memory.all()?.len(),
            active_beliefs: self.beliefs.all_active().len(),
            promoted_skills: self
                .skills
                .by_stage(adam_skills::SkillStage::Promoted)
                .len(),
            rejected_skills: self
                .skills
                .by_stage(adam_skills::SkillStage::Rejected)
                .len(),
            pending_proposals: self.proposals.pending().len(),
            accepted_proposals: self.proposals.accepted().len(),
        })
    }
}

/// Stable, machine-readable label for a proposal kind, used in event payloads.
///
/// Matches the CP/1 `MutationKind` vocabulary rather than Rust's variant
/// spelling, so an event log reads the same as the documents it accompanies.
fn kind_label(kind: &ProposalKind) -> &'static str {
    match kind {
        ProposalKind::RetireSkill { .. } => "retire_skill",
        ProposalKind::ReconcileBelief { .. } => "reconcile_belief",
        ProposalKind::InvestigateConflict { .. } => "investigate_conflict",
        ProposalKind::AmendGenome { .. } => "amend_genome",
    }
}

/// What a proposal acts on: a genome field path, a skill name, a belief
/// statement, or a conflict topic.
fn target_of(kind: &ProposalKind) -> String {
    match kind {
        ProposalKind::RetireSkill { skill_name } => skill_name.clone(),
        ProposalKind::ReconcileBelief { statement } => statement.clone(),
        ProposalKind::InvestigateConflict { topic } => topic.clone(),
        ProposalKind::AmendGenome { field, .. } => field.clone(),
    }
}

/// Stable label for an applied effect, for event payloads.
fn effect_label(effect: &AppliedEffect) -> &'static str {
    match effect {
        AppliedEffect::SkillRetired { .. } => "skill_retired",
        AppliedEffect::GenomeAmended { .. } => "genome_amended",
        AppliedEffect::AdvisoryOnly { .. } => "advisory_only",
    }
}

/// Applies an EVE-approved amendment to one of the genome's list fields
/// (`values`, `goals`, `capabilities`, `policies`). `field` must be
/// `"<list>.append"` (adds `suggested_value`, deduplicated) or
/// `"<list>.remove"` (removes an entry equal to `current_value`). Returns
/// whether the list actually changed, so callers can avoid committing a
/// no-op genome version.
///
/// Public because an evaluator has to build the candidate genome the *same*
/// way acceptance will. A second implementation of "what this mutation means"
/// would let a provider measure one change while ADAM later applies another,
/// and the comparison would silently be about the wrong thing.
pub fn apply_list_amendment(
    genome: &mut Genome,
    field: &str,
    current_value: &str,
    suggested_value: &str,
) -> Result<bool, OrganismError> {
    let (list_name, operation) = field
        .split_once('.')
        .ok_or_else(|| OrganismError::UnsupportedGenomeField(field.to_string()))?;

    let list: &mut Vec<String> = match list_name {
        "values" => &mut genome.values,
        "goals" => &mut genome.goals,
        "capabilities" => &mut genome.capabilities,
        "policies" => &mut genome.policies,
        _ => return Err(OrganismError::UnsupportedGenomeField(field.to_string())),
    };

    match operation {
        "append" => {
            if list.iter().any(|item| item == suggested_value) {
                Ok(false)
            } else {
                list.push(suggested_value.to_string());
                Ok(true)
            }
        }
        "remove" => {
            let len_before = list.len();
            list.retain(|item| item != current_value);
            Ok(list.len() != len_before)
        }
        _ => Err(OrganismError::UnsupportedGenomeField(field.to_string())),
    }
}
