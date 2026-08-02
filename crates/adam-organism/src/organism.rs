//! The organism: composes genome history, memory, skills, beliefs, and the
//! evolution engine into one coherent, stateful unit. This is the surface
//! the MCP server (Phase 7) wraps — every `adam_*` tool maps to one method
//! here.

use adam_beliefs::{Belief, BeliefError, BeliefId, BeliefRegistry};
use adam_eve::{
    EvaluationResult, EvaluationThresholds, Recommendation, SimulationEvaluator, TrialFn,
};
use adam_evolution::{
    BeliefInstabilitySignal, EvolutionEngine, EvolutionProposal, EvolutionSignals,
    EvolutionThresholds, ProposalError, ProposalId, ProposalKind, ProposalStore,
    RecurringConflictSignal, SkillFailureSignal,
};
use adam_governance::{AuditEntry, EvolutionLimits, GovernanceError, GovernanceGate};
use adam_kernel::{Genome, GenomeDiff, GenomeError, GenomeHistory, GenomeVersion, VersionId};
use adam_memory::{MemoryError, MemoryId, MemoryKind, MemoryRecord, MemoryStore, Provenance};
use adam_skills::{Skill, SkillRegistry};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::embedding::embed;

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
        "genome field '{0}' cannot be amended automatically (only preferences.* is supported)"
    )]
    UnsupportedGenomeField(String),
    #[error(
        "genome field '{field}' requires an EVE evaluation recommending approval before it can be accepted (found: {found})"
    )]
    EveApprovalRequired { field: String, found: String },
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
    eve: SimulationEvaluator,
    evaluations: std::collections::HashMap<ProposalId, EvaluationResult>,
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
            eve: SimulationEvaluator::new(EvaluationThresholds::default(), 5),
            evaluations: std::collections::HashMap::new(),
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
            eve: SimulationEvaluator::new(EvaluationThresholds::default(), 5),
            evaluations: std::collections::HashMap::new(),
            genome_path: Some(genome_path.to_string()),
        };
        organism.persist_genome()?;
        Ok(organism)
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

    pub fn memory_query(
        &self,
        query: &str,
        kind: Option<MemoryKind>,
        top_k: usize,
    ) -> Result<Vec<(MemoryRecord, f32)>, OrganismError> {
        Ok(self.memory.query_similar(&embed(query), kind, top_k)?)
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

        let mut retractions: std::collections::HashMap<String, (f32, u32)> =
            std::collections::HashMap::new();
        for belief in self.beliefs.all() {
            if !belief.is_active() {
                retractions
                    .entry(belief.statement.clone())
                    .and_modify(|(confidence, count)| {
                        *confidence = belief.confidence;
                        *count += 1;
                    })
                    .or_insert((belief.confidence, 1));
            }
        }
        let belief_instabilities = retractions
            .into_iter()
            .map(
                |(statement, (confidence, retraction_count))| BeliefInstabilitySignal {
                    statement,
                    confidence,
                    retraction_count,
                },
            )
            .collect();

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

    /// Score a pending proposal through EVE's sandboxed simulation before
    /// it is accepted, storing the result so [`Organism::accept_mutation`]
    /// can consult it. The caller supplies the actual trial mechanics
    /// (`trial_fn`, e.g. a sandbox test replay) — this crate only wires
    /// the result into the organism's approval gate for genome amendments
    /// beyond `preferences.*`, which are too consequential to trust on
    /// self-reported confidence alone.
    pub fn evaluate_mutation(
        &mut self,
        id: ProposalId,
        trial_fn: &TrialFn,
    ) -> Result<EvaluationResult, OrganismError> {
        let proposal = self
            .proposals
            .get(id)
            .ok_or(OrganismError::ProposalNotFound(id))?;
        let result = self.eve.evaluate(proposal, trial_fn);
        self.evaluations.insert(id, result.clone());
        Ok(result)
    }

    /// Like [`Organism::evaluate_mutation`], but for callers (such as the
    /// MCP transport) that cannot pass a Rust closure and instead report
    /// trial outcomes they already collected as plain data.
    pub fn evaluate_mutation_from_trials(
        &mut self,
        id: ProposalId,
        trials: Vec<adam_eve::TrialOutcome>,
    ) -> Result<EvaluationResult, OrganismError> {
        let proposal = self
            .proposals
            .get(id)
            .ok_or(OrganismError::ProposalNotFound(id))?;
        let result = adam_eve::evaluate_from_trials(self.eve.thresholds(), proposal, trials);
        self.evaluations.insert(id, result.clone());
        Ok(result)
    }

    /// Accept a pending proposal and apply its concrete effect. This is
    /// the one place mutation actually happens — everywhere else, changes
    /// require this explicit, auditable step. Gated by the evolution rate
    /// limit: if the organism has already accepted too many mutations in
    /// the current window, this fails and the proposal remains `Proposed`
    /// (untouched) for later retry rather than being silently dropped.
    pub fn accept_mutation(&mut self, id: ProposalId) -> Result<AppliedEffect, OrganismError> {
        self.governance.authorize_acceptance()?;

        let kind = {
            let proposal = self
                .proposals
                .get_mut(id)
                .ok_or(OrganismError::ProposalNotFound(id))?;
            proposal.accept()?;
            proposal.kind.clone()
        };
        let effect = self.apply(id, &kind)?;
        self.governance.log_acceptance(id, format!("{effect:?}"));
        Ok(effect)
    }

    pub fn reject_mutation(&mut self, id: ProposalId) -> Result<(), OrganismError> {
        let proposal = self
            .proposals
            .get_mut(id)
            .ok_or(OrganismError::ProposalNotFound(id))?;
        proposal.reject()?;
        self.governance.log_rejection(id);
        Ok(())
    }

    fn apply(
        &mut self,
        id: ProposalId,
        kind: &ProposalKind,
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
                if let Some(key) = field.strip_prefix("preferences.") {
                    // Preferences are low-stakes and reversible (see
                    // DESIGN.md), so they remain ungated by EVE.
                    genome.preferences.insert(key.to_string(), suggested_value.clone());
                } else {
                    self.require_eve_approval(id, field)?;
                    apply_list_amendment(&mut genome, field, current_value, suggested_value)?;
                }
                let new_version = self
                    .history
                    .commit(genome, format!("accepted mutation: amend {field}"));
                self.persist_genome()?;
                let label = self.history.get(new_version)?.label.clone();
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

    /// Genome fields beyond `preferences.*` (values/goals/capabilities/
    /// policies) touch core identity and are irreversible in the sense
    /// that rollback is a forward-only new commit, not a silent undo — so
    /// they require a prior EVE evaluation on this exact proposal that
    /// recommended `Approve` before they can be accepted.
    fn require_eve_approval(&self, id: ProposalId, field: &str) -> Result<(), OrganismError> {
        match self.evaluations.get(&id) {
            Some(result) if result.recommendation == Recommendation::Approve => Ok(()),
            Some(result) => Err(OrganismError::EveApprovalRequired {
                field: field.to_string(),
                found: format!("{:?}", result.recommendation),
            }),
            None => Err(OrganismError::EveApprovalRequired {
                field: field.to_string(),
                found: "no evaluation recorded".to_string(),
            }),
        }
    }

    // -- reflection ----------------------------------------------------

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

/// Applies an EVE-approved amendment to one of the genome's list fields
/// (`values`, `goals`, `capabilities`, `policies`). `field` must be
/// `"<list>.append"` (adds `suggested_value`, deduplicated) or
/// `"<list>.remove"` (removes an entry equal to `current_value`).
fn apply_list_amendment(
    genome: &mut Genome,
    field: &str,
    current_value: &str,
    suggested_value: &str,
) -> Result<(), OrganismError> {
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
            if !list.iter().any(|item| item == suggested_value) {
                list.push(suggested_value.to_string());
            }
            Ok(())
        }
        "remove" => {
            list.retain(|item| item != current_value);
            Ok(())
        }
        _ => Err(OrganismError::UnsupportedGenomeField(field.to_string())),
    }
}
