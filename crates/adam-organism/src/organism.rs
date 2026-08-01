//! The organism: composes genome history, memory, skills, beliefs, and the
//! evolution engine into one coherent, stateful unit. This is the surface
//! the MCP server (Phase 7) wraps — every `adam_*` tool maps to one method
//! here.

use adam_beliefs::{Belief, BeliefError, BeliefId, BeliefRegistry};
use adam_evolution::{
    EvolutionEngine, EvolutionProposal, EvolutionSignals, EvolutionThresholds, ProposalError,
    ProposalId, ProposalKind, ProposalStore,
};
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
    #[error("proposal {0} not found")]
    ProposalNotFound(ProposalId),
    #[error("skill '{0}' not found")]
    SkillNotFound(String),
    #[error("genome field '{0}' cannot be amended automatically (only preferences.* is supported)")]
    UnsupportedGenomeField(String),
}

/// The concrete, auditable outcome of applying an accepted proposal.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "effect")]
pub enum AppliedEffect {
    SkillRetired { skill_name: String },
    GenomeAmended { new_version: VersionId, label: String },
    AdvisoryOnly { note: String },
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
        Ok(self.history.rollback(target, reason)?)
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

    /// Accept a pending proposal and apply its concrete effect. This is
    /// the one place mutation actually happens — everywhere else, changes
    /// require this explicit, auditable step.
    pub fn accept_mutation(&mut self, id: ProposalId) -> Result<AppliedEffect, OrganismError> {
        let kind = {
            let proposal = self
                .proposals
                .get_mut(id)
                .ok_or(OrganismError::ProposalNotFound(id))?;
            proposal.accept()?;
            proposal.kind.clone()
        };
        self.apply(&kind)
    }

    pub fn reject_mutation(&mut self, id: ProposalId) -> Result<(), OrganismError> {
        let proposal = self
            .proposals
            .get_mut(id)
            .ok_or(OrganismError::ProposalNotFound(id))?;
        proposal.reject()?;
        Ok(())
    }

    fn apply(&mut self, kind: &ProposalKind) -> Result<AppliedEffect, OrganismError> {
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
                suggested_value,
                ..
            } => {
                let key = field
                    .strip_prefix("preferences.")
                    .ok_or_else(|| OrganismError::UnsupportedGenomeField(field.clone()))?;
                let mut genome = self.history.head().genome.clone();
                genome
                    .preferences
                    .insert(key.to_string(), suggested_value.clone());
                let new_version = self
                    .history
                    .commit(genome, format!("accepted mutation: amend {field}"));
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

    // -- reflection ----------------------------------------------------

    pub fn reflect(&self) -> Result<ReflectionSummary, OrganismError> {
        let head = self.history.head();
        Ok(ReflectionSummary {
            genome_version: head.label.clone(),
            genome_version_id: head.id,
            total_memories: self.memory.all()?.len(),
            active_beliefs: self.beliefs.all_active().len(),
            promoted_skills: self.skills.by_stage(adam_skills::SkillStage::Promoted).len(),
            rejected_skills: self.skills.by_stage(adam_skills::SkillStage::Rejected).len(),
            pending_proposals: self.proposals.pending().len(),
            accepted_proposals: self.proposals.accepted().len(),
        })
    }
}
