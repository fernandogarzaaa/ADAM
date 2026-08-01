//! In-memory registry indexing skills by id and lifecycle stage.

use std::collections::HashMap;

use crate::skill::{Skill, SkillId, SkillStage};

/// Tracks every known skill for an ADAM organism. This is the
/// organism-facing surface: callers register discovered skills, drive them
/// through their lifecycle via the [`Skill`] API, and query the registry
/// for skills ready to run (`Promoted`) or in need of attention
/// (`Rejected`).
#[derive(Debug, Default)]
pub struct SkillRegistry {
    skills: HashMap<SkillId, Skill>,
}

impl SkillRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new or updated skill snapshot, keyed by its id.
    pub fn upsert(&mut self, skill: Skill) -> SkillId {
        let id = skill.id;
        self.skills.insert(id, skill);
        id
    }

    pub fn get(&self, id: SkillId) -> Option<&Skill> {
        self.skills.get(&id)
    }

    pub fn get_mut(&mut self, id: SkillId) -> Option<&mut Skill> {
        self.skills.get_mut(&id)
    }

    /// All skills currently in a given lifecycle stage.
    pub fn by_stage(&self, stage: SkillStage) -> Vec<&Skill> {
        self.skills.values().filter(|s| s.stage == stage).collect()
    }

    /// All promoted skills whose trigger conditions include `condition` —
    /// the lookup an organism performs when deciding how to react to a
    /// situation.
    pub fn find_applicable(&self, condition: &str) -> Vec<&Skill> {
        self.skills
            .values()
            .filter(|s| {
                s.stage == SkillStage::Promoted
                    && s.trigger_conditions.iter().any(|c| c == condition)
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.skills.len()
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    pub fn find_by_name(&self, name: &str) -> Option<&Skill> {
        self.skills.values().find(|s| s.name == name)
    }

    /// Remove a skill by name (used when an accepted evolution proposal
    /// retires a chronically failing skill). Returns the removed skill.
    pub fn remove_by_name(&mut self, name: &str) -> Option<Skill> {
        let id = self.find_by_name(name)?.id;
        self.skills.remove(&id)
    }
}
