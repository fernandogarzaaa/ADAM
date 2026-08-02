//! In-memory registry indexing beliefs by id, plus contradiction bookkeeping.

use std::collections::HashMap;

use crate::belief::{Belief, BeliefError, BeliefId};

/// Tracks every belief an ADAM organism currently holds or has held.
#[derive(Debug, Default)]
pub struct BeliefRegistry {
    beliefs: HashMap<BeliefId, Belief>,
}

impl BeliefRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn upsert(&mut self, belief: Belief) -> BeliefId {
        let id = belief.id;
        self.beliefs.insert(id, belief);
        id
    }

    pub fn get(&self, id: BeliefId) -> Option<&Belief> {
        self.beliefs.get(&id)
    }

    pub fn get_mut(&mut self, id: BeliefId) -> Option<&mut Belief> {
        self.beliefs.get_mut(&id)
    }

    pub fn all_active(&self) -> Vec<&Belief> {
        self.beliefs.values().filter(|b| b.is_active()).collect()
    }

    /// Every known belief regardless of status — used by automatic signal
    /// collection, which needs to see retracted/superseded beliefs to
    /// detect instability, not just what is currently active.
    pub fn all(&self) -> Vec<&Belief> {
        self.beliefs.values().collect()
    }

    pub fn len(&self) -> usize {
        self.beliefs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.beliefs.is_empty()
    }

    /// Record a symmetric contradiction between two beliefs.
    pub fn mark_contradicting(&mut self, a: BeliefId, b: BeliefId) {
        if let Some(belief) = self.beliefs.get_mut(&a) {
            belief.mark_contradicts(b);
        }
        if let Some(belief) = self.beliefs.get_mut(&b) {
            belief.mark_contradicts(a);
        }
    }

    /// Resolve a marked contradiction between two beliefs by confidence,
    /// superseding the loser. Returns the id of the belief that stayed
    /// active.
    pub fn resolve_conflict(&mut self, a: BeliefId, b: BeliefId) -> Result<BeliefId, BeliefError> {
        let mut belief_a = self
            .beliefs
            .remove(&a)
            .ok_or(BeliefError::NotContradicting(a, b))?;
        let mut belief_b = self
            .beliefs
            .remove(&b)
            .ok_or(BeliefError::NotContradicting(a, b))?;

        let result = belief_a.resolve_against(&mut belief_b);

        let winner = match result {
            Ok(true) => a,
            Ok(false) => b,
            Err(err) => {
                self.beliefs.insert(a, belief_a);
                self.beliefs.insert(b, belief_b);
                return Err(err);
            }
        };

        self.beliefs.insert(a, belief_a);
        self.beliefs.insert(b, belief_b);
        Ok(winner)
    }

    /// All beliefs whose statement contains `needle` (case-insensitive) —
    /// the lookup an organism performs when checking what it currently
    /// believes about a topic before acting.
    pub fn find_about(&self, needle: &str) -> Vec<&Belief> {
        let needle = needle.to_lowercase();
        self.beliefs
            .values()
            .filter(|b| b.is_active() && b.statement.to_lowercase().contains(&needle))
            .collect()
    }
}
