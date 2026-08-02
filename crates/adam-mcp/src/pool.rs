//! A registry of [`Organism`]s keyed by an opaque organism id, so one
//! `adam-mcp` process can serve more than one organism (e.g. one per
//! end user or per project) instead of the single implicit organism
//! every prior phase assumed. Organisms are created lazily on first use
//! via a caller-supplied factory, and kept alive for the life of the
//! process — there is no eviction, matching the "one organism per
//! process" simplicity this replaces at the smallest possible scope
//! increase.

use std::collections::HashMap;

use adam_organism::{Organism, OrganismError};

/// The organism id used when a caller doesn't specify one, preserving
/// single-organism behavior and the existing `ADAM_MEMORY_PATH`/
/// `ADAM_GENOME_PATH` env vars for anyone not opting into multi-organism
/// use.
pub const DEFAULT_ORGANISM_ID: &str = "default";

type Factory = Box<dyn FnMut(&str) -> Result<Organism, OrganismError>>;

pub struct OrganismPool {
    organisms: HashMap<String, Organism>,
    factory: Factory,
}

impl OrganismPool {
    pub fn new(factory: impl FnMut(&str) -> Result<Organism, OrganismError> + 'static) -> Self {
        Self {
            organisms: HashMap::new(),
            factory: Box::new(factory),
        }
    }

    /// Fetch the organism for `id`, creating it via the factory on first
    /// use. Every subsequent call with the same `id` returns the same
    /// in-memory organism until the process exits.
    pub fn get_or_create(&mut self, id: &str) -> Result<&mut Organism, OrganismError> {
        if !self.organisms.contains_key(id) {
            let organism = (self.factory)(id)?;
            self.organisms.insert(id.to_string(), organism);
        }
        Ok(self
            .organisms
            .get_mut(id)
            .expect("just inserted or already present"))
    }

    /// Ids of every organism created so far in this process.
    pub fn known_ids(&self) -> Vec<&str> {
        self.organisms.keys().map(String::as_str).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ephemeral_pool() -> OrganismPool {
        OrganismPool::new(|_id| Organism::new("ADAM", "test", ":memory:"))
    }

    #[test]
    fn the_same_id_returns_the_same_organism_across_calls() {
        let mut pool = ephemeral_pool();
        let v1 = pool.get_or_create("alice").unwrap().identity().id;
        let v2 = pool.get_or_create("alice").unwrap().identity().id;
        assert_eq!(v1, v2);
    }

    #[test]
    fn different_ids_get_independent_organisms() {
        let mut pool = ephemeral_pool();
        pool.get_or_create("alice")
            .unwrap()
            .memory_store(
                adam_memory::MemoryKind::Episodic,
                "alice's memory",
                "test",
                vec![],
                0.9,
                0.0,
            )
            .unwrap();

        let bob_memories = pool
            .get_or_create("bob")
            .unwrap()
            .reflect()
            .unwrap()
            .total_memories;
        assert_eq!(bob_memories, 0);
        assert_eq!(pool.known_ids().len(), 2);
    }
}
