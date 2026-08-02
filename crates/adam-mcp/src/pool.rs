//! A registry of [`Organism`]s keyed by an opaque organism id, so one
//! `adam-mcp` process can serve more than one organism (e.g. one per
//! end user or per project) instead of the single implicit organism
//! every prior phase assumed. Organisms are created lazily on first use
//! via a caller-supplied factory. Ids are validated and the number held
//! in memory at once is bounded (see [`MAX_ORGANISMS`]), since an
//! `organism_id` is attacker-controlled input coming straight off the
//! MCP transport.

use std::collections::{HashMap, VecDeque};

use adam_organism::{Organism, OrganismError};

/// The organism id used when a caller doesn't specify one, preserving
/// single-organism behavior and the existing `ADAM_MEMORY_PATH`/
/// `ADAM_GENOME_PATH` env vars for anyone not opting into multi-organism
/// use.
pub const DEFAULT_ORGANISM_ID: &str = "default";

/// Maximum organisms held in memory at once. Factories in practice back
/// each organism with an open SQLite connection, so an unbounded pool
/// driven by attacker-controlled `organism_id`s is a file-descriptor and
/// memory exhaustion vector. Past this limit the least-recently-used
/// organism is dropped from memory (not deleted from disk — the next
/// request for it re-creates it via the factory).
const MAX_ORGANISMS: usize = 64;

type Factory = Box<dyn FnMut(&str) -> Result<Organism, OrganismError>>;

/// Errors from pool operations, distinct from [`OrganismError`] so a
/// rejected id is clearly distinguishable from a failure inside the
/// organism itself.
#[derive(Debug, thiserror::Error)]
pub enum PoolError {
    #[error(
        "invalid organism_id '{0}': must be 1-64 characters of [A-Za-z0-9_-] \
         (organism_id is used to build a filesystem path, so path separators \
         and traversal sequences are rejected)"
    )]
    InvalidId(String),
    #[error(transparent)]
    Organism(#[from] OrganismError),
}

pub struct OrganismPool {
    organisms: HashMap<String, Organism>,
    /// Least-recently-used order, oldest first. Kept in lockstep with
    /// `organisms`: every key present in one is present in the other.
    recency: VecDeque<String>,
    factory: Factory,
}

impl OrganismPool {
    pub fn new(factory: impl FnMut(&str) -> Result<Organism, OrganismError> + 'static) -> Self {
        Self {
            organisms: HashMap::new(),
            recency: VecDeque::new(),
            factory: Box::new(factory),
        }
    }

    /// Fetch the organism for `id`, creating it via the factory on first
    /// use. Every subsequent call with the same `id` returns the same
    /// in-memory organism, as long as it hasn't been evicted under
    /// memory pressure (see [`MAX_ORGANISMS`]) — eviction only drops it
    /// from memory, so a factory backed by persistent storage (the
    /// production `adam-mcp` factory) transparently reloads the same
    /// state on next use.
    pub fn get_or_create(&mut self, id: &str) -> Result<&mut Organism, PoolError> {
        validate_id(id)?;

        if !self.organisms.contains_key(id) {
            if self.organisms.len() >= MAX_ORGANISMS {
                if let Some(evicted) = self.recency.pop_front() {
                    self.organisms.remove(&evicted);
                }
            }
            let organism = (self.factory)(id)?;
            self.organisms.insert(id.to_string(), organism);
        }

        self.touch(id);
        Ok(self
            .organisms
            .get_mut(id)
            .expect("just inserted or already present"))
    }

    fn touch(&mut self, id: &str) {
        self.recency.retain(|existing| existing != id);
        self.recency.push_back(id.to_string());
    }

    /// Ids of every organism currently held in this process (subject to
    /// eviction — see [`MAX_ORGANISMS`]).
    pub fn known_ids(&self) -> Vec<&str> {
        self.organisms.keys().map(String::as_str).collect()
    }
}

/// `organism_id` is interpolated directly into filesystem paths by the
/// production factory (`<data_dir>/<id>_memory.db`), so it must not
/// contain path separators or traversal sequences. Restricting to a
/// small safe character set is simpler and more robust than trying to
/// block every traversal encoding.
fn validate_id(id: &str) -> Result<(), PoolError> {
    let valid = !id.is_empty()
        && id.len() <= 64
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    if valid {
        Ok(())
    } else {
        Err(PoolError::InvalidId(id.to_string()))
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

    #[test]
    fn ids_containing_path_separators_are_rejected() {
        let mut pool = ephemeral_pool();
        for hostile in ["../evil", "a/b", "a\\b", "..", ""] {
            let err = pool.get_or_create(hostile).map(|_| ()).unwrap_err();
            assert!(matches!(err, PoolError::InvalidId(_)), "id: {hostile:?}");
        }
    }

    #[test]
    fn an_id_longer_than_64_characters_is_rejected() {
        let mut pool = ephemeral_pool();
        let too_long = "a".repeat(65);
        let err = pool.get_or_create(&too_long).map(|_| ()).unwrap_err();
        assert!(matches!(err, PoolError::InvalidId(_)));
    }

    #[test]
    fn the_pool_evicts_the_least_recently_used_organism_past_the_cap() {
        let mut pool = ephemeral_pool();
        for i in 0..MAX_ORGANISMS {
            pool.get_or_create(&format!("org{i}")).unwrap();
        }
        assert_eq!(pool.known_ids().len(), MAX_ORGANISMS);

        // One more organism should evict "org0", the least recently used.
        pool.get_or_create("overflow").unwrap();
        assert_eq!(pool.known_ids().len(), MAX_ORGANISMS);
        assert!(!pool.known_ids().contains(&"org0"));
        assert!(pool.known_ids().contains(&"overflow"));
    }
}
