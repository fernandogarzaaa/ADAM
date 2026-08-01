//! Conflict resolution between contradictory memories.

use crate::record::{MemoryId, RelationKind};
use crate::store::{MemoryError, MemoryStore};

impl MemoryStore {
    /// Resolve a contradiction between two memories by keeping the
    /// higher-confidence one and marking the other as superseded — never
    /// deleting it, so the contradiction and its resolution remain
    /// auditable. Ties are broken in favor of `a`. Returns the winning
    /// memory's id.
    pub fn resolve_conflict(&self, a: MemoryId, b: MemoryId) -> Result<MemoryId, MemoryError> {
        let record_a = self.get(a)?.ok_or(MemoryError::NotFound(a))?;
        let record_b = self.get(b)?.ok_or(MemoryError::NotFound(b))?;

        let (winner, loser) = if record_a.confidence >= record_b.confidence {
            (record_a.id, record_b.id)
        } else {
            (record_b.id, record_a.id)
        };

        self.mark_superseded(loser, winner)?;
        self.relate(loser, winner, RelationKind::SupersededBy)?;
        Ok(winner)
    }
}
