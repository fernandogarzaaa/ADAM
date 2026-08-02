//! Conflict resolution between contradictory memories.

use rusqlite::params;

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

    /// Recurring conflict topics: memories that have won a contradiction
    /// resolution more than once, grouped by content, with how many times
    /// each has been contested. Backs automatic evolution signal
    /// collection — a topic contested repeatedly suggests a systemic gap
    /// rather than one-off noise.
    pub fn conflict_topics(&self) -> Result<Vec<(String, u32)>, MemoryError> {
        let mut stmt = self.conn.prepare(
            "SELECT m.content, COUNT(*) as occurrences
             FROM memory_relations r
             JOIN memories m ON m.id = r.to_id
             WHERE r.kind = ?1
             GROUP BY m.content
             HAVING COUNT(*) > 1
             ORDER BY occurrences DESC",
        )?;
        let rows = stmt.query_map(params![RelationKind::SupersededBy.as_str()], |row| {
            let content: String = row.get(0)?;
            let occurrences: i64 = row.get(1)?;
            Ok((content, occurrences as u32))
        })?;

        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod conflict_topic_tests {
    use super::*;
    use crate::record::{MemoryKind, MemoryRecord, Provenance};

    fn record(content: &str, confidence: f32) -> MemoryRecord {
        MemoryRecord::new(
            MemoryKind::Semantic,
            content,
            vec![0.1, 0.2],
            confidence,
            Provenance {
                origin: "test".to_string(),
                evidence: vec![],
            },
            0.0,
        )
    }

    #[test]
    fn a_topic_contested_more_than_once_is_reported_as_recurring() {
        let store = MemoryStore::open(":memory:").unwrap();
        let winner = record("rust prevents data races", 0.9);
        store.store(&winner).unwrap();

        let loser_one = record("rust does not prevent data races", 0.4);
        store.store(&loser_one).unwrap();
        store.resolve_conflict(winner.id, loser_one.id).unwrap();

        let loser_two = record("actually rust has no such guarantee", 0.3);
        store.store(&loser_two).unwrap();
        store.resolve_conflict(winner.id, loser_two.id).unwrap();

        let topics = store.conflict_topics().unwrap();
        assert_eq!(topics.len(), 1);
        assert_eq!(topics[0], ("rust prevents data races".to_string(), 2));
    }

    #[test]
    fn a_topic_contested_only_once_is_not_recurring() {
        let store = MemoryStore::open(":memory:").unwrap();
        let winner = record("stable topic", 0.9);
        store.store(&winner).unwrap();
        let loser = record("challenger", 0.2);
        store.store(&loser).unwrap();
        store.resolve_conflict(winner.id, loser.id).unwrap();

        assert!(store.conflict_topics().unwrap().is_empty());
    }
}
