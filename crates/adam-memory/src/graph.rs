//! Graph relationships between memories (`memory_relations` table).

use chrono::Utc;
use rusqlite::params;
use uuid::Uuid;

use crate::record::{MemoryId, MemoryRecord, RelationKind};
use crate::store::{row_to_record, MemoryError, MemoryStore};

impl MemoryStore {
    /// Create a directed, typed edge `from -> to` in the memory graph.
    pub fn relate(
        &self,
        from: MemoryId,
        to: MemoryId,
        kind: RelationKind,
    ) -> Result<Uuid, MemoryError> {
        let id = Uuid::new_v4();
        let created_at = Utc::now();
        self.conn.execute(
            "INSERT INTO memory_relations (id, from_id, to_id, kind, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                id.to_string(),
                from.to_string(),
                to.to_string(),
                kind.as_str(),
                created_at.to_rfc3339(),
            ],
        )?;
        Ok(id)
    }

    /// Memories that `id` points *to*, along with the relation type,
    /// e.g. the semantic memory an episodic memory generalizes into.
    pub fn related_outgoing(
        &self,
        id: MemoryId,
    ) -> Result<Vec<(MemoryRecord, RelationKind)>, MemoryError> {
        self.related_via("from_id", "to_id", id)
    }

    /// Memories that point *to* `id`, along with the relation type,
    /// e.g. the episodic memories a semantic memory was generalized from.
    pub fn related_incoming(
        &self,
        id: MemoryId,
    ) -> Result<Vec<(MemoryRecord, RelationKind)>, MemoryError> {
        self.related_via("to_id", "from_id", id)
    }

    fn related_via(
        &self,
        anchor_col: &str,
        other_col: &str,
        id: MemoryId,
    ) -> Result<Vec<(MemoryRecord, RelationKind)>, MemoryError> {
        let sql = format!(
            "SELECT m.id, m.kind, m.content, m.embedding, m.confidence, m.origin, m.evidence,
                    m.created_at, m.last_accessed_at, m.access_count, m.decay_rate, m.superseded_by,
                    r.kind
             FROM memory_relations r
             JOIN memories m ON m.id = r.{other_col}
             WHERE r.{anchor_col} = ?1"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params![id.to_string()], |row| {
            let relation_kind: String = row.get(12)?;
            Ok((row_to_record(row)?, relation_kind))
        })?;

        let mut out = Vec::new();
        for row in rows {
            let (record_result, relation_kind) = row?;
            let record = record_result?;
            let relation =
                RelationKind::parse(&relation_kind).expect("stored relation kind is always valid");
            out.push((record, relation));
        }
        Ok(out)
    }
}
