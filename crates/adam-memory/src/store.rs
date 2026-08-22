//! SQLite-backed durable storage for [`MemoryRecord`]s.

use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension, Row};
use thiserror::Error;

use crate::record::{MemoryId, MemoryKind, MemoryRecord, Provenance};

/// Errors raised by [`MemoryStore`] operations.
#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("memory {0} not found")]
    NotFound(MemoryId),
    #[error("corrupt stored data: {0}")]
    Corrupt(String),
}

/// Durable store for ADAM's episodic, semantic, procedural, and self
/// memories, backed by an embedded SQLite database (or `:memory:` for
/// ephemeral/test use).
pub struct MemoryStore {
    pub(crate) conn: Connection,
}

impl MemoryStore {
    /// Open (or create) a memory database at `path`. Pass `":memory:"` for
    /// an ephemeral in-process database.
    pub fn open(path: &str) -> Result<Self, MemoryError> {
        let conn = Connection::open(path)?;
        let store = Self { conn };
        store.init_schema()?;
        Ok(store)
    }

    fn init_schema(&self) -> Result<(), MemoryError> {
        self.conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS memories (
                id              TEXT PRIMARY KEY,
                kind            TEXT NOT NULL,
                content         TEXT NOT NULL,
                embedding       BLOB NOT NULL,
                confidence      REAL NOT NULL,
                origin          TEXT NOT NULL,
                evidence        TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                last_accessed_at TEXT NOT NULL,
                access_count    INTEGER NOT NULL,
                decay_rate      REAL NOT NULL,
                superseded_by   TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);

            CREATE TABLE IF NOT EXISTS memory_relations (
                id          TEXT PRIMARY KEY,
                from_id     TEXT NOT NULL,
                to_id       TEXT NOT NULL,
                kind        TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                FOREIGN KEY(from_id) REFERENCES memories(id),
                FOREIGN KEY(to_id) REFERENCES memories(id)
            );
            CREATE INDEX IF NOT EXISTS idx_relations_from ON memory_relations(from_id);
            CREATE INDEX IF NOT EXISTS idx_relations_to ON memory_relations(to_id);
            ",
        )?;
        Ok(())
    }

    /// Persist a new memory record.
    pub fn store(&self, record: &MemoryRecord) -> Result<(), MemoryError> {
        self.conn.execute(
            "INSERT INTO memories (
                id, kind, content, embedding, confidence, origin, evidence,
                created_at, last_accessed_at, access_count, decay_rate, superseded_by
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                record.id.to_string(),
                record.kind.as_str(),
                record.content,
                encode_embedding(&record.embedding),
                record.confidence,
                record.provenance.origin,
                serde_json::to_string(&record.provenance.evidence)?,
                record.created_at.to_rfc3339(),
                record.last_accessed_at.to_rfc3339(),
                record.access_count,
                record.decay_rate,
                record.superseded_by.map(|id| id.to_string()),
            ],
        )?;
        Ok(())
    }

    /// Look up a memory by id without updating its access bookkeeping.
    pub fn get(&self, id: MemoryId) -> Result<Option<MemoryRecord>, MemoryError> {
        self.conn
            .query_row(
                "SELECT id, kind, content, embedding, confidence, origin, evidence,
                        created_at, last_accessed_at, access_count, decay_rate, superseded_by
                 FROM memories WHERE id = ?1",
                params![id.to_string()],
                row_to_record,
            )
            .optional()
            .map_err(MemoryError::from)?
            .transpose()
    }

    /// Retrieve a memory and record that it was accessed (bumps
    /// `access_count` and refreshes `last_accessed_at`), which resets its
    /// effective decay clock.
    pub fn access(&self, id: MemoryId) -> Result<MemoryRecord, MemoryError> {
        let record = self.get(id)?.ok_or(MemoryError::NotFound(id))?;
        let now = Utc::now();
        self.conn.execute(
            "UPDATE memories SET last_accessed_at = ?1, access_count = access_count + 1 WHERE id = ?2",
            params![now.to_rfc3339(), id.to_string()],
        )?;
        let mut updated = record;
        updated.last_accessed_at = now;
        updated.access_count += 1;
        Ok(updated)
    }

    /// All memories of a given kind, most recently created first.
    pub fn query_by_kind(&self, kind: MemoryKind) -> Result<Vec<MemoryRecord>, MemoryError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, kind, content, embedding, confidence, origin, evidence,
                    created_at, last_accessed_at, access_count, decay_rate, superseded_by
             FROM memories WHERE kind = ?1 ORDER BY created_at DESC",
        )?;
        let rows = stmt.query_map(params![kind.as_str()], row_to_record)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row??);
        }
        Ok(out)
    }

    /// Every memory currently stored.
    pub fn all(&self) -> Result<Vec<MemoryRecord>, MemoryError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, kind, content, embedding, confidence, origin, evidence,
                    created_at, last_accessed_at, access_count, decay_rate, superseded_by
             FROM memories ORDER BY created_at ASC",
        )?;
        let rows = stmt.query_map([], row_to_record)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row??);
        }
        Ok(out)
    }

    /// Overwrite a memory's confidence score in place (used by decay and
    /// conflict resolution — the memory's identity and provenance are
    /// unchanged, only the trust level shifts).
    pub fn update_confidence(&self, id: MemoryId, confidence: f32) -> Result<(), MemoryError> {
        let affected = self.conn.execute(
            "UPDATE memories SET confidence = ?1 WHERE id = ?2",
            params![confidence.clamp(0.0, 1.0), id.to_string()],
        )?;
        if affected == 0 {
            return Err(MemoryError::NotFound(id));
        }
        Ok(())
    }

    /// Mark a memory as superseded by another, without deleting it —
    /// provenance requires that nothing ever disappears from history.
    pub fn mark_superseded(&self, id: MemoryId, by: MemoryId) -> Result<(), MemoryError> {
        let affected = self.conn.execute(
            "UPDATE memories SET superseded_by = ?1 WHERE id = ?2",
            params![by.to_string(), id.to_string()],
        )?;
        if affected == 0 {
            return Err(MemoryError::NotFound(id));
        }
        Ok(())
    }
}

pub(crate) fn encode_embedding(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub(crate) fn decode_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

pub(crate) fn row_to_record(row: &Row) -> rusqlite::Result<Result<MemoryRecord, MemoryError>> {
    Ok((|| -> Result<MemoryRecord, MemoryError> {
        let id: String = row.get(0)?;
        let kind: String = row.get(1)?;
        let content: String = row.get(2)?;
        let embedding: Vec<u8> = row.get(3)?;
        let confidence: f32 = row.get(4)?;
        let origin: String = row.get(5)?;
        let evidence: String = row.get(6)?;
        let created_at: String = row.get(7)?;
        let last_accessed_at: String = row.get(8)?;
        let access_count: u32 = row.get(9)?;
        let decay_rate: f32 = row.get(10)?;
        let superseded_by: Option<String> = row.get(11)?;

        Ok(MemoryRecord {
            id: id.parse().map_err(|e| {
                MemoryError::Corrupt(format!("stored id '{id}' is not a valid UUID: {e}"))
            })?,
            kind: MemoryKind::parse(&kind).ok_or_else(|| {
                MemoryError::Corrupt(format!("stored kind '{kind}' is not valid"))
            })?,
            content,
            embedding: decode_embedding(&embedding),
            confidence,
            provenance: Provenance {
                origin,
                evidence: serde_json::from_str(&evidence)?,
            },
            created_at: parse_rfc3339(&created_at)?,
            last_accessed_at: parse_rfc3339(&last_accessed_at)?,
            access_count,
            decay_rate,
            superseded_by: superseded_by
                .map(|s| {
                    s.parse().map_err(|e| {
                        MemoryError::Corrupt(format!("stored id '{s}' is not a valid UUID: {e}"))
                    })
                })
                .transpose()?,
        })
    })())
}

fn parse_rfc3339(s: &str) -> Result<DateTime<Utc>, MemoryError> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| {
            MemoryError::Corrupt(format!("stored timestamp '{s}' is not valid RFC3339: {e}"))
        })
}

#[cfg(test)]
mod corruption_tests {
    use super::*;
    use crate::record::{MemoryKind, MemoryRecord, Provenance};

    fn record() -> MemoryRecord {
        MemoryRecord::new(
            MemoryKind::Semantic,
            "a well-formed memory",
            vec![0.1, 0.2],
            0.7,
            Provenance {
                origin: "test".to_string(),
                evidence: vec![],
            },
            0.0,
        )
    }

    /// A hand-edited or incompatible-schema row with a non-UUID `id`
    /// column must surface as a `MemoryError::Corrupt`, not panic the
    /// process via an `.expect()` in the row-decoding path.
    #[test]
    fn reading_a_row_with_a_corrupt_id_returns_an_error_instead_of_panicking() {
        let store = MemoryStore::open(":memory:").unwrap();
        let good = record();
        store.store(&good).unwrap();
        store
            .conn
            .execute(
                "UPDATE memories SET id = 'not-a-uuid' WHERE id = ?1",
                params![good.id.to_string()],
            )
            .unwrap();

        let err = store.all().unwrap_err();
        assert!(
            matches!(err, MemoryError::Corrupt(_)),
            "expected MemoryError::Corrupt, got {err:?}"
        );
    }

    /// Likewise for an out-of-band `kind` value that no longer matches any
    /// `MemoryKind` variant (e.g. written by an incompatible future schema
    /// version).
    #[test]
    fn reading_a_row_with_an_unknown_kind_returns_an_error_instead_of_panicking() {
        let store = MemoryStore::open(":memory:").unwrap();
        let good = record();
        store.store(&good).unwrap();
        store
            .conn
            .execute(
                "UPDATE memories SET kind = 'not-a-real-kind' WHERE id = ?1",
                params![good.id.to_string()],
            )
            .unwrap();

        let err = store.get(good.id).unwrap_err();
        assert!(
            matches!(err, MemoryError::Corrupt(_)),
            "expected MemoryError::Corrupt, got {err:?}"
        );
    }

    /// And for a `created_at` timestamp that isn't valid RFC3339.
    #[test]
    fn reading_a_row_with_a_corrupt_timestamp_returns_an_error_instead_of_panicking() {
        let store = MemoryStore::open(":memory:").unwrap();
        let good = record();
        store.store(&good).unwrap();
        store
            .conn
            .execute(
                "UPDATE memories SET created_at = 'not-a-timestamp' WHERE id = ?1",
                params![good.id.to_string()],
            )
            .unwrap();

        let err = store.get(good.id).unwrap_err();
        assert!(
            matches!(err, MemoryError::Corrupt(_)),
            "expected MemoryError::Corrupt, got {err:?}"
        );
    }
}
