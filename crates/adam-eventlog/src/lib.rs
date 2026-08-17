//! Durable, append-only storage for CP/1 events.
//!
//! ADAM emits fourteen kinds of event and, before this crate, every one of them
//! went into `adam_protocol::NullSink` and was discarded. The events were
//! already correct — `correlation_id` groups a developmental turn and `seal()`
//! carries a content hash — they simply had nowhere to go.
//!
//! # Why this is episodic memory and not "logging"
//!
//! ADAM implements semantic memory (beliefs), procedural memory (skills) and
//! self memory (identity/genome). The one it does not implement is episodic:
//! what happened, when, and what caused it. That is what an event log is, so
//! this crate is the fourth memory type rather than an operational convenience.
//! Consequently it is append-only — a log that can be rewritten cannot credibly
//! answer "why did the organism change?", which is the only question it exists
//! to answer.
//!
//! `UPDATE` and `DELETE` are refused by SQLite triggers, not merely by the
//! absence of a method, so a caller holding a raw [`rusqlite::Connection`] to
//! the same file is still bound by them.

use std::sync::Mutex;

use adam_protocol::{Event, EventSink};
use rusqlite::{params, Connection, Row};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EventLogError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("could not seal event: {0}")]
    Canonical(String),
    #[error("event {id} is already in the log; events are append-only and ids are unique")]
    Duplicate { id: String },
}

/// One row of the log, as stored.
///
/// `document` is the full sealed event; the flattened columns exist so the log
/// is queryable without deserializing every row, and are derived from the
/// event rather than supplied independently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredEvent {
    /// Order of insertion. The log's ordering guarantee: `occurred_at` has
    /// millisecond resolution and several events within one turn routinely
    /// share a timestamp, so it cannot be the order.
    pub seq: i64,
    pub id: String,
    pub kind: String,
    pub occurred_at: String,
    pub actor: String,
    pub subject_id: String,
    pub subject_type: String,
    pub correlation_id: String,
    pub causation_id: Option<String>,
    pub payload: String,
    pub content_hash: String,
    pub document: String,
}

impl StoredEvent {
    fn from_row(row: &Row<'_>) -> rusqlite::Result<Self> {
        Ok(Self {
            seq: row.get("seq")?,
            id: row.get("id")?,
            kind: row.get("kind")?,
            occurred_at: row.get("occurred_at")?,
            actor: row.get("actor")?,
            subject_id: row.get("subject_id")?,
            subject_type: row.get("subject_type")?,
            correlation_id: row.get("correlation_id")?,
            causation_id: row.get("causation_id")?,
            payload: row.get("payload")?,
            content_hash: row.get("content_hash")?,
            document: row.get("document")?,
        })
    }

    /// Rebuild the CP/1 event from the stored document.
    pub fn event(&self) -> Result<Event, EventLogError> {
        Ok(serde_json::from_str(&self.document)?)
    }
}

/// A durable [`EventSink`] backed by SQLite.
pub struct SqliteEventLog {
    conn: Mutex<Connection>,
}

impl SqliteEventLog {
    /// Open (or create) a log at `path`. `":memory:"` gives an ephemeral one,
    /// matching `adam_memory::MemoryStore::open`.
    pub fn open(path: &str) -> Result<Self, EventLogError> {
        let conn = Connection::open(path)?;
        // WAL so a reader never blocks the organism mid-turn, and FULL
        // synchronous so an event `append` returned from has actually reached
        // the disk — surviving process termination is the point of the crate.
        // Both are no-ops for ":memory:".
        let _: String = conn.pragma_update_and_check(None, "journal_mode", "WAL", |row| row.get(0))?;
        conn.pragma_update(None, "synchronous", "FULL")?;
        let log = Self {
            conn: Mutex::new(conn),
        };
        log.init_schema()?;
        Ok(log)
    }

    fn init_schema(&self) -> Result<(), EventLogError> {
        self.lock().execute_batch(
            "
            CREATE TABLE IF NOT EXISTS events (
                seq            INTEGER PRIMARY KEY AUTOINCREMENT,
                id             TEXT NOT NULL UNIQUE,
                kind           TEXT NOT NULL,
                occurred_at    TEXT NOT NULL,
                actor          TEXT NOT NULL,
                subject_id     TEXT NOT NULL,
                subject_type   TEXT NOT NULL,
                correlation_id TEXT NOT NULL,
                causation_id   TEXT,
                payload        TEXT NOT NULL,
                content_hash   TEXT NOT NULL,
                document       TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_correlation ON events(correlation_id);
            CREATE INDEX IF NOT EXISTS idx_events_occurred_at ON events(occurred_at);
            CREATE INDEX IF NOT EXISTS idx_events_causation ON events(causation_id);
            CREATE INDEX IF NOT EXISTS idx_events_subject ON events(subject_id);

            -- Append-only, enforced by the database rather than by convention.
            CREATE TRIGGER IF NOT EXISTS events_no_update
            BEFORE UPDATE ON events
            BEGIN
                SELECT RAISE(ABORT, 'events are append-only: UPDATE refused');
            END;

            CREATE TRIGGER IF NOT EXISTS events_no_delete
            BEFORE DELETE ON events
            BEGIN
                SELECT RAISE(ABORT, 'events are append-only: DELETE refused');
            END;
            ",
        )?;
        Ok(())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().expect("event log mutex poisoned")
    }

    /// Append one event, returning its `seq`.
    ///
    /// Separate from [`EventSink::emit`], which cannot fail by design. A caller
    /// that needs to know whether the write landed calls this.
    pub fn append(&self, event: &Event) -> Result<i64, EventLogError> {
        // The hash is read back off the sealed form rather than off the struct:
        // `seal()` is what computes it, so the struct field is not yet
        // authoritative at this point.
        let sealed = event
            .seal()
            .map_err(|err| EventLogError::Canonical(err.to_string()))?;
        let content_hash = sealed
            .get("provenance")
            .and_then(|p| p.get("content_hash"))
            .and_then(|h| h.as_str())
            .unwrap_or_default()
            .to_string();
        let document = serde_json::to_string(&sealed)?;
        let payload = serde_json::to_string(&event.payload)?;

        let conn = self.lock();
        let result = conn.execute(
            "INSERT INTO events (
                id, kind, occurred_at, actor, subject_id, subject_type,
                correlation_id, causation_id, payload, content_hash, document
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                event.id,
                event.kind.as_str(),
                event.occurred_at.as_str(),
                event.actor.as_str(),
                event.subject_id,
                format!("{:?}", event.subject_type),
                event.correlation_id,
                event.causation_id,
                payload,
                content_hash,
                document,
            ],
        );
        match result {
            Ok(_) => Ok(conn.last_insert_rowid()),
            Err(rusqlite::Error::SqliteFailure(err, _))
                if err.code == rusqlite::ErrorCode::ConstraintViolation =>
            {
                Err(EventLogError::Duplicate {
                    id: event.id.clone(),
                })
            }
            Err(err) => Err(EventLogError::Sqlite(err)),
        }
    }

    /// Every event, in insertion order.
    pub fn all(&self) -> Result<Vec<StoredEvent>, EventLogError> {
        self.query("SELECT * FROM events ORDER BY seq", params![])
    }

    /// One developmental turn, in insertion order.
    pub fn turn(&self, correlation_id: &str) -> Result<Vec<StoredEvent>, EventLogError> {
        self.query(
            "SELECT * FROM events WHERE correlation_id = ?1 ORDER BY seq",
            params![correlation_id],
        )
    }

    /// Events naming `id` as their cause — the forward edge of the causal DAG.
    pub fn caused_by(&self, id: &str) -> Result<Vec<StoredEvent>, EventLogError> {
        self.query(
            "SELECT * FROM events WHERE causation_id = ?1 ORDER BY seq",
            params![id],
        )
    }

    pub fn by_id(&self, id: &str) -> Result<Option<StoredEvent>, EventLogError> {
        Ok(self
            .query("SELECT * FROM events WHERE id = ?1", params![id])?
            .pop())
    }

    /// Walk `causation_id` backwards from `id` to a root, oldest first.
    ///
    /// This is the query that answers "why did the organism change?", so it is
    /// a method rather than something every caller reassembles. A cycle would
    /// make it loop, so visited ids terminate the walk; the log should not
    /// contain one, and a truncated chain beats hanging.
    pub fn causal_chain(&self, id: &str) -> Result<Vec<StoredEvent>, EventLogError> {
        let mut chain = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut cursor = Some(id.to_string());
        while let Some(current) = cursor {
            if !seen.insert(current.clone()) {
                break;
            }
            let Some(stored) = self.by_id(&current)? else {
                break;
            };
            cursor = stored.causation_id.clone();
            chain.push(stored);
        }
        chain.reverse();
        Ok(chain)
    }

    pub fn len(&self) -> Result<i64, EventLogError> {
        let conn = self.lock();
        Ok(conn.query_row("SELECT COUNT(*) FROM events", params![], |row| row.get(0))?)
    }

    pub fn is_empty(&self) -> Result<bool, EventLogError> {
        Ok(self.len()? == 0)
    }

    fn query(
        &self,
        sql: &str,
        args: impl rusqlite::Params,
    ) -> Result<Vec<StoredEvent>, EventLogError> {
        let conn = self.lock();
        let mut statement = conn.prepare(sql)?;
        let rows = statement.query_map(args, StoredEvent::from_row)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }
}

/// Emission is fire-and-forget, as [`EventSink`] requires: a subsystem
/// announcing a fact must not fail because of what a listener does with it.
///
/// A failed write is therefore reported on stderr and dropped. That is a real
/// limitation, stated rather than hidden — a caller who needs the write to be
/// part of its own success calls [`SqliteEventLog::append`].
impl EventSink for SqliteEventLog {
    fn emit(&self, event: &Event) {
        if let Err(err) = self.append(event) {
            eprintln!(
                "[adam-eventlog] dropped {} ({}): {err}",
                event.kind.as_str(),
                event.id
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adam_protocol::{EventKind, PayloadValue, SubjectType};
    use std::collections::BTreeMap;

    fn event(kind: EventKind, correlation: &str) -> Event {
        let mut payload = BTreeMap::new();
        payload.insert("k".to_string(), PayloadValue::Text("v".into()));
        Event::new(
            kind.emitters()[0],
            kind,
            "33333333-3333-4333-8333-333333333333",
            SubjectType::Genome,
            correlation,
            payload,
            "adam:test",
        )
    }

    fn log() -> SqliteEventLog {
        SqliteEventLog::open(":memory:").expect("open")
    }

    #[test]
    fn appends_and_reads_back() {
        let log = log();
        let e = event(EventKind::GenomeCommitted, "corr-1");
        log.append(&e).expect("append");
        let all = log.all().expect("all");
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].id, e.id);
        assert_eq!(all[0].correlation_id, "corr-1");
        assert_eq!(all[0].occurred_at, e.occurred_at.as_str());
        assert_eq!(all[0].content_hash.len(), 64, "content hash must be stored");
    }

    #[test]
    fn round_trips_the_full_document() {
        let log = log();
        let e = event(EventKind::BeliefUpdated, "corr-1");
        log.append(&e).expect("append");
        let stored = log.by_id(&e.id).expect("query").expect("present");
        let rebuilt = stored.event().expect("deserialize");
        assert_eq!(rebuilt.id, e.id);
        assert_eq!(rebuilt.payload, e.payload);
        assert_eq!(rebuilt.correlation_id, e.correlation_id);
        assert_eq!(rebuilt.kind, e.kind);
        assert_eq!(rebuilt.actor, e.actor);
    }

    #[test]
    fn preserves_insertion_order_within_a_turn() {
        let log = log();
        let kinds = [
            EventKind::MemoryConsolidated,
            EventKind::BeliefUpdated,
            EventKind::MutationProposed,
            EventKind::GenomeCommitted,
        ];
        for kind in kinds {
            log.append(&event(kind, "corr-turn")).expect("append");
        }
        log.append(&event(EventKind::BeliefUpdated, "other-turn"))
            .expect("append");

        let turn = log.turn("corr-turn").expect("turn");
        let observed: Vec<&str> = turn.iter().map(|e| e.kind.as_str()).collect();
        assert_eq!(
            observed,
            vec![
                "MemoryConsolidated",
                "BeliefUpdated",
                "MutationProposed",
                "GenomeCommitted"
            ]
        );
    }

    #[test]
    fn rejects_a_duplicate_id() {
        let log = log();
        let e = event(EventKind::GenomeCommitted, "corr-1");
        log.append(&e).expect("first");
        let err = log.append(&e).expect_err("second must fail");
        assert!(matches!(err, EventLogError::Duplicate { .. }));
        assert_eq!(log.len().expect("len"), 1);
    }

    #[test]
    fn refuses_update_and_delete() {
        let log = log();
        log.append(&event(EventKind::GenomeCommitted, "corr-1"))
            .expect("append");
        let conn = log.lock();
        assert!(
            conn.execute("UPDATE events SET kind = 'forged'", params![])
                .is_err(),
            "UPDATE must be refused"
        );
        assert!(
            conn.execute("DELETE FROM events", params![]).is_err(),
            "DELETE must be refused"
        );
    }

    #[test]
    fn walks_a_causal_chain_backwards() {
        let log = log();
        let root = event(EventKind::MemoryConsolidated, "corr-1");
        let middle = event(EventKind::BeliefUpdated, "corr-1").caused_by(root.id.clone());
        let leaf = event(EventKind::MutationProposed, "corr-1").caused_by(middle.id.clone());
        for e in [&root, &middle, &leaf] {
            log.append(e).expect("append");
        }

        let chain = log.causal_chain(&leaf.id).expect("chain");
        let ids: Vec<&str> = chain.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(
            ids,
            vec![root.id.as_str(), middle.id.as_str(), leaf.id.as_str()]
        );

        let forward = log.caused_by(&root.id).expect("caused_by");
        assert_eq!(forward.len(), 1);
        assert_eq!(forward[0].id, middle.id);
    }

    #[test]
    fn survives_reopening_the_file() {
        let dir = std::env::temp_dir().join(format!("adam-eventlog-{}", nonce()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let file = dir.join("events.db");
        let path = file.to_str().expect("utf8");

        let id = {
            let log = SqliteEventLog::open(path).expect("open");
            let e = event(EventKind::GenomeCommitted, "corr-restart");
            log.append(&e).expect("append");
            e.id
        };

        let reopened = SqliteEventLog::open(path).expect("reopen");
        assert!(
            reopened.by_id(&id).expect("query").is_some(),
            "event must survive closing the log"
        );
        assert_eq!(reopened.len().expect("len"), 1);
        drop(reopened);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn works_as_an_event_sink() {
        let log = std::sync::Arc::new(log());
        let sink: std::sync::Arc<dyn EventSink> = log.clone();
        sink.emit(&event(EventKind::GenomeCommitted, "corr-sink"));
        assert_eq!(log.turn("corr-sink").expect("turn").len(), 1);
    }

    fn nonce() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    }
}
