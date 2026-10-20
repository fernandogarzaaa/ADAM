//! The tree store: ADAM's [`MemoryStore`] plus the layer's sidecar
//! tables in the same SQLite file (spec section 2), and the
//! summarizer lease (spec section 6).

use adam_memory::{MemoryId, MemoryStore};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension, Row};

use crate::config::TreeConfig;
use crate::error::TreeError;
use crate::model::{Lease, NodeState, TreeNode};

const NODE_COLUMNS: &str = "memory_id, level, thread_key, contact_key, channel, external_id, \
    span_start, span_end, covered_count, token_estimate, pinned, state";

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS tree_nodes (
    memory_id       TEXT PRIMARY KEY REFERENCES memories(id),
    level           INTEGER NOT NULL,
    thread_key      TEXT NOT NULL,
    contact_key     TEXT NOT NULL,
    channel         TEXT,
    external_id     TEXT,
    span_start      TEXT NOT NULL,
    span_end        TEXT NOT NULL,
    covered_count   INTEGER NOT NULL DEFAULT 0,
    token_estimate  INTEGER NOT NULL,
    pinned          INTEGER NOT NULL DEFAULT 0,
    state           TEXT NOT NULL DEFAULT 'open'
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_tree_external
    ON tree_nodes(channel, external_id) WHERE external_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_tree_thread ON tree_nodes(thread_key, level, span_start);
CREATE INDEX IF NOT EXISTS idx_tree_contact ON tree_nodes(contact_key, level);
CREATE TABLE IF NOT EXISTS tree_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
";

/// Tree-of-summaries layer over one ADAM database file.
///
/// The sidecar tables live in the same file as ADAM's tables, opened
/// through a second connection: ADAM's public API offers no way to
/// run the layer's own SQL through its connection. WAL journal mode
/// is enabled on the file — a persistent property of the database,
/// so ADAM's connection writes in WAL too; a caller's deployment
/// should still prefer the topology rule from spec section 6 (all
/// writers through one adam-mcp endpoint). With `":memory:"` the
/// two connections are necessarily separate databases; use a file
/// path for anything real.
pub struct TreeStore {
    pub(crate) store: MemoryStore,
    pub(crate) conn: Connection,
    pub(crate) config: TreeConfig,
}

impl TreeStore {
    /// Open (or create) a tree store over the ADAM database at
    /// `path`, with default configuration.
    pub fn open(path: &str) -> Result<Self, TreeError> {
        Self::with_config(path, TreeConfig::default())
    }

    /// Open (or create) a tree store with explicit configuration.
    /// Creates the sidecar tables if absent; pre-existing ADAM
    /// records are untouched and tree-invisible until adopted.
    pub fn with_config(path: &str, config: TreeConfig) -> Result<Self, TreeError> {
        let store = MemoryStore::open(path)?;
        let conn = Connection::open(path)?;
        conn.busy_timeout(config.busy_timeout)?;
        // WAL is a property of the database file, not the connection:
        // enabling it here covers ADAM's connection as well. On an
        // in-memory database this reports "memory" and changes
        // nothing.
        let _journal: String = conn.query_row("PRAGMA journal_mode=WAL", [], |row| row.get(0))?;
        let tree = Self {
            store,
            conn,
            config,
        };
        tree.init_schema()?;
        Ok(tree)
    }

    fn init_schema(&self) -> Result<(), TreeError> {
        self.conn.execute_batch(SCHEMA)?;
        Ok(())
    }

    /// The underlying ADAM store: the escape hatch for non-tree
    /// records and ADAM-level operations (decay, conflict).
    pub fn memory_store(&self) -> &MemoryStore {
        &self.store
    }

    /// This store's configuration.
    pub fn config(&self) -> &TreeConfig {
        &self.config
    }

    /// The sidecar row for one memory, if it participates in the tree.
    pub fn node(&self, id: MemoryId) -> Result<Option<TreeNode>, TreeError> {
        self.conn
            .query_row(
                &format!("SELECT {NODE_COLUMNS} FROM tree_nodes WHERE memory_id = ?1"),
                params![id.to_string()],
                row_to_node,
            )
            .optional()
            .map_err(TreeError::from)?
            .transpose()
    }

    /// The sidecar row owning one `(channel, external_id)` pair, if
    /// any. This is the leaf idempotency lookup.
    pub(crate) fn node_by_external(
        &self,
        channel: &str,
        external_id: &str,
    ) -> Result<Option<TreeNode>, TreeError> {
        self.conn
            .query_row(
                &format!(
                    "SELECT {NODE_COLUMNS} FROM tree_nodes \
                     WHERE channel = ?1 AND external_id = ?2"
                ),
                params![channel, external_id],
                row_to_node,
            )
            .optional()
            .map_err(TreeError::from)?
            .transpose()
    }

    /// Sidecar rows matching every given filter (`None` = no filter
    /// on that column). Ordering is the caller's job.
    pub(crate) fn query_nodes(
        &self,
        thread_key: Option<&str>,
        contact_key: Option<&str>,
        level: Option<u32>,
        state: Option<NodeState>,
    ) -> Result<Vec<TreeNode>, TreeError> {
        let sql = format!(
            "SELECT {NODE_COLUMNS} FROM tree_nodes
             WHERE (?1 IS NULL OR thread_key = ?1)
               AND (?2 IS NULL OR contact_key = ?2)
               AND (?3 IS NULL OR level = ?3)
               AND (?4 IS NULL OR state = ?4)"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(
            params![
                thread_key,
                contact_key,
                level.map(|l| l as i64),
                state.map(|s| s.as_str())
            ],
            row_to_node,
        )?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row??);
        }
        Ok(out)
    }

    pub(crate) fn insert_node(&self, node: &TreeNode) -> Result<(), TreeError> {
        self.conn.execute(
            "INSERT INTO tree_nodes (memory_id, level, thread_key, contact_key, channel,
                    external_id, span_start, span_end, covered_count, token_estimate,
                    pinned, state)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                node.memory_id.to_string(),
                node.level as i64,
                node.thread_key,
                node.contact_key,
                node.channel,
                node.external_id,
                node.span_start.to_rfc3339(),
                node.span_end.to_rfc3339(),
                node.covered_count as i64,
                node.token_estimate as i64,
                node.pinned as i64,
                node.state.as_str(),
            ],
        )?;
        Ok(())
    }

    pub(crate) fn set_node_state(&self, id: MemoryId, state: NodeState) -> Result<(), TreeError> {
        self.conn.execute(
            "UPDATE tree_nodes SET state = ?1 WHERE memory_id = ?2",
            params![state.as_str(), id.to_string()],
        )?;
        Ok(())
    }

    /// Try to acquire the single summarizer lease for `holder`.
    /// Returns `None` when another holder's lease is still live.
    ///
    /// Acquisition is one conditional UPDATE on the expiry row: it
    /// only lands when the stored expiry is in the past, so two
    /// racing acquirers cannot both win (spec section 6). A crashed
    /// summarizer's lease simply expires and the next scheduled pass
    /// proceeds; the adoption sweep makes the resumed pass correct.
    pub fn try_acquire_lease(&self, holder: &str) -> Result<Option<Lease>, TreeError> {
        self.conn.execute(
            "INSERT OR IGNORE INTO tree_meta (key, value) \
             VALUES ('summarizer_lease_expiry_ms', '0')",
            [],
        )?;
        self.conn.execute(
            "INSERT OR IGNORE INTO tree_meta (key, value) \
             VALUES ('summarizer_lease_holder', '')",
            [],
        )?;
        let now_ms = Utc::now().timestamp_millis();
        let expiry_ms = now_ms + self.config.lease_ttl.as_millis() as i64;
        let affected = self.conn.execute(
            "UPDATE tree_meta SET value = ?1
             WHERE key = 'summarizer_lease_expiry_ms' AND CAST(value AS INTEGER) <= ?2",
            params![expiry_ms.to_string(), now_ms],
        )?;
        if affected == 0 {
            return Ok(None);
        }
        self.conn.execute(
            "UPDATE tree_meta SET value = ?1 WHERE key = 'summarizer_lease_holder'",
            params![holder],
        )?;
        Ok(Some(Lease {
            holder: holder.to_string(),
            expires_at: DateTime::from_timestamp_millis(expiry_ms).unwrap_or_else(Utc::now),
        }))
    }

    /// The current lease holder and expiry, if a lease was ever taken.
    pub fn current_lease(&self) -> Result<Option<Lease>, TreeError> {
        let holder: Option<String> = self
            .conn
            .query_row(
                "SELECT value FROM tree_meta WHERE key = 'summarizer_lease_holder'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        let expiry: Option<String> = self
            .conn
            .query_row(
                "SELECT value FROM tree_meta WHERE key = 'summarizer_lease_expiry_ms'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        match (holder, expiry) {
            (Some(holder), Some(expiry)) => {
                let expiry_ms: i64 = expiry.parse().map_err(|_| {
                    TreeError::Corrupt(format!("lease expiry '{expiry}' is not an integer"))
                })?;
                Ok(Some(Lease {
                    holder,
                    expires_at: DateTime::from_timestamp_millis(expiry_ms).unwrap_or_else(Utc::now),
                }))
            }
            _ => Ok(None),
        }
    }

    /// Extend a held lease by the configured TTL. Returns the renewed
    /// lease, or `None` when the caller no longer holds it.
    pub fn renew_lease(&self, lease: &Lease) -> Result<Option<Lease>, TreeError> {
        let now_ms = Utc::now().timestamp_millis();
        let expiry_ms = now_ms + self.config.lease_ttl.as_millis() as i64;
        let affected = self.conn.execute(
            "UPDATE tree_meta SET value = ?1
             WHERE key = 'summarizer_lease_expiry_ms'
               AND (SELECT value FROM tree_meta WHERE key = 'summarizer_lease_holder') = ?2",
            params![expiry_ms.to_string(), lease.holder],
        )?;
        if affected == 0 {
            return Ok(None);
        }
        Ok(Some(Lease {
            holder: lease.holder.clone(),
            expires_at: DateTime::from_timestamp_millis(expiry_ms).unwrap_or_else(Utc::now),
        }))
    }

    /// Release a held lease so the next pass can proceed immediately.
    /// Releasing someone else's lease is a no-op.
    pub fn release_lease(&self, lease: &Lease) -> Result<(), TreeError> {
        self.conn.execute(
            "UPDATE tree_meta SET value = '0'
             WHERE key = 'summarizer_lease_expiry_ms'
               AND (SELECT value FROM tree_meta WHERE key = 'summarizer_lease_holder') = ?1",
            params![lease.holder],
        )?;
        Ok(())
    }
}

fn row_to_node(row: &Row) -> rusqlite::Result<Result<TreeNode, TreeError>> {
    Ok((|| -> Result<TreeNode, TreeError> {
        let id: String = row.get(0)?;
        let level: i64 = row.get(1)?;
        let thread_key: String = row.get(2)?;
        let contact_key: String = row.get(3)?;
        let channel: Option<String> = row.get(4)?;
        let external_id: Option<String> = row.get(5)?;
        let span_start: String = row.get(6)?;
        let span_end: String = row.get(7)?;
        let covered_count: i64 = row.get(8)?;
        let token_estimate: i64 = row.get(9)?;
        let pinned: i64 = row.get(10)?;
        let state: String = row.get(11)?;

        Ok(TreeNode {
            memory_id: id.parse().map_err(|e| {
                TreeError::Corrupt(format!("tree node id '{id}' is not a valid UUID: {e}"))
            })?,
            level: u32::try_from(level).map_err(|_| {
                TreeError::Corrupt(format!("tree node level {level} is out of range"))
            })?,
            thread_key,
            contact_key,
            channel,
            external_id,
            span_start: parse_rfc3339(&span_start)?,
            span_end: parse_rfc3339(&span_end)?,
            covered_count: usize::try_from(covered_count).map_err(|_| {
                TreeError::Corrupt(format!("covered_count {covered_count} is out of range"))
            })?,
            token_estimate: usize::try_from(token_estimate).map_err(|_| {
                TreeError::Corrupt(format!("token_estimate {token_estimate} is out of range"))
            })?,
            pinned: pinned != 0,
            state: NodeState::parse(&state).ok_or_else(|| {
                TreeError::Corrupt(format!("tree node state '{state}' is invalid"))
            })?,
        })
    })())
}

fn parse_rfc3339(s: &str) -> Result<DateTime<Utc>, TreeError> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| {
            TreeError::Corrupt(format!("stored timestamp '{s}' is not valid RFC3339: {e}"))
        })
}
