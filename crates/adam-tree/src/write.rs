//! Leaf writes, pinning, and crash-recovery adoption
//! (spec sections 3, 5, 7).

use adam_memory::{MemoryId, MemoryKind, MemoryRecord, Provenance, RelationKind};
use rusqlite::params;

use crate::error::TreeError;
use crate::model::{
    summary_level, token_estimate, AdoptionReport, Embedder, LeafOutcome, LeafWrite, NodeState,
    TreeNode,
};
use crate::store::TreeStore;

impl TreeStore {
    /// Store one conversational leaf (spec `write_leaf`).
    ///
    /// Idempotent on `(channel, external_id)`: a retry returns the
    /// existing leaf's id with `duplicate: true` and writes nothing.
    /// This closes ADAM's write-idempotency gap at the layer level,
    /// using the same duplicate-rejection idea as adam-eventlog.
    ///
    /// The write is claim-first. The record's id is known before it
    /// is stored (`MemoryRecord::new` mints it), so the sidecar row
    /// is inserted first, in the `pending` state, carrying the real
    /// thread key. The unique `(channel, external_id)` index is the
    /// single atomic gate: only the claim's owner goes on to store
    /// a record, so at most one record per pair is ever stored by a
    /// writer that completes, and a stored leaf's thread placement
    /// was fixed before its record existed. (The claim insert
    /// relies on SQLite foreign-key enforcement staying off on the
    /// sidecar connection, the default: the row references a
    /// `memories` id that lands a moment later. Enabling FK
    /// enforcement would make claims fail loudly, never silently
    /// corrupt.)
    ///
    /// The previous order (record first, sidecar second) had a
    /// window between the two writes: an adoption sweep in that
    /// window adopted the orphan record into the fallback thread
    /// bucket, and the channel retry then stored a second record
    /// for the same pair. Claim-first removes the window: a crash
    /// leaves only a pending claim or a pending claim plus its
    /// record, and both shapes settle deterministically (see
    /// [`TreeStore::settle_existing`] and [`TreeStore::adopt_orphans`]).
    pub fn write_leaf(
        &self,
        embedder: &dyn Embedder,
        input: &LeafWrite,
    ) -> Result<LeafOutcome, TreeError> {
        if input.channel.is_empty()
            || input.contact_key.is_empty()
            || input.thread_key.is_empty()
            || input.external_id.is_empty()
        {
            return Err(TreeError::InvalidInput(
                "channel, contact_key, thread_key, and external_id must be non-empty".to_string(),
            ));
        }
        // Fast path: the pair is already owned. Settling also heals
        // whatever a crashed earlier attempt left behind.
        if let Some(existing) = self.node_by_external(&input.channel, &input.external_id)? {
            if let Some(owner) = self.settle_existing(&existing, true)? {
                return Ok(LeafOutcome {
                    memory_id: owner.memory_id,
                    duplicate: true,
                });
            }
        }
        let embedding = embedder.embed(&input.content)?;
        let mut record = MemoryRecord::new(
            MemoryKind::Episodic,
            input.content.clone(),
            embedding,
            input.confidence,
            Provenance {
                origin: format!("channel:{}:{}", input.channel, input.contact_key),
                evidence: vec![input.external_id.clone()],
            },
            0.0,
        );
        if let Some(at) = input.at {
            record.created_at = at;
            record.last_accessed_at = at;
        }
        let node = TreeNode {
            memory_id: record.id,
            level: 0,
            thread_key: input.thread_key.clone(),
            contact_key: input.contact_key.clone(),
            channel: Some(input.channel.clone()),
            external_id: Some(input.external_id.clone()),
            span_start: record.created_at,
            span_end: record.created_at,
            covered_count: 0,
            token_estimate: token_estimate(&record.content),
            pinned: input.pinned,
            state: NodeState::Pending,
        };
        match self.insert_node(&node) {
            Ok(()) => {}
            Err(err) => {
                // Lost the claim race against a concurrent writer of
                // the same (channel, external_id): the unique index
                // rejected our row, and their leaf is the canonical
                // one. Our record was never stored, so this path
                // leaves no orphan behind.
                if let Some(existing) =
                    self.node_by_external(&input.channel, &input.external_id)?
                {
                    if let Some(owner) = self.settle_existing(&existing, false)? {
                        return Ok(LeafOutcome {
                            memory_id: owner.memory_id,
                            duplicate: true,
                        });
                    }
                }
                return Err(err);
            }
        }
        // We own the claim; only now may the record be stored.
        if let Err(err) = self.store.store(&record) {
            // Release the claim so the pair's retry starts clean.
            let _ = self.delete_node(record.id);
            return Err(err.into());
        }
        if !self.finalize_claim(record.id)? {
            // Residual window: a retry settled our claim as stale
            // in the instant before the record landed (its guarded
            // delete still saw no record), so the claim row is gone.
            // Re-claim now that the record exists. If another writer
            // owns the pair instead, their leaf is canonical and
            // this record stays an unadopted orphan, the same
            // race-loser rule adoption already applies.
            let mut open_node = node.clone();
            open_node.state = NodeState::Open;
            match self.insert_node(&open_node) {
                Ok(()) => {}
                Err(err) => {
                    if let Some(existing) =
                        self.node_by_external(&input.channel, &input.external_id)?
                    {
                        if let Some(owner) = self.settle_existing(&existing, false)? {
                            return Ok(LeafOutcome {
                                memory_id: owner.memory_id,
                                duplicate: true,
                            });
                        }
                    }
                    return Err(err);
                }
            }
        }
        Ok(LeafOutcome {
            memory_id: record.id,
            duplicate: false,
        })
    }

    /// Resolve a sidecar row that already owns a leaf's
    /// `(channel, external_id)` pair, healing the states a crashed
    /// claim-first write can leave behind:
    ///
    /// - a settled row (open or covered) is the owner, returned
    ///   as-is;
    /// - a pending row whose record exists is a write that crashed
    ///   between storing its record and finalizing its claim:
    ///   finalize it and return it as the owner;
    /// - a pending row with no record is a write that crashed
    ///   between claiming and storing. With `drop_stale` (the
    ///   `write_leaf` fast path, where the caller is that pair's
    ///   retry) the stale claim is deleted and `None` is returned
    ///   so the caller writes fresh. Without it (a claim this call
    ///   just raced and lost) the row may belong to a live writer,
    ///   so it is returned as the owner untouched.
    ///
    /// Returns `None` only when the pair is now unowned.
    fn settle_existing(
        &self,
        node: &TreeNode,
        drop_stale: bool,
    ) -> Result<Option<TreeNode>, TreeError> {
        if node.state != NodeState::Pending {
            return Ok(Some(node.clone()));
        }
        if self.store.get(node.memory_id)?.is_some() {
            self.finalize_claim(node.memory_id)?;
            let mut settled = node.clone();
            settled.state = NodeState::Open;
            return Ok(Some(settled));
        }
        if !drop_stale {
            return Ok(Some(node.clone()));
        }
        if self.delete_stale_claim(node.memory_id)? {
            return Ok(None);
        }
        // The guarded delete refused: the record landed
        // concurrently. Whatever owns the pair now is the owner.
        match node.channel.as_deref().zip(node.external_id.as_deref()) {
            Some((channel, external_id)) => self.node_by_external(channel, external_id),
            None => Ok(None),
        }
    }

    /// Pin or unpin a tree node (spec `set_pinned`). Pinned leaves
    /// are never summarized and never demoted in recall. Pinning a
    /// covered leaf is allowed; it does not rewrite the summary that
    /// already covered it.
    pub fn set_pinned(&self, id: MemoryId, pinned: bool) -> Result<(), TreeError> {
        let affected = self.conn.execute(
            "UPDATE tree_nodes SET pinned = ?1 WHERE memory_id = ?2",
            params![pinned as i64, id.to_string()],
        )?;
        if affected == 0 {
            return Err(TreeError::NotInTree(id));
        }
        Ok(())
    }

    /// Repair partial writes left by crashed writers (spec
    /// section 3), so a crash at any pass step self-heals on the next
    /// run instead of corrupting the tree:
    ///
    /// 0. Pending leaf claims whose record landed are finalized to
    ///    open (a claim-first write that crashed between storing its
    ///    record and finalizing). A pending claim with no record is
    ///    left alone: it belongs to a live writer or to a write that
    ///    crashed before storing, and the pair's next `write_leaf`
    ///    settles it.
    /// 1. Episodic records with a `channel:` origin and no sidecar
    ///    row are adopted as orphan leaves (thread key falls back to
    ///    the date-bucketed slug from spec section 2). With
    ///    claim-first writes this only fires for records written
    ///    outside `write_leaf` (the escape hatch, pre-claim data).
    /// 2. Semantic records with a `summary:tree:L<n>` origin and no
    ///    sidecar row get their row rebuilt from their evidence, in
    ///    ascending level order so children exist before parents.
    /// 3. For every tree summary, each evidence child gets its
    ///    `Generalizes` edge written if missing and its sidecar state
    ///    flipped to covered if still open.
    ///
    /// Coverage is derived state: a child counts as covered only when
    /// its state is `covered` AND the edge exists. This sweep runs at
    /// the start of every summarization run and is safe to call on
    /// its own at any time.
    pub fn adopt_orphans(&self) -> Result<AdoptionReport, TreeError> {
        let mut report = AdoptionReport::default();

        // 0. Finalize claims whose records landed.
        let pending = self.query_nodes(None, None, None, Some(NodeState::Pending))?;
        for node in &pending {
            if self.store.get(node.memory_id)?.is_some()
                && self.finalize_claim(node.memory_id)?
            {
                report.claims_finalized += 1;
            }
        }

        let records = self.store.all()?;

        // 1. Orphan leaves.
        for record in &records {
            if record.kind != MemoryKind::Episodic {
                continue;
            }
            let Some(rest) = record.provenance.origin.strip_prefix("channel:") else {
                continue;
            };
            if self.node(record.id)?.is_some() {
                continue;
            }
            let (channel, contact_key) = match rest.split_once(':') {
                Some((channel, contact)) => (channel.to_string(), contact.to_string()),
                None => (rest.to_string(), String::new()),
            };
            let external_id = record.provenance.evidence.first().cloned();
            if let Some(ext) = &external_id {
                if self.node_by_external(&channel, ext)?.is_some() {
                    // A different record already owns this
                    // (channel, external_id): this record is the
                    // race-loser of a concurrent write_leaf.
                    continue;
                }
            }
            let node = TreeNode {
                memory_id: record.id,
                level: 0,
                thread_key: format!("{contact_key}/{}", record.created_at.format("%Y-%m")),
                contact_key,
                channel: Some(channel),
                external_id,
                span_start: record.created_at,
                span_end: record.created_at,
                covered_count: 0,
                token_estimate: token_estimate(&record.content),
                pinned: false,
                state: NodeState::Open,
            };
            match self.insert_node(&node) {
                Ok(()) => report.leaves_adopted += 1,
                Err(err) => {
                    if self.node(record.id)?.is_some() {
                        continue;
                    }
                    return Err(err);
                }
            }
        }

        // 2 + 3. Orphan summaries (ascending level), then coverage
        // repair for every tree summary.
        let mut summaries: Vec<&MemoryRecord> = records
            .iter()
            .filter(|r| r.kind == MemoryKind::Semantic)
            .filter(|r| summary_level(&r.provenance.origin).is_some())
            .collect();
        summaries.sort_by_key(|r| summary_level(&r.provenance.origin).unwrap_or(0));
        for record in summaries {
            let level = summary_level(&record.provenance.origin).unwrap_or(0);
            if self.node(record.id)?.is_none() {
                let child_ids: Vec<MemoryId> = record
                    .provenance
                    .evidence
                    .iter()
                    .filter_map(|e| e.parse().ok())
                    .collect();
                let mut children: Vec<TreeNode> = Vec::new();
                for child_id in &child_ids {
                    if let Some(child) = self.node(*child_id)? {
                        children.push(child);
                    }
                }
                if children.is_empty() {
                    // A summary whose children are unknown cannot be
                    // placed in the tree; leave it for the operator
                    // rather than guessing.
                    continue;
                }
                let span_start = children.iter().map(|n| n.span_start).min().unwrap();
                let span_end = children.iter().map(|n| n.span_end).max().unwrap();
                let contact_key = children[0].contact_key.clone();
                let thread_key = if children
                    .iter()
                    .all(|n| n.thread_key == children[0].thread_key)
                {
                    children[0].thread_key.clone()
                } else {
                    contact_key.clone()
                };
                self.insert_node(&TreeNode {
                    memory_id: record.id,
                    level,
                    thread_key,
                    contact_key,
                    channel: None,
                    external_id: None,
                    span_start,
                    span_end,
                    covered_count: children.len(),
                    token_estimate: token_estimate(&record.content),
                    pinned: false,
                    state: NodeState::Open,
                })?;
                report.summaries_adopted += 1;
            }
            report.children_flipped += self.repair_coverage(record)?;
        }

        Ok(report)
    }

    /// Ensure every evidence child of `summary` has its `Generalizes`
    /// edge and is marked covered. Returns how many children flipped
    /// from open to covered.
    fn repair_coverage(&self, summary: &MemoryRecord) -> Result<usize, TreeError> {
        let mut flipped = 0;
        for evidence in &summary.provenance.evidence {
            let Ok(child_id) = evidence.parse::<MemoryId>() else {
                continue;
            };
            let Some(child) = self.node(child_id)? else {
                continue;
            };
            let outgoing = self.store.related_outgoing(child_id)?;
            let has_edge = outgoing
                .iter()
                .any(|(rec, kind)| rec.id == summary.id && *kind == RelationKind::Generalizes);
            if !has_edge {
                self.store
                    .relate(child_id, summary.id, RelationKind::Generalizes)?;
            }
            if child.state == NodeState::Open {
                self.set_node_state(child_id, NodeState::Covered)?;
                flipped += 1;
            }
        }
        Ok(flipped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::HashEmbedder;
    use chrono::{DateTime, Duration, Utc};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    const CONTACT: &str = "alex@example.com";
    const THREAD: &str = "alex@example.com/contract-review";

    fn temp_db(tag: &str) -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let path = std::env::temp_dir().join(format!(
            "adam-tree-write-{tag}-{}-{n}.db",
            std::process::id()
        ));
        cleanup(&path);
        path
    }

    fn cleanup(path: &Path) {
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
        }
    }

    fn open(tag: &str) -> (TreeStore, PathBuf) {
        let path = temp_db(tag);
        let store = TreeStore::open(path.to_str().unwrap()).unwrap();
        (store, path)
    }

    /// A leaf record and its pending claim, built exactly as
    /// `write_leaf` builds them, so a test can stop the write at a
    /// chosen step and drive the remaining steps by hand. That is
    /// what makes the crash interleavings deterministic: no
    /// threads, no sleeps, no timing luck.
    fn claim_and_record(external_id: &str, at: DateTime<Utc>) -> (MemoryRecord, TreeNode) {
        let mut record = MemoryRecord::new(
            MemoryKind::Episodic,
            format!("content for {external_id}"),
            vec![0.1, 0.9],
            1.0,
            Provenance {
                origin: format!("channel:email:{CONTACT}"),
                evidence: vec![external_id.to_string()],
            },
            0.0,
        );
        record.created_at = at;
        record.last_accessed_at = at;
        let node = TreeNode {
            memory_id: record.id,
            level: 0,
            thread_key: THREAD.to_string(),
            contact_key: CONTACT.to_string(),
            channel: Some("email".to_string()),
            external_id: Some(external_id.to_string()),
            span_start: at,
            span_end: at,
            covered_count: 0,
            token_estimate: token_estimate(&record.content),
            pinned: false,
            state: NodeState::Pending,
        };
        (record, node)
    }

    fn leaf_write(external_id: &str, content: &str, at: DateTime<Utc>) -> LeafWrite {
        let mut input = LeafWrite::new("email", CONTACT, THREAD, external_id, content);
        input.at = Some(at);
        input
    }

    fn records_for(store: &TreeStore, external_id: &str) -> usize {
        store
            .store
            .all()
            .unwrap()
            .iter()
            .filter(|r| r.provenance.evidence.iter().any(|e| e == external_id))
            .count()
    }

    /// Crash window 1: claim inserted, record never stored.
    /// Adoption must not invent a fallback-bucket leaf out of the
    /// claim, and the retry must settle the stale claim and land
    /// the leaf, once, in its real thread.
    #[test]
    fn crash_after_claim_retry_writes_fresh_in_real_thread() {
        let (store, path) = open("claim-crash");
        let at = Utc::now() - Duration::hours(1);
        let (record, claim) = claim_and_record("ext-claim", at);
        store.insert_node(&claim).unwrap();

        let adopted = store.adopt_orphans().unwrap();
        assert_eq!(adopted.leaves_adopted, 0);
        assert_eq!(adopted.claims_finalized, 0);
        let node = store.node(record.id).unwrap().unwrap();
        assert_eq!(node.state, NodeState::Pending);
        assert_eq!(node.thread_key, THREAD);

        let outcome = store
            .write_leaf(&HashEmbedder, &leaf_write("ext-claim", &record.content, at))
            .unwrap();
        assert!(!outcome.duplicate);
        assert_eq!(records_for(&store, "ext-claim"), 1);
        let node = store.node(outcome.memory_id).unwrap().unwrap();
        assert_eq!(node.thread_key, THREAD);
        assert_eq!(node.state, NodeState::Open);
        store.assert_partition().unwrap();
        cleanup(&path);
    }

    /// Crash window 2: claim inserted and record stored, claim
    /// never finalized. Adoption finalizes it in its real thread
    /// (no fallback bucket, no thread guessing), and the retry is
    /// a duplicate of the same single record.
    #[test]
    fn crash_after_store_adoption_finalizes_and_retry_is_duplicate() {
        let (store, path) = open("store-crash");
        let at = Utc::now() - Duration::hours(2);
        let (record, claim) = claim_and_record("ext-stored", at);
        store.insert_node(&claim).unwrap();
        store.store.store(&record).unwrap();

        let adopted = store.adopt_orphans().unwrap();
        assert_eq!(adopted.claims_finalized, 1);
        assert_eq!(adopted.leaves_adopted, 0);
        let node = store.node(record.id).unwrap().unwrap();
        assert_eq!(node.state, NodeState::Open);
        assert_eq!(node.thread_key, THREAD);

        let outcome = store
            .write_leaf(&HashEmbedder, &leaf_write("ext-stored", &record.content, at))
            .unwrap();
        assert!(outcome.duplicate);
        assert_eq!(outcome.memory_id, record.id);
        assert_eq!(records_for(&store, "ext-stored"), 1);
        store.assert_partition().unwrap();
        cleanup(&path);
    }

    /// The interleaving that duplicated leaves before claim-first
    /// writes (PR #23's documented race): a writer's record lands,
    /// an adoption sweep from a second instance runs before the
    /// write completes, then the channel retries the pair. Driven
    /// step by step; exactly one record, in the real thread.
    #[test]
    fn adoption_window_interleaving_lands_one_leaf_in_real_thread() {
        let path = temp_db("window");
        let a = TreeStore::open(path.to_str().unwrap()).unwrap();
        let b = TreeStore::open(path.to_str().unwrap()).unwrap();
        let at = Utc::now() - Duration::hours(3);
        let (record, claim) = claim_and_record("ext-window", at);

        // Writer A: claim and record in, finalize not yet run.
        a.insert_node(&claim).unwrap();
        a.store.store(&record).unwrap();

        // The sweep runs inside the window.
        let adopted = b.adopt_orphans().unwrap();
        assert_eq!(adopted.leaves_adopted, 0);
        assert_eq!(adopted.claims_finalized, 1);

        // The channel retry of the same pair.
        let outcome = b
            .write_leaf(&HashEmbedder, &leaf_write("ext-window", &record.content, at))
            .unwrap();
        assert!(outcome.duplicate);
        assert_eq!(outcome.memory_id, record.id);
        assert_eq!(records_for(&b, "ext-window"), 1);
        let node = b.node(record.id).unwrap().unwrap();
        assert_eq!(node.thread_key, THREAD);
        assert_eq!(node.state, NodeState::Open);
        b.assert_partition().unwrap();
        cleanup(&path);
    }

    /// The unique `(channel, external_id)` index is the claim gate:
    /// a second claimant with a different record id cannot take a
    /// claimed pair over.
    #[test]
    fn second_claim_for_same_pair_is_rejected_by_the_index() {
        let path = temp_db("gate");
        let a = TreeStore::open(path.to_str().unwrap()).unwrap();
        let b = TreeStore::open(path.to_str().unwrap()).unwrap();
        let at = Utc::now();
        let (_, claim_a) = claim_and_record("ext-gate", at);
        let (_, claim_b) = claim_and_record("ext-gate", at);
        a.insert_node(&claim_a).unwrap();
        assert!(b.insert_node(&claim_b).is_err());
        let owner = b.node_by_external("email", "ext-gate").unwrap().unwrap();
        assert_eq!(owner.memory_id, claim_a.memory_id);
        cleanup(&path);
    }
}
