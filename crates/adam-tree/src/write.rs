//! Leaf writes, pinning, and crash-recovery adoption
//! (spec sections 3, 5, 7).

use adam_memory::{MemoryId, MemoryKind, MemoryRecord, Provenance, RelationKind};
use rusqlite::params;

use crate::error::TreeError;
use crate::model::{
    summary_level, token_estimate, AdoptionReport, Embedder, LeafClaim, LeafOutcome, LeafWrite,
    NodeState, TreeNode,
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
    /// The write is claim-first. A record's id is minted by
    /// `MemoryRecord::new`, before anything is stored, so the write
    /// first records a claim on the pair in `tree_claims`, carrying
    /// the leaf's real thread key. The claim table's primary key is
    /// the single atomic gate: only the claim's owner goes on to
    /// store a record, so a completed write stores exactly one
    /// record per pair. The sidecar row is inserted last, once the
    /// record exists to satisfy its foreign key, and takes its
    /// placement from the claim.
    ///
    /// The order this replaces (record first, sidecar second, no
    /// claim) had a window between the two writes: an adoption
    /// sweep inside it adopted the orphan record into the fallback
    /// thread bucket, and the channel retry then stored a second
    /// record for the pair. With the claim carrying the placement,
    /// the sweep instead completes the interrupted write in its
    /// real thread (see [`TreeStore::adopt_orphans`] step 0 and
    /// [`TreeStore::land_claim`]). The claim cannot live in
    /// `tree_nodes` itself: that table's foreign key to
    /// `memories(id)` is enforced, so a row there cannot precede
    /// its record.
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
        // Fast path: the pair already has its leaf. A claim left on
        // the pair is residue of the completed write, which
        // adoption retires; nothing to do here.
        if let Some(existing) = self.node_by_external(&input.channel, &input.external_id)? {
            return Ok(LeafOutcome {
                memory_id: existing.memory_id,
                duplicate: true,
            });
        }
        // A claim without a leaf is an earlier attempt that stopped
        // mid-write. If its record landed, land its node from the
        // claim and report the duplicate; if not, the claim is
        // stale and this retry clears it and writes fresh.
        if let Some(claim) = self.claim_by_external(&input.channel, &input.external_id)? {
            if let Some((node, _)) = self.land_claim(&claim)? {
                return Ok(LeafOutcome {
                    memory_id: node.memory_id,
                    duplicate: true,
                });
            }
            self.delete_claim_owned(&claim)?;
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
        let claim = LeafClaim {
            memory_id: record.id,
            channel: input.channel.clone(),
            external_id: input.external_id.clone(),
            thread_key: input.thread_key.clone(),
            contact_key: input.contact_key.clone(),
            pinned: input.pinned,
        };
        match self.insert_claim(&claim) {
            Ok(()) => {}
            Err(err) => {
                // Lost the claim race against a concurrent writer
                // of the same pair: their leaf is canonical and
                // ours was never stored, so this path leaves no
                // orphan behind. If their record already landed,
                // land their node from their claim and point at it.
                // If it has not landed, success would point at a
                // record that may never exist (their writer can
                // still fail or die), so the lost-race error stands
                // and the caller retries; their claim settles the
                // pair on that retry.
                if let Some(other) = self.claim_by_external(&input.channel, &input.external_id)? {
                    if let Some((node, _)) = self.land_claim(&other)? {
                        return Ok(LeafOutcome {
                            memory_id: node.memory_id,
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
            let _ = self.delete_claim_owned(&claim);
            return Err(err.into());
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
            state: NodeState::Open,
        };
        match self.insert_node(&node) {
            Ok(()) => {}
            Err(err) => {
                // The pair's node exists under another id: a retry
                // settled our claim as stale and completed first.
                // Their leaf is canonical; ours stays an unadopted
                // orphan, the race-loser rule adoption already
                // applies. Any other failure leaves the claim in
                // place, and adoption lands the node from it.
                if let Some(existing) = self.node_by_external(&input.channel, &input.external_id)? {
                    return Ok(LeafOutcome {
                        memory_id: existing.memory_id,
                        duplicate: true,
                    });
                }
                return Err(err);
            }
        }
        let _ = self.delete_claim_owned(&claim);
        Ok(LeafOutcome {
            memory_id: record.id,
            duplicate: false,
        })
    }

    /// Complete an interrupted leaf write from its claim: when the
    /// claim's record exists, ensure the leaf's sidecar row is
    /// landed with the claim's placement and retire the claim.
    /// Returns the pair's node and whether this call inserted it,
    /// or `None` when the claim's record never landed (a write
    /// that crashed before storing, or one still in flight).
    fn land_claim(&self, claim: &LeafClaim) -> Result<Option<(TreeNode, bool)>, TreeError> {
        let Some(record) = self.store.get(claim.memory_id)? else {
            return Ok(None);
        };
        if let Some(node) = self.node(claim.memory_id)? {
            self.delete_claim_owned(claim)?;
            return Ok(Some((node, false)));
        }
        if let Some(existing) = self.node_by_external(&claim.channel, &claim.external_id)? {
            // The pair was completed by another leaf; this claim's
            // record is a race-loser and the claim is residue.
            self.delete_claim_owned(claim)?;
            return Ok(Some((existing, false)));
        }
        let node = TreeNode {
            memory_id: claim.memory_id,
            level: 0,
            thread_key: claim.thread_key.clone(),
            contact_key: claim.contact_key.clone(),
            channel: Some(claim.channel.clone()),
            external_id: Some(claim.external_id.clone()),
            span_start: record.created_at,
            span_end: record.created_at,
            covered_count: 0,
            token_estimate: token_estimate(&record.content),
            pinned: claim.pinned,
            state: NodeState::Open,
        };
        if let Err(err) = self.insert_node(&node) {
            // A concurrent writer or sweep landed the pair between
            // the checks above and this insert (the same race
            // adoption step 1 already absorbs). Report the landed
            // node instead of failing the caller; only a failure
            // with no landed node is a real error.
            let landed = match self.node(claim.memory_id)? {
                Some(n) => Some(n),
                None => self.node_by_external(&claim.channel, &claim.external_id)?,
            };
            return match landed {
                Some(existing) => {
                    self.delete_claim_owned(claim)?;
                    Ok(Some((existing, false)))
                }
                None => Err(err),
            };
        }
        self.delete_claim_owned(claim)?;
        Ok(Some((node, true)))
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
    /// 0. Leaf claims left by interrupted `write_leaf` calls are
    ///    resolved: a claim whose record landed gets its sidecar row
    ///    landed from the claim (in the leaf's real thread) or is
    ///    retired if the row already landed. A claim with no record
    ///    is left alone: it belongs to a live writer or to a write
    ///    that crashed before storing, and the pair's next
    ///    `write_leaf` settles it.
    /// 1. Episodic records with a `channel:` origin and no sidecar
    ///    row are adopted as orphan leaves (thread key falls back to
    ///    the date-bucketed slug from spec section 2). Records that
    ///    came through `write_leaf` were already landed from their
    ///    claims in step 0; the fallback here serves records
    ///    written outside `write_leaf` (the escape hatch, pre-claim
    ///    data).
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

        // 0. Resolve leaf claims.
        for claim in self.all_claims()? {
            match self.land_claim(&claim)? {
                Some((_, inserted)) => {
                    report.claims_resolved += 1;
                    if inserted {
                        report.leaves_adopted += 1;
                    }
                }
                None => {
                    // No record landed for this claim; leave it for
                    // the pair's next write_leaf to settle.
                }
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

    /// A leaf record and its claim, built exactly as `write_leaf`
    /// builds them, so a test can stop the write at a chosen step
    /// and drive the remaining steps by hand. That is what makes
    /// the crash interleavings deterministic: no threads, no
    /// sleeps, no timing luck.
    fn record_and_claim(external_id: &str, at: DateTime<Utc>) -> (MemoryRecord, LeafClaim) {
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
        let claim = LeafClaim {
            memory_id: record.id,
            channel: "email".to_string(),
            external_id: external_id.to_string(),
            thread_key: THREAD.to_string(),
            contact_key: CONTACT.to_string(),
            pinned: false,
        };
        (record, claim)
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

    /// Crash window 1: claim recorded, record never stored.
    /// Adoption must not invent a fallback-bucket leaf out of the
    /// claim, and the retry must settle the stale claim and land
    /// the leaf, once, in its real thread.
    #[test]
    fn crash_after_claim_retry_writes_fresh_in_real_thread() {
        let (store, path) = open("claim-crash");
        let at = Utc::now() - Duration::hours(1);
        let (record, claim) = record_and_claim("ext-claim", at);
        store.insert_claim(&claim).unwrap();

        let adopted = store.adopt_orphans().unwrap();
        assert_eq!(adopted.leaves_adopted, 0);
        assert_eq!(adopted.claims_resolved, 0);
        assert!(store
            .claim_by_external("email", "ext-claim")
            .unwrap()
            .is_some());

        let outcome = store
            .write_leaf(&HashEmbedder, &leaf_write("ext-claim", &record.content, at))
            .unwrap();
        assert!(!outcome.duplicate);
        assert_eq!(records_for(&store, "ext-claim"), 1);
        let node = store.node(outcome.memory_id).unwrap().unwrap();
        assert_eq!(node.thread_key, THREAD);
        assert_eq!(node.state, NodeState::Open);
        assert!(store
            .claim_by_external("email", "ext-claim")
            .unwrap()
            .is_none());
        store.assert_partition().unwrap();
        cleanup(&path);
    }

    /// Crash window 2: claim recorded and record stored, sidecar
    /// row never inserted. Adoption lands the leaf from the claim,
    /// in its real thread (never the fallback bucket), and the
    /// retry is a duplicate of the same single record.
    #[test]
    fn crash_after_store_adoption_lands_leaf_in_real_thread() {
        let (store, path) = open("store-crash");
        let at = Utc::now() - Duration::hours(2);
        let (record, claim) = record_and_claim("ext-stored", at);
        store.insert_claim(&claim).unwrap();
        store.store.store(&record).unwrap();

        let adopted = store.adopt_orphans().unwrap();
        assert_eq!(adopted.claims_resolved, 1);
        assert_eq!(adopted.leaves_adopted, 1);
        let node = store.node(record.id).unwrap().unwrap();
        assert_eq!(node.thread_key, THREAD);
        assert_eq!(node.state, NodeState::Open);
        assert!(store
            .claim_by_external("email", "ext-stored")
            .unwrap()
            .is_none());

        let outcome = store
            .write_leaf(
                &HashEmbedder,
                &leaf_write("ext-stored", &record.content, at),
            )
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
        let (record, claim) = record_and_claim("ext-window", at);

        // Writer A: claim and record in, sidecar row not yet.
        a.insert_claim(&claim).unwrap();
        a.store.store(&record).unwrap();

        // The sweep runs inside the window.
        let adopted = b.adopt_orphans().unwrap();
        assert_eq!(adopted.leaves_adopted, 1);
        assert_eq!(adopted.claims_resolved, 1);

        // The channel retry of the same pair.
        let outcome = b
            .write_leaf(
                &HashEmbedder,
                &leaf_write("ext-window", &record.content, at),
            )
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

    /// The claim table's primary key is the gate: a second
    /// claimant with a different record id cannot take a claimed
    /// pair over.
    #[test]
    fn second_claim_for_same_pair_is_rejected_by_the_index() {
        let path = temp_db("gate");
        let a = TreeStore::open(path.to_str().unwrap()).unwrap();
        let b = TreeStore::open(path.to_str().unwrap()).unwrap();
        let at = Utc::now();
        let (_, claim_a) = record_and_claim("ext-gate", at);
        let (_, claim_b) = record_and_claim("ext-gate", at);
        a.insert_claim(&claim_a).unwrap();
        assert!(b.insert_claim(&claim_b).is_err());
        let owner = b.claim_by_external("email", "ext-gate").unwrap().unwrap();
        assert_eq!(owner.memory_id, claim_a.memory_id);
        cleanup(&path);
    }

    /// A claim left behind by a write that landed its sidecar row
    /// but crashed before retiring the claim is cleaned up by
    /// adoption, without touching the leaf.
    #[test]
    fn landed_write_claim_residue_is_retired_by_adoption() {
        let (store, path) = open("residue");
        let at = Utc::now() - Duration::hours(1);
        let outcome = store
            .write_leaf(
                &HashEmbedder,
                &leaf_write("ext-residue", "content for ext-residue", at),
            )
            .unwrap();
        assert!(!outcome.duplicate);
        // Simulate the crash residue: the claim reappears.
        store
            .insert_claim(&LeafClaim {
                memory_id: outcome.memory_id,
                channel: "email".to_string(),
                external_id: "ext-residue".to_string(),
                thread_key: THREAD.to_string(),
                contact_key: CONTACT.to_string(),
                pinned: false,
            })
            .unwrap();

        let adopted = store.adopt_orphans().unwrap();
        assert_eq!(adopted.claims_resolved, 1);
        assert_eq!(adopted.leaves_adopted, 0);
        assert!(store
            .claim_by_external("email", "ext-residue")
            .unwrap()
            .is_none());
        assert_eq!(records_for(&store, "ext-residue"), 1);
        store.assert_partition().unwrap();
        cleanup(&path);
    }

    /// Each of the four key fields is required on its own: any one
    /// of them empty (the others set) rejects the write, and a
    /// rejected write stores nothing.
    #[test]
    fn write_leaf_rejects_each_empty_key_field() {
        let (store, path) = open("validate");
        let base = LeafWrite::new("email", CONTACT, THREAD, "ext-valid", "hello");
        for field in ["channel", "contact", "thread", "external"] {
            let mut input = base.clone();
            match field {
                "channel" => input.channel = String::new(),
                "contact" => input.contact_key = String::new(),
                "thread" => input.thread_key = String::new(),
                _ => input.external_id = String::new(),
            }
            let err = store.write_leaf(&HashEmbedder, &input).unwrap_err();
            assert!(matches!(err, TreeError::InvalidInput(_)), "{field}: {err}");
        }
        assert_eq!(store.store.all().unwrap().len(), 0);
        cleanup(&path);
    }

    /// Step 1 of adoption takes Episodic records only: an orphan
    /// episodic record is adopted into the fallback bucket, and a
    /// Semantic record that merely wears a `channel:` origin is
    /// not a leaf and stays out of the tree.
    #[test]
    fn adoption_takes_episodic_orphans_only() {
        let (store, path) = open("kinds");
        let at = Utc::now() - Duration::days(30);
        let mut episodic = MemoryRecord::new(
            MemoryKind::Episodic,
            "orphaned exchange",
            vec![0.2, 0.8],
            1.0,
            Provenance {
                origin: format!("channel:email:{CONTACT}"),
                evidence: vec!["ext-orphan".to_string()],
            },
            0.0,
        );
        episodic.created_at = at;
        episodic.last_accessed_at = at;
        store.store.store(&episodic).unwrap();
        let semantic = MemoryRecord::new(
            MemoryKind::Semantic,
            "not a leaf",
            vec![0.2, 0.8],
            1.0,
            Provenance {
                origin: format!("channel:email:{CONTACT}"),
                evidence: vec!["ext-not-leaf".to_string()],
            },
            0.0,
        );
        store.store.store(&semantic).unwrap();

        let adopted = store.adopt_orphans().unwrap();
        assert_eq!(adopted.leaves_adopted, 1);
        let node = store.node(episodic.id).unwrap().unwrap();
        assert_eq!(
            node.thread_key,
            format!("{CONTACT}/{}", at.format("%Y-%m")),
            "orphans land in the date-bucketed fallback thread"
        );
        assert!(store.node(semantic.id).unwrap().is_none());
        cleanup(&path);
    }

    /// Coverage repair must add the `Generalizes` edge when the
    /// child is related to the summary by some OTHER relation: a
    /// `Supports` edge to the same summary is not coverage.
    #[test]
    fn repair_coverage_adds_generalizes_beside_other_relations() {
        let (store, path) = open("repair");
        let at = Utc::now() - Duration::hours(1);
        let outcome = store
            .write_leaf(
                &HashEmbedder,
                &leaf_write("ext-repair", "repairable leaf", at),
            )
            .unwrap();
        let summary = MemoryRecord::new(
            MemoryKind::Semantic,
            "repair summary",
            vec![0.3, 0.7],
            0.9,
            Provenance {
                origin: "summary:tree:L1".to_string(),
                evidence: vec![outcome.memory_id.to_string()],
            },
            0.0,
        );
        store.store.store(&summary).unwrap();
        store
            .store
            .relate(outcome.memory_id, summary.id, RelationKind::Supports)
            .unwrap();

        let adopted = store.adopt_orphans().unwrap();
        assert_eq!(adopted.children_flipped, 1);
        let outgoing = store.store.related_outgoing(outcome.memory_id).unwrap();
        assert!(
            outgoing
                .iter()
                .any(|(rec, kind)| rec.id == summary.id && *kind == RelationKind::Generalizes),
            "the Generalizes edge must be added beside the Supports edge"
        );
        assert_eq!(
            store.node(outcome.memory_id).unwrap().unwrap().state,
            NodeState::Covered
        );
        cleanup(&path);
    }
}
