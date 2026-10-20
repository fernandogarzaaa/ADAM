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
    /// The ADAM record write and the sidecar insert are not one
    /// transaction (they travel over two connections); a crash
    /// between them leaves an orphan leaf that the next adoption
    /// sweep heals (see [`TreeStore::adopt_orphans`]).
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
        if let Some(existing) = self.node_by_external(&input.channel, &input.external_id)? {
            return Ok(LeafOutcome {
                memory_id: existing.memory_id,
                duplicate: true,
            });
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
        self.store.store(&record)?;
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
            Ok(()) => Ok(LeafOutcome {
                memory_id: record.id,
                duplicate: false,
            }),
            Err(err) => {
                // Lost a race against a concurrent writer of the same
                // (channel, external_id): the unique index rejected
                // our row, and their leaf is the canonical one. Our
                // just-stored record stays as an unadopted orphan,
                // which the adoption sweep deliberately skips for the
                // same reason.
                if let Some(existing) = self.node_by_external(&input.channel, &input.external_id)? {
                    return Ok(LeafOutcome {
                        memory_id: existing.memory_id,
                        duplicate: true,
                    });
                }
                Err(err)
            }
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
    /// 1. Episodic records with a `channel:` origin and no sidecar
    ///    row are adopted as orphan leaves (thread key falls back to
    ///    the date-bucketed slug from spec section 2).
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
