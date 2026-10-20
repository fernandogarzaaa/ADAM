//! The read path: three-band context, tree search, and drill-down
//! (spec sections 4 and 7).

use std::collections::{HashMap, HashSet};

use adam_memory::{MemoryError, MemoryId, MemoryRecord, RelationKind};
use chrono::{DateTime, Utc};

use crate::error::TreeError;
use crate::model::{
    token_estimate, ContextBundle, ContextRecord, ContextRequest, DrillResult, Embedder, NodeState,
    SearchHit, TreeNode,
};
use crate::store::TreeStore;

impl TreeStore {
    /// The only read a channel worker needs to answer a message
    /// (spec `read_context`). Three bands, filled in order, truncated
    /// only at band boundaries, never mid-record:
    ///
    /// 1. Recent leaves of the active thread, verbatim, newest
    ///    first — at most 45% of the budget.
    /// 2. The summary chain: the active thread's summaries (top
    ///    first), then the contact's L3, then sibling-thread L2
    ///    nodes — at most 40%.
    /// 3. Similarity recall over the whole store for the query,
    ///    excluding anything already returned — the remaining ~15%.
    ///    Pinned leaves of the active contact that match at all are
    ///    surfaced first (spec section 5).
    ///
    /// The one exception to the budget: a single summary larger than
    /// its entire band is returned alone, with `overflow` set.
    pub fn read_context(
        &self,
        embedder: &dyn Embedder,
        request: &ContextRequest,
    ) -> Result<ContextBundle, TreeError> {
        let budget = if request.token_budget == 0 {
            self.config.default_token_budget
        } else {
            request.token_budget
        };
        let band1_cap = budget * 45 / 100;
        let band2_cap = budget * 40 / 100;
        let band3_cap = budget - band1_cap - band2_cap;

        let mut included: HashSet<MemoryId> = HashSet::new();
        let mut tokens_used = 0usize;
        let mut overflow = false;

        // Band 1: recent leaves, verbatim, newest first.
        let mut leaves = match &request.thread_key {
            Some(thread) => self.query_nodes(Some(thread.as_str()), None, Some(0), None)?,
            None => self.query_nodes(None, Some(&request.contact_key), Some(0), None)?,
        };
        leaves.sort_by(|a, b| {
            b.span_start
                .cmp(&a.span_start)
                .then_with(|| b.span_end.cmp(&a.span_end))
                .then_with(|| b.memory_id.to_string().cmp(&a.memory_id.to_string()))
        });
        let mut recent_leaves = Vec::new();
        let mut band_used = 0usize;
        for leaf in &leaves {
            if band_used + leaf.token_estimate > band1_cap {
                continue;
            }
            let Some(record) = self.touch(leaf.memory_id)? else {
                continue;
            };
            band_used += leaf.token_estimate;
            included.insert(leaf.memory_id);
            recent_leaves.push(self.context_record(&record, Some(leaf)));
        }
        tokens_used += band_used;

        // Band 2: the summary chain.
        let chain = self.summary_chain(request)?;
        let mut summaries = Vec::new();
        let mut band_used = 0usize;
        for node in &chain {
            if included.contains(&node.memory_id) {
                continue;
            }
            if band_used + node.token_estimate > band2_cap {
                if summaries.is_empty() && !overflow {
                    // Single-oversize exception (spec section 4):
                    // one summary larger than its whole band is
                    // returned alone and the band overflows, flagged,
                    // so the caller sees the true cost.
                    let Some(record) = self.touch(node.memory_id)? else {
                        continue;
                    };
                    band_used += node.token_estimate;
                    included.insert(node.memory_id);
                    summaries.push(self.context_record(&record, Some(node)));
                    overflow = true;
                }
                continue;
            }
            let Some(record) = self.touch(node.memory_id)? else {
                continue;
            };
            band_used += node.token_estimate;
            included.insert(node.memory_id);
            summaries.push(self.context_record(&record, Some(node)));
        }
        tokens_used += band_used;

        // Band 3: relevant recall over the whole store.
        let mut recall = Vec::new();
        if let Some(query) = &request.query {
            if !query.is_empty() {
                let query_embedding = embedder.embed(query)?;
                let scored = self
                    .store
                    .query_similar(&query_embedding, None, usize::MAX)?;
                let mut band_used = 0usize;
                // Two passes over the same ranking: pinned leaves of
                // the active contact first, then everything else.
                for pinned_pass in [true, false] {
                    for (record, score) in &scored {
                        if included.contains(&record.id) {
                            continue;
                        }
                        let node = self.node(record.id)?;
                        let pinned_contact_leaf = node.as_ref().is_some_and(|n| {
                            n.pinned && n.level == 0 && n.contact_key == request.contact_key
                        });
                        if pinned_pass {
                            if !(pinned_contact_leaf && *score > 0.0) {
                                continue;
                            }
                        } else if pinned_contact_leaf && *score > 0.0 {
                            continue; // handled in the first pass
                        }
                        let cost = node
                            .as_ref()
                            .map_or_else(|| token_estimate(&record.content), |n| n.token_estimate);
                        if band_used + cost > band3_cap {
                            continue;
                        }
                        let Some(touched) = self.touch(record.id)? else {
                            continue;
                        };
                        band_used += cost;
                        included.insert(record.id);
                        let mut payload = self.context_record(&touched, node.as_ref());
                        payload.score = Some(*score);
                        recall.push(payload);
                    }
                }
                tokens_used += band_used;
            }
        }

        Ok(ContextBundle {
            recent_leaves,
            summaries,
            recall,
            tokens_used,
            overflow,
        })
    }

    /// Band 2's candidate order (spec section 4): the active
    /// thread's open summaries, highest level first; then the
    /// contact's open L3; then open L2 nodes of the contact's other
    /// threads. At contact scope (no thread), the contact's open
    /// summaries, highest level first.
    fn summary_chain(&self, request: &ContextRequest) -> Result<Vec<TreeNode>, TreeError> {
        let mut chain: Vec<TreeNode> = Vec::new();
        let mut seen: HashSet<MemoryId> = HashSet::new();
        let mut push_all = |nodes: Vec<TreeNode>, chain: &mut Vec<TreeNode>| {
            for node in nodes {
                if seen.insert(node.memory_id) {
                    chain.push(node);
                }
            }
        };
        match &request.thread_key {
            Some(thread) => {
                let mut thread_summaries = self
                    .query_nodes(Some(thread.as_str()), None, None, Some(NodeState::Open))?
                    .into_iter()
                    .filter(|n| n.level >= 1)
                    .collect::<Vec<_>>();
                thread_summaries.sort_by(|a, b| {
                    b.level
                        .cmp(&a.level)
                        .then_with(|| b.span_start.cmp(&a.span_start))
                });
                push_all(thread_summaries, &mut chain);
                let l3 = self.query_nodes(
                    None,
                    Some(&request.contact_key),
                    Some(3),
                    Some(NodeState::Open),
                )?;
                push_all(l3, &mut chain);
                let mut siblings = self
                    .query_nodes(
                        None,
                        Some(&request.contact_key),
                        Some(2),
                        Some(NodeState::Open),
                    )?
                    .into_iter()
                    .filter(|n| &n.thread_key != thread)
                    .collect::<Vec<_>>();
                siblings.sort_by_key(|a| std::cmp::Reverse(a.span_start));
                push_all(siblings, &mut chain);
            }
            None => {
                let mut all = self
                    .query_nodes(
                        None,
                        Some(&request.contact_key),
                        None,
                        Some(NodeState::Open),
                    )?
                    .into_iter()
                    .filter(|n| n.level >= 1)
                    .collect::<Vec<_>>();
                all.sort_by(|a, b| {
                    b.level
                        .cmp(&a.level)
                        .then_with(|| b.span_start.cmp(&a.span_start))
                });
                push_all(all, &mut chain);
            }
        }
        Ok(chain)
    }

    /// Similarity search over tree records (spec `search_tree`):
    /// results carry their level and keys from the sidecar, and a
    /// covered leaf ranks below its covering summary at equal
    /// similarity, so search surfaces the compressed view first.
    pub fn search_tree(
        &self,
        embedder: &dyn Embedder,
        query: &str,
        contact_key: Option<&str>,
        top_k: usize,
    ) -> Result<Vec<SearchHit>, TreeError> {
        let query_embedding = embedder.embed(query)?;
        let scored = self
            .store
            .query_similar(&query_embedding, None, usize::MAX)?;
        let mut hits: Vec<SearchHit> = Vec::new();
        // Covered node -> its covering parent, for the demotion rule.
        let mut parents: HashMap<MemoryId, MemoryId> = HashMap::new();
        for (record, score) in &scored {
            let Some(node) = self.node(record.id)? else {
                continue;
            };
            if let Some(contact) = contact_key {
                if node.contact_key != contact {
                    continue;
                }
            }
            if node.state == NodeState::Covered {
                if let Some(parent) = self.tree_parents(record.id)?.into_iter().next() {
                    parents.insert(record.id, parent);
                }
            }
            let mut payload = self.context_record(record, Some(&node));
            payload.score = Some(*score);
            hits.push(SearchHit {
                record: payload,
                score: *score,
            });
        }
        let hit_ids: HashSet<MemoryId> = hits.iter().map(|h| h.record.memory_id).collect();
        // Rank by score bucket (1e-6 grid, so the order is total),
        // then demote covered nodes whose covering parent is also a
        // hit, then by id for determinism.
        hits.sort_by(|a, b| {
            let bucket_a = (a.score * 1_000_000.0).round() as i64;
            let bucket_b = (b.score * 1_000_000.0).round() as i64;
            let demote = |hit: &SearchHit| -> u8 {
                match parents.get(&hit.record.memory_id) {
                    Some(parent) if hit_ids.contains(parent) => 1,
                    _ => 0,
                }
            };
            bucket_b
                .cmp(&bucket_a)
                .then_with(|| demote(a).cmp(&demote(b)))
                .then_with(|| {
                    a.record
                        .memory_id
                        .to_string()
                        .cmp(&b.record.memory_id.to_string())
                })
        });
        hits.truncate(top_k);
        Ok(hits)
    }

    /// Expand a node into its direct children in span order (spec
    /// `drill_down`). Recursing on a child summary walks the tree;
    /// the bottom level returns the verbatim leaves, because ADAM
    /// never deletes what a summary covered. Records that do not fit
    /// the budget whole are skipped, never cut.
    pub fn drill_down(&self, id: MemoryId, token_budget: usize) -> Result<DrillResult, TreeError> {
        if self.store.get(id)?.is_none() {
            return Err(TreeError::Memory(MemoryError::NotFound(id)));
        }
        let incoming = self.store.related_incoming(id)?;
        let mut children: Vec<(MemoryRecord, Option<TreeNode>)> = Vec::new();
        for (record, kind) in incoming {
            if kind != RelationKind::Generalizes {
                continue;
            }
            let node = self.node(record.id)?;
            children.push((record, node));
        }
        children.sort_by(|a, b| {
            let (a_start, a_end) = span_of(a);
            let (b_start, b_end) = span_of(b);
            a_start
                .cmp(&b_start)
                .then_with(|| a_end.cmp(&b_end))
                .then_with(|| a.0.id.to_string().cmp(&b.0.id.to_string()))
        });
        let mut out = Vec::new();
        let mut used = 0usize;
        for (record, node) in &children {
            let cost = node
                .as_ref()
                .map_or_else(|| token_estimate(&record.content), |n| n.token_estimate);
            if used + cost > token_budget {
                continue;
            }
            let Some(touched) = self.touch(record.id)? else {
                continue;
            };
            used += cost;
            out.push(self.context_record(&touched, node.as_ref()));
        }
        Ok(DrillResult {
            children: out,
            tokens_used: used,
        })
    }

    /// Build the read-path payload for one record, joined with its
    /// sidecar row when it has one.
    fn context_record(&self, record: &MemoryRecord, node: Option<&TreeNode>) -> ContextRecord {
        match node {
            Some(node) => ContextRecord {
                memory_id: record.id,
                level: Some(node.level),
                kind: record.kind,
                content: record.content.clone(),
                origin: record.provenance.origin.clone(),
                span_start: node.span_start,
                span_end: node.span_end,
                token_estimate: node.token_estimate,
                children: node.covered_count,
                score: None,
            },
            None => ContextRecord {
                memory_id: record.id,
                level: None,
                kind: record.kind,
                content: record.content.clone(),
                origin: record.provenance.origin.clone(),
                span_start: record.created_at,
                span_end: record.created_at,
                token_estimate: token_estimate(&record.content),
                children: 0,
                score: None,
            },
        }
    }

    /// Best-effort access bookkeeping (spec section 4): every
    /// returned record goes through ADAM's `access`, but a lost race
    /// on that update must not fail the read — the decay clock is
    /// not worth failing a reply over.
    fn touch(&self, id: MemoryId) -> Result<Option<MemoryRecord>, TreeError> {
        match self.store.access(id) {
            Ok(record) => Ok(Some(record)),
            Err(_) => Ok(self.store.get(id)?),
        }
    }
}

/// A child's span: its sidecar span, or its creation instant when it
/// has no sidecar row.
fn span_of(child: &(MemoryRecord, Option<TreeNode>)) -> (DateTime<Utc>, DateTime<Utc>) {
    match &child.1 {
        Some(node) => (node.span_start, node.span_end),
        None => (child.0.created_at, child.0.created_at),
    }
}
