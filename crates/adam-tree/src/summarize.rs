//! Summarization passes: triggers, the never-summarize filter, the
//! lease-gated pass with crash adoption first, and the partition
//! assertion after (spec sections 3, 5, 6).

use std::collections::HashSet;

use adam_memory::{MemoryId, MemoryKind, MemoryRecord, Provenance, RelationKind};
use chrono::{DateTime, Duration, Utc};

use crate::detect::contains_credential_shaped;
use crate::error::TreeError;
use crate::model::{
    token_estimate, Embedder, NodeState, SummarizeScope, Summarizer, SummaryChild, SummaryReport,
    SummaryRequest, TreeNode,
};
use crate::store::TreeStore;

/// One pass to run: summarize `children` into one node at
/// `target_level`.
struct PassUnit {
    target_level: u32,
    contact_key: String,
    thread_key: Option<String>,
    children: Vec<TreeNode>,
}

impl TreeStore {
    /// Run one summarization scope now (spec `summarize_now`).
    ///
    /// Lease-gated: when another holder's lease is live, nothing runs
    /// and the report carries `skipped_reason: "lease held"`. The
    /// scoped pass is forced (thresholds do not apply) except that a
    /// summary always needs at least two eligible children; the L2
    /// cascade inside a thread scope still follows the due rules.
    pub fn summarize_now(
        &self,
        scope: &SummarizeScope,
        holder: &str,
        summarizer: &dyn Summarizer,
        embedder: &dyn Embedder,
    ) -> Result<SummaryReport, TreeError> {
        self.run_gated(scope, true, holder, summarizer, embedder)
    }

    /// The scheduled entry point: run every due pass in the tree
    /// (thresholds only, nothing forced), lease-gated like
    /// [`TreeStore::summarize_now`].
    pub fn run_due_passes(
        &self,
        holder: &str,
        summarizer: &dyn Summarizer,
        embedder: &dyn Embedder,
    ) -> Result<SummaryReport, TreeError> {
        self.run_gated(&SummarizeScope::AllDue, false, holder, summarizer, embedder)
    }

    fn run_gated(
        &self,
        scope: &SummarizeScope,
        force: bool,
        holder: &str,
        summarizer: &dyn Summarizer,
        embedder: &dyn Embedder,
    ) -> Result<SummaryReport, TreeError> {
        let mut report = SummaryReport::default();
        let Some(lease) = self.try_acquire_lease(holder)? else {
            report.skipped_reason = Some("lease held".to_string());
            return Ok(report);
        };
        let result = self.run_scoped(scope, force, summarizer, embedder, &mut report);
        // Releasing is best-effort: a failed release only means the
        // next pass waits for the TTL.
        let _ = self.release_lease(&lease);
        result?;
        if report.passes_run == 0 && report.skipped_reason.is_none() {
            report.skipped_reason = Some("nothing due".to_string());
        }
        Ok(report)
    }

    fn run_scoped(
        &self,
        scope: &SummarizeScope,
        force: bool,
        summarizer: &dyn Summarizer,
        embedder: &dyn Embedder,
        report: &mut SummaryReport,
    ) -> Result<(), TreeError> {
        // Adoption first: a previous crash must converge before new
        // candidates are read (spec section 3).
        self.adopt_orphans()?;
        match scope {
            SummarizeScope::Thread(thread) => {
                self.thread_passes(thread, force, summarizer, embedder, report)
            }
            SummarizeScope::Contact(contact) => {
                self.contact_pass(contact, force, summarizer, embedder, report)
            }
            SummarizeScope::AllDue => {
                let threads: HashSet<String> = self
                    .query_nodes(None, None, None, None)?
                    .into_iter()
                    .map(|n| n.thread_key)
                    .collect();
                for thread in &threads {
                    self.thread_passes(thread, force, summarizer, embedder, report)?;
                }
                let contacts: HashSet<String> = self
                    .query_nodes(None, None, Some(2), None)?
                    .into_iter()
                    .map(|n| n.contact_key)
                    .collect();
                for contact in &contacts {
                    self.contact_pass(contact, force, summarizer, embedder, report)?;
                }
                Ok(())
            }
        }
    }

    /// L1 over a thread's open leaves, then the L2 cascade over the
    /// thread's open L1 nodes (re-gathered after L1 lands, so fresh
    /// summaries can cascade in the same run).
    fn thread_passes(
        &self,
        thread_key: &str,
        force_l1: bool,
        summarizer: &dyn Summarizer,
        embedder: &dyn Embedder,
        report: &mut SummaryReport,
    ) -> Result<(), TreeError> {
        for (child_level, force) in [(0u32, force_l1), (1u32, false)] {
            let open = self.query_nodes(
                Some(thread_key),
                None,
                Some(child_level),
                Some(NodeState::Open),
            )?;
            let eligible = self.eligible_open(open)?;
            if eligible.is_empty() {
                continue;
            }
            if !self.qualifies(child_level + 1, &eligible, force, Utc::now()) {
                continue;
            }
            let unit = PassUnit {
                target_level: child_level + 1,
                contact_key: eligible[0].contact_key.clone(),
                thread_key: Some(thread_key.to_string()),
                children: eligible,
            };
            self.run_unit_guarded(&unit, summarizer, embedder, report)?;
        }
        Ok(())
    }

    /// L3 over a contact's open L2 nodes, across all of the
    /// contact's threads and channels. This is where the tree
    /// becomes channel-agnostic (spec section 2).
    fn contact_pass(
        &self,
        contact_key: &str,
        force: bool,
        summarizer: &dyn Summarizer,
        embedder: &dyn Embedder,
        report: &mut SummaryReport,
    ) -> Result<(), TreeError> {
        let open = self.query_nodes(None, Some(contact_key), Some(2), Some(NodeState::Open))?;
        let eligible = self.eligible_open(open)?;
        if !self.qualifies(3, &eligible, force, Utc::now()) {
            return Ok(());
        }
        let unit = PassUnit {
            target_level: 3,
            contact_key: contact_key.to_string(),
            thread_key: None,
            children: eligible,
        };
        self.run_unit_guarded(&unit, summarizer, embedder, report)
    }

    /// Run one pass. Content-level rejections (oversize after retry,
    /// credential-shaped summary) are reported through
    /// `report.skipped_reason` with the children left open, per spec
    /// section 5; every other error propagates.
    fn run_unit_guarded(
        &self,
        unit: &PassUnit,
        summarizer: &dyn Summarizer,
        embedder: &dyn Embedder,
        report: &mut SummaryReport,
    ) -> Result<(), TreeError> {
        match self.run_unit(unit, summarizer, embedder, report) {
            Ok(()) => Ok(()),
            Err(TreeError::SummaryTooLarge { .. }) => {
                report.skipped_reason = Some("summary oversize after retry".to_string());
                Ok(())
            }
            Err(TreeError::CredentialContent) => {
                report.skipped_reason =
                    Some("summary rejected: credential-shaped content".to_string());
                Ok(())
            }
            Err(err) => Err(err),
        }
    }

    fn run_unit(
        &self,
        unit: &PassUnit,
        summarizer: &dyn Summarizer,
        embedder: &dyn Embedder,
        report: &mut SummaryReport,
    ) -> Result<(), TreeError> {
        let mut children = unit.children.clone();
        sort_by_span(&mut children);
        let span_start = children
            .iter()
            .map(|n| n.span_start)
            .min()
            .unwrap_or_else(Utc::now);
        let span_end = children
            .iter()
            .map(|n| n.span_end)
            .max()
            .unwrap_or_else(Utc::now);
        let covered_tokens: usize = children.iter().map(|n| n.token_estimate).sum();
        let ratio = if unit.target_level == 1 {
            self.config.l1_target_ratio
        } else {
            self.config.ln_target_ratio
        };
        let target = ((covered_tokens as f32 * ratio).ceil() as usize).max(1);

        let mut payloads: Vec<SummaryChild> = Vec::with_capacity(children.len());
        for child in &children {
            let record = self.store.get(child.memory_id)?.ok_or_else(|| {
                TreeError::Corrupt(format!("tree node {} has no ADAM record", child.memory_id))
            })?;
            payloads.push(SummaryChild {
                memory_id: child.memory_id,
                content: record.content,
                channel: child.channel.clone(),
                span_start: child.span_start,
                span_end: child.span_end,
                token_estimate: child.token_estimate,
            });
        }

        let mut attempt = 1;
        let text = loop {
            let request = SummaryRequest {
                level: unit.target_level,
                contact_key: unit.contact_key.clone(),
                thread_key: unit.thread_key.clone(),
                span_start,
                span_end,
                children: payloads.clone(),
                target_max_tokens: target,
                attempt,
                retry_note: (attempt > 1).then(|| {
                    format!(
                        "the previous attempt exceeded the {target} token target; \
                         compress harder"
                    )
                }),
            };
            let text = summarizer.summarize(&request)?;
            if contains_credential_shaped(&text) {
                return Err(TreeError::CredentialContent);
            }
            let tokens = token_estimate(&text);
            if tokens <= target {
                break text;
            }
            if attempt >= 2 {
                return Err(TreeError::SummaryTooLarge { tokens, target });
            }
            attempt += 1;
        };

        let summary = MemoryRecord::new(
            MemoryKind::Semantic,
            text.clone(),
            embedder.embed(&text)?,
            0.9,
            Provenance {
                origin: format!("summary:tree:L{}", unit.target_level),
                evidence: children.iter().map(|c| c.memory_id.to_string()).collect(),
            },
            0.0,
        );
        let summary_id = summary.id;
        self.store.store(&summary)?;
        let thread_key = match &unit.thread_key {
            Some(thread) => thread.clone(),
            None => {
                if children
                    .iter()
                    .all(|c| c.thread_key == children[0].thread_key)
                {
                    children[0].thread_key.clone()
                } else {
                    unit.contact_key.clone()
                }
            }
        };
        self.insert_node(&TreeNode {
            memory_id: summary_id,
            level: unit.target_level,
            thread_key,
            contact_key: unit.contact_key.clone(),
            channel: None,
            external_id: None,
            span_start,
            span_end,
            covered_count: children.len(),
            token_estimate: token_estimate(&text),
            pinned: false,
            state: NodeState::Open,
        })?;
        for child in &children {
            self.store
                .relate(child.memory_id, summary_id, RelationKind::Generalizes)?;
            self.set_node_state(child.memory_id, NodeState::Covered)?;
        }
        // Post-pass partition assertion for the children just covered
        // (spec section 3): a violation fails the pass loudly rather
        // than being silently repaired.
        for child in &children {
            let parents = self.tree_parents(child.memory_id)?;
            if parents.len() != 1 || parents[0] != summary_id {
                return Err(TreeError::PartitionViolation(
                    child.memory_id,
                    parents.len(),
                ));
            }
        }
        report.passes_run += 1;
        report.summaries_created.push(summary_id);
        report.nodes_covered += children.len();
        Ok(())
    }

    /// The never-summarize filter (spec section 5), enforced in code,
    /// not just in the summarizer's prompt: pinned nodes, `approval:`
    /// records, SelfKnowledge records, and credential-shaped content
    /// are never pass inputs.
    fn is_summarizable(&self, node: &TreeNode, record: &MemoryRecord) -> bool {
        if node.pinned {
            return false;
        }
        if record.kind == MemoryKind::SelfKnowledge {
            return false;
        }
        if record.provenance.origin.starts_with("approval:") {
            return false;
        }
        !contains_credential_shaped(&record.content)
    }

    /// Filter open nodes down to summarization candidates.
    fn eligible_open(&self, nodes: Vec<TreeNode>) -> Result<Vec<TreeNode>, TreeError> {
        let mut out = Vec::new();
        for node in nodes {
            let record = self.store.get(node.memory_id)?.ok_or_else(|| {
                TreeError::Corrupt(format!("tree node {} has no ADAM record", node.memory_id))
            })?;
            if self.is_summarizable(&node, &record) {
                out.push(node);
            }
        }
        Ok(out)
    }

    /// Whether a pass over `children` (into level `target_level`)
    /// should run: forced passes still need two children; otherwise
    /// the section 3 triggers decide.
    fn qualifies(
        &self,
        target_level: u32,
        children: &[TreeNode],
        force: bool,
        now: DateTime<Utc>,
    ) -> bool {
        if children.len() < 2 {
            return false;
        }
        if force {
            return true;
        }
        if target_level == 1 {
            self.leaves_due(children, now)
        } else {
            self.children_due(children)
        }
    }

    fn leaves_due(&self, leaves: &[TreeNode], now: DateTime<Utc>) -> bool {
        let cfg = &self.config;
        if leaves.len() >= cfg.leaf_count_trigger {
            return true;
        }
        if leaves.iter().map(|n| n.token_estimate).sum::<usize>() >= cfg.leaf_token_trigger {
            return true;
        }
        if leaves.len() >= cfg.leaf_age_min_count {
            if let Some(oldest) = leaves.iter().map(|n| n.span_start).min() {
                return now - oldest > Duration::days(cfg.leaf_age_trigger_days);
            }
        }
        false
    }

    fn children_due(&self, nodes: &[TreeNode]) -> bool {
        nodes.len() >= self.config.child_count_trigger
            || nodes.iter().map(|n| n.token_estimate).sum::<usize>()
                >= self.config.child_token_trigger
    }

    /// The tree parents of a node: `Generalizes` targets that are
    /// themselves tree nodes.
    pub(crate) fn tree_parents(&self, id: MemoryId) -> Result<Vec<MemoryId>, TreeError> {
        let outgoing = self.store.related_outgoing(id)?;
        let mut parents = Vec::new();
        for (record, kind) in outgoing {
            if kind == RelationKind::Generalizes && self.node(record.id)?.is_some() {
                parents.push(record.id);
            }
        }
        Ok(parents)
    }

    /// Verify the coverage partition over the whole tree (spec
    /// sections 2 and 3): every covered node has exactly one tree
    /// parent, every open node has none.
    pub fn assert_partition(&self) -> Result<(), TreeError> {
        let nodes = self.query_nodes(None, None, None, None)?;
        for node in &nodes {
            let parents = self.tree_parents(node.memory_id)?;
            let ok = match node.state {
                NodeState::Open => parents.is_empty(),
                NodeState::Covered => parents.len() == 1,
            };
            if !ok {
                return Err(TreeError::PartitionViolation(node.memory_id, parents.len()));
            }
        }
        Ok(())
    }
}

/// Deterministic candidate order: span start, then span end, then id.
fn sort_by_span(nodes: &mut [TreeNode]) {
    nodes.sort_by(|a, b| {
        a.span_start
            .cmp(&b.span_start)
            .then_with(|| a.span_end.cmp(&b.span_end))
            .then_with(|| a.memory_id.to_string().cmp(&b.memory_id.to_string()))
    });
}
