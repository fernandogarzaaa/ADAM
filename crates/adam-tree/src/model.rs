//! Public data types and backend traits for the tree layer.

use adam_memory::{MemoryId, MemoryKind};
use chrono::{DateTime, Utc};

use crate::error::TreeError;

/// The one token measure used by the whole layer: `ceil(chars / 4)`.
/// Trigger math and budget math share it, so the two can never
/// disagree about units (spec section 2).
pub fn token_estimate(content: &str) -> usize {
    content.chars().count().div_ceil(4)
}

/// Produces embeddings for leaf and summary content. ADAM treats
/// embeddings as opaque vectors; the production implementation wraps
/// adam-organism's MiniLM embedder, tests use deterministic fakes.
pub trait Embedder {
    /// Embed one text into a vector.
    fn embed(&self, text: &str) -> Result<Vec<f32>, TreeError>;
}

/// Produces abstractive summary text under the compression contract
/// in spec section 5. The crate is model-agnostic: the orchestrator
/// supplies the LLM-backed implementation, and a pass rejects any
/// result over its target or containing credential-shaped content.
pub trait Summarizer {
    /// Summarize one pass's children. `request.attempt` is 1 for the
    /// first try and 2 for the retry after an oversize result, with
    /// `request.retry_note` explaining the rejection.
    fn summarize(&self, request: &SummaryRequest) -> Result<String, TreeError>;
}

/// One child being summarized, in span order.
#[derive(Debug, Clone)]
pub struct SummaryChild {
    /// The child's ADAM record id.
    pub memory_id: MemoryId,
    /// The child's verbatim content.
    pub content: String,
    /// The channel the child happened on; leaves only.
    pub channel: Option<String>,
    /// Start of the child's covered span.
    pub span_start: DateTime<Utc>,
    /// End of the child's covered span.
    pub span_end: DateTime<Utc>,
    /// The child's token estimate.
    pub token_estimate: usize,
}

/// Everything a [`Summarizer`] needs for one pass.
#[derive(Debug, Clone)]
pub struct SummaryRequest {
    /// Level being created: 1 (thread chunk), 2 (thread), 3 (contact).
    pub level: u32,
    /// The contact the summarized nodes belong to.
    pub contact_key: String,
    /// The thread for L1/L2 passes; `None` for L3 contact passes.
    pub thread_key: Option<String>,
    /// Start of the covered span (earliest child span start).
    pub span_start: DateTime<Utc>,
    /// End of the covered span (latest child span end).
    pub span_end: DateTime<Utc>,
    /// The children to summarize, in span order.
    pub children: Vec<SummaryChild>,
    /// Hard target: the returned text's [`token_estimate`] must not
    /// exceed this. Derived from the covered children's token sum by
    /// the level's target ratio.
    pub target_max_tokens: usize,
    /// 1 for the first attempt, 2 for the retry after an oversize
    /// result.
    pub attempt: u32,
    /// Set on the retry: why the previous attempt was rejected.
    pub retry_note: Option<String>,
}

/// Input for [`crate::TreeStore::write_leaf`]. One leaf is one
/// exchange (an inbound message plus Cookie's reply) or one
/// standalone inbound unit (an unanswered message is an open thread
/// and stays its own leaf). The worker resolves `contact_key` and
/// `thread_key` before calling; the tree never guesses identity
/// merges after the fact.
#[derive(Debug, Clone)]
pub struct LeafWrite {
    /// Channel the exchange happened on: email, sms, whatsapp,
    /// messenger, imessage, ...
    pub channel: String,
    /// Inan's stable identifier for the person, across channels.
    pub contact_key: String,
    /// `<contact_key>/<thread_slug>`; a logical conversation, not a
    /// channel artifact, so one thread can span channels.
    pub thread_key: String,
    /// The channel's message id for this unit; the idempotency key.
    pub external_id: String,
    /// Verbatim content, with speaker labels (`contact:` / `cookie:`)
    /// inside the text. Never edited after write.
    pub content: String,
    /// 1.0 for directly observed conversation; lower only for
    /// transcribed or relayed content.
    pub confidence: f32,
    /// Pinned leaves are never summarized and never demoted.
    pub pinned: bool,
    /// Overrides the leaf's creation time (backfills, tests);
    /// `None` means now.
    pub at: Option<DateTime<Utc>>,
}

impl LeafWrite {
    /// A leaf write with the defaults: confidence 1.0, unpinned,
    /// created now.
    pub fn new(
        channel: impl Into<String>,
        contact_key: impl Into<String>,
        thread_key: impl Into<String>,
        external_id: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            channel: channel.into(),
            contact_key: contact_key.into(),
            thread_key: thread_key.into(),
            external_id: external_id.into(),
            content: content.into(),
            confidence: 1.0,
            pinned: false,
            at: None,
        }
    }
}

/// Result of a leaf write: the leaf's id, and whether the write was
/// a retry of an already-stored `(channel, external_id)` pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeafOutcome {
    /// The leaf's ADAM record id.
    pub memory_id: MemoryId,
    /// True when the pair was already stored and nothing was written.
    pub duplicate: bool,
}

/// Coverage state of a tree node (the sidecar `state` column).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeState {
    /// Not yet covered by a next-level summary; a current top node.
    Open,
    /// Covered by exactly one next-level summary.
    Covered,
}

impl NodeState {
    /// The stored string form.
    pub fn as_str(&self) -> &'static str {
        match self {
            NodeState::Open => "open",
            NodeState::Covered => "covered",
        }
    }

    /// Parse the stored string form.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "open" => Some(NodeState::Open),
            "covered" => Some(NodeState::Covered),
            _ => None,
        }
    }
}

/// One row of the `tree_nodes` sidecar table: everything about a
/// conversational record that ADAM itself cannot express (spec
/// section 2).
#[derive(Debug, Clone, PartialEq)]
pub struct TreeNode {
    /// The ADAM record this row describes.
    pub memory_id: MemoryId,
    /// 0 = leaf, 1..=3 = summary levels.
    pub level: u32,
    /// The logical conversation this node belongs to.
    pub thread_key: String,
    /// The person this node belongs to, across channels.
    pub contact_key: String,
    /// Leaves only: where the leaf actually happened. `None` on
    /// summaries.
    pub channel: Option<String>,
    /// Leaves only: the channel message id (the idempotency key).
    pub external_id: Option<String>,
    /// Start of the covered span (a leaf's own creation time).
    pub span_start: DateTime<Utc>,
    /// End of the covered span.
    pub span_end: DateTime<Utc>,
    /// Direct children covered by this node (summaries only).
    pub covered_count: usize,
    /// `ceil(chars / 4)` of the record's content.
    pub token_estimate: usize,
    /// Never summarize, never demote.
    pub pinned: bool,
    /// Coverage state.
    pub state: NodeState,
}

/// Request for [`crate::TreeStore::read_context`] (spec section 7).
#[derive(Debug, Clone)]
pub struct ContextRequest {
    /// The contact whose context to read.
    pub contact_key: String,
    /// The active thread; `None` reads at contact scope.
    pub thread_key: Option<String>,
    /// The inbound message text, embedded for the recall band.
    pub query: Option<String>,
    /// Total token budget across the three bands; 0 means the
    /// configured default.
    pub token_budget: usize,
}

impl ContextRequest {
    /// A context request for one active thread, with default budget
    /// and no recall query.
    pub fn for_thread(contact_key: impl Into<String>, thread_key: impl Into<String>) -> Self {
        Self {
            contact_key: contact_key.into(),
            thread_key: Some(thread_key.into()),
            query: None,
            token_budget: 0,
        }
    }
}

/// One record in a read-path response (spec section 4 payload).
#[derive(Debug, Clone)]
pub struct ContextRecord {
    /// The ADAM record id.
    pub memory_id: MemoryId,
    /// Tree level, or `None` for ADAM records outside the tree
    /// (possible in the recall band).
    pub level: Option<u32>,
    /// The ADAM record kind.
    pub kind: MemoryKind,
    /// The verbatim record content. Never truncated mid-record.
    pub content: String,
    /// The record's provenance origin.
    pub origin: String,
    /// Start of the record's span.
    pub span_start: DateTime<Utc>,
    /// End of the record's span.
    pub span_end: DateTime<Utc>,
    /// The record's token estimate.
    pub token_estimate: usize,
    /// Direct children, for summaries that can be drilled into.
    pub children: usize,
    /// Similarity score; set for recall and search results.
    pub score: Option<f32>,
}

/// The three-band result of `read_context` (spec section 4): recent
/// leaves verbatim, then the summary chain, then relevant recall.
#[derive(Debug, Clone)]
pub struct ContextBundle {
    /// Band 1: newest leaves of the active thread, verbatim,
    /// newest first. At most 45% of the budget.
    pub recent_leaves: Vec<ContextRecord>,
    /// Band 2: the summary chain, thread scope before contact scope.
    /// At most 40% of the budget.
    pub summaries: Vec<ContextRecord>,
    /// Band 3: similarity recall over the whole store, excluding
    /// anything already returned. At most the remaining ~15%.
    pub recall: Vec<ContextRecord>,
    /// Sum of the returned records' token estimates.
    pub tokens_used: usize,
    /// True when the single-oversize-summary exception fired: one
    /// summary larger than its whole band was returned alone and the
    /// total may exceed the budget (spec section 4).
    pub overflow: bool,
}

/// Result of [`crate::TreeStore::drill_down`]: a summary's direct
/// children in span order.
#[derive(Debug, Clone)]
pub struct DrillResult {
    /// The children that fit the budget, whole records only.
    pub children: Vec<ContextRecord>,
    /// Sum of the returned children's token estimates.
    pub tokens_used: usize,
}

/// One hit from [`crate::TreeStore::search_tree`].
#[derive(Debug, Clone)]
pub struct SearchHit {
    /// The matched record, with its level and keys from the sidecar.
    pub record: ContextRecord,
    /// Cosine similarity to the query.
    pub score: f32,
}

/// Scope for [`crate::TreeStore::summarize_now`] (spec section 7).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SummarizeScope {
    /// One thread: its L1 pass, with the L2 cascade by due rules.
    Thread(String),
    /// One contact: its L3 pass over open thread summaries.
    Contact(String),
    /// Everything due, tree-wide.
    AllDue,
}

/// Outcome of a summarization run (spec section 7 response shape).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SummaryReport {
    /// Passes that produced a summary.
    pub passes_run: usize,
    /// Ids of the summaries created, in creation order.
    pub summaries_created: Vec<MemoryId>,
    /// Children flipped to covered across all passes.
    pub nodes_covered: usize,
    /// Why nothing (or not everything) ran: `"lease held"`,
    /// `"nothing due"`, an oversize rejection, ...
    pub skipped_reason: Option<String>,
}

/// What an adoption sweep repaired (spec section 3): partial writes
/// left behind by crashed writers, healed instead of corrupting the
/// tree.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AdoptionReport {
    /// Orphan leaves given a sidecar row.
    pub leaves_adopted: usize,
    /// Orphan summaries whose sidecar row was rebuilt from evidence.
    pub summaries_adopted: usize,
    /// Children whose coverage (edge plus covered state) was repaired.
    pub children_flipped: usize,
}

/// A held summarizer lease (spec section 6). Only the orchestrator
/// holds it; channel workers never summarize.
#[derive(Debug, Clone)]
pub struct Lease {
    /// The holder's identity string.
    pub holder: String,
    /// When the lease lapses without renewal.
    pub expires_at: DateTime<Utc>,
}

/// Parse the level out of a summary record's origin
/// (`summary:tree:L<level>`).
pub(crate) fn summary_level(origin: &str) -> Option<u32> {
    origin.strip_prefix("summary:tree:L")?.parse().ok()
}
