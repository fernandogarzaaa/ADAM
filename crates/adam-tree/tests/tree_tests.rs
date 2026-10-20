//! Integration tests for adam-tree, mapped to the spec's test plan
//! (memory-layer-SPEC.md section 8). File-backed databases are used
//! throughout so the locking and WAL behavior under test is the
//! deployed one.

use std::cell::RefCell;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use adam_memory::{MemoryKind, MemoryRecord, MemoryStore, Provenance};
use adam_tree::{
    token_estimate, ConstEmbedder, ContextRequest, FakeSummarizer, HashEmbedder, LeafOutcome,
    LeafWrite, MemoryId, NodeState, RelationKind, SummarizeScope, Summarizer, SummaryRequest,
    TreeConfig, TreeError, TreeStore,
};
use chrono::{Duration, Utc};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

fn temp_db(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let path = std::env::temp_dir().join(format!(
        "adam-tree-test-{tag}-{}-{n}.db",
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

/// Content of exactly `tokens` tokens (4 chars each, per the layer's
/// token measure), prefixed by a lowercase marker.
fn sized(marker: &str, tokens: usize) -> String {
    let mut s = marker.to_string();
    while s.chars().count() < tokens * 4 {
        s.push('x');
    }
    s
}

const CONTACT: &str = "alex@example.com";
const THREAD: &str = "alex@example.com/contract-review";

fn leaf(
    store: &TreeStore,
    channel: &str,
    thread: &str,
    external_id: &str,
    content: String,
) -> LeafOutcome {
    store
        .write_leaf(
            &HashEmbedder,
            &LeafWrite::new(channel, CONTACT, thread, external_id, content),
        )
        .unwrap()
}

fn leaf_at(
    store: &TreeStore,
    thread: &str,
    external_id: &str,
    content: String,
    at: chrono::DateTime<Utc>,
) -> LeafOutcome {
    let mut input = LeafWrite::new("email", CONTACT, thread, external_id, content);
    input.at = Some(at);
    store.write_leaf(&HashEmbedder, &input).unwrap()
}

/// Insert a `tree_nodes` row directly, for records planted outside
/// the layer's own write path (approvals, self-knowledge, corrupt
/// summaries).
fn plant_node(
    path: &Path,
    record: &MemoryRecord,
    level: i64,
    covered_count: i64,
    token_override: Option<i64>,
) {
    let conn = rusqlite::Connection::open(path).unwrap();
    conn.execute(
        "INSERT INTO tree_nodes (memory_id, level, thread_key, contact_key, channel,
                external_id, span_start, span_end, covered_count, token_estimate,
                pinned, state)
         VALUES (?1, ?2, ?3, ?4, NULL, NULL, ?5, ?6, ?7, ?8, 0, 'open')",
        rusqlite::params![
            record.id.to_string(),
            level,
            THREAD,
            CONTACT,
            record.created_at.to_rfc3339(),
            record.created_at.to_rfc3339(),
            covered_count,
            token_override.unwrap_or(token_estimate(&record.content) as i64),
        ],
    )
    .unwrap();
}

// T2 (spec section 8): a leaf round-trips with the spec's shape.
#[test]
fn leaf_write_round_trips_with_spec_shape() {
    let (store, path) = open("roundtrip");
    let content =
        "contact: can we move the deploy to friday?\ncookie: yes, friday works.".to_string();
    let outcome = leaf(&store, "email", THREAD, "msg-1", content.clone());
    assert!(!outcome.duplicate);

    let record = store
        .memory_store()
        .get(outcome.memory_id)
        .unwrap()
        .unwrap();
    assert_eq!(record.kind, MemoryKind::Episodic);
    assert_eq!(record.content, content);
    assert_eq!(record.provenance.origin, "channel:email:alex@example.com");
    assert_eq!(record.provenance.evidence, vec!["msg-1".to_string()]);
    assert_eq!(record.decay_rate, 0.0);

    let node = store.node(outcome.memory_id).unwrap().unwrap();
    assert_eq!(node.level, 0);
    assert_eq!(node.thread_key, THREAD);
    assert_eq!(node.contact_key, CONTACT);
    assert_eq!(node.channel.as_deref(), Some("email"));
    assert_eq!(node.external_id.as_deref(), Some("msg-1"));
    assert_eq!(node.token_estimate, token_estimate(&content));
    assert_eq!(node.state, NodeState::Open);
    cleanup(&path);
}

// T3: the (channel, external_id) pair is the idempotency key.
#[test]
fn duplicate_leaf_write_returns_same_id_and_writes_nothing() {
    let (store, path) = open("dup");
    let first = leaf(&store, "email", THREAD, "msg-1", sized("leaf-", 50));
    let second = leaf(&store, "email", THREAD, "msg-1", sized("leaf-", 50));
    assert!(!first.duplicate);
    assert!(second.duplicate);
    assert_eq!(first.memory_id, second.memory_id);
    let episodics = store
        .memory_store()
        .query_by_kind(MemoryKind::Episodic)
        .unwrap();
    assert_eq!(episodics.len(), 1);
    cleanup(&path);
}

// T1: the sidecar migration leaves pre-existing ADAM records
// untouched and tree-invisible.
#[test]
fn sidecar_opens_cleanly_over_existing_adam_database() {
    let path = temp_db("migration");
    let memory = MemoryStore::open(path.to_str().unwrap()).unwrap();
    let record = MemoryRecord::new(
        MemoryKind::Semantic,
        "pre-existing belief",
        vec![0.1, 0.2],
        0.8,
        Provenance {
            origin: "reflection:self".to_string(),
            evidence: vec![],
        },
        0.0,
    );
    memory.store(&record).unwrap();
    drop(memory);

    let tree = TreeStore::open(path.to_str().unwrap()).unwrap();
    let fetched = tree.memory_store().get(record.id).unwrap().unwrap();
    assert_eq!(fetched.content, "pre-existing belief");
    assert!(tree.node(record.id).unwrap().is_none());
    tree.assert_partition().unwrap();
    cleanup(&path);
}

// T6: passes fire at exactly the count, token, and age thresholds.
#[test]
fn triggers_fire_at_thresholds_and_not_before() {
    // Count trigger: 24 open leaves.
    let (store, path) = open("trigger-count");
    let mut ids = Vec::new();
    for i in 0..23 {
        ids.push(
            leaf(
                &store,
                "email",
                THREAD,
                &format!("c-{i}"),
                sized("leaf-", 40),
            )
            .memory_id,
        );
    }
    let report = store
        .run_due_passes("orchestrator", &FakeSummarizer, &HashEmbedder)
        .unwrap();
    assert!(report.summaries_created.is_empty());
    ids.push(leaf(&store, "email", THREAD, "c-23", sized("leaf-", 40)).memory_id);
    let report = store
        .run_due_passes("orchestrator", &FakeSummarizer, &HashEmbedder)
        .unwrap();
    assert_eq!(report.summaries_created.len(), 1);
    assert_eq!(report.nodes_covered, 24);
    for id in &ids {
        assert_eq!(store.node(*id).unwrap().unwrap().state, NodeState::Covered);
    }
    cleanup(&path);

    // Token trigger: 10 leaves x 600 tokens = 6,000, below the count
    // trigger, so only the token sum can fire it.
    let (store, path) = open("trigger-tokens");
    for i in 0..9 {
        leaf(
            &store,
            "email",
            THREAD,
            &format!("t-{i}"),
            sized("leaf-", 600),
        );
    }
    let report = store
        .run_due_passes("orchestrator", &FakeSummarizer, &HashEmbedder)
        .unwrap();
    assert!(report.summaries_created.is_empty());
    leaf(&store, "email", THREAD, "t-9", sized("leaf-", 600));
    let report = store
        .run_due_passes("orchestrator", &FakeSummarizer, &HashEmbedder)
        .unwrap();
    assert_eq!(report.summaries_created.len(), 1);
    cleanup(&path);

    // Age trigger: needs both 7+ days and at least 6 open leaves.
    let (store, path) = open("trigger-age");
    let old = Utc::now() - Duration::days(8);
    for i in 0..5 {
        leaf_at(&store, THREAD, &format!("a-{i}"), sized("leaf-", 40), old);
    }
    let report = store
        .run_due_passes("orchestrator", &FakeSummarizer, &HashEmbedder)
        .unwrap();
    assert!(report.summaries_created.is_empty());
    leaf_at(&store, THREAD, "a-5", sized("leaf-", 40), old);
    let report = store
        .run_due_passes("orchestrator", &FakeSummarizer, &HashEmbedder)
        .unwrap();
    assert_eq!(report.summaries_created.len(), 1);
    assert_eq!(report.nodes_covered, 6);
    cleanup(&path);
}

// T7: L2 fires on the open-child count/token triggers; L3 covers
// children from two channels under one contact.
#[test]
fn l2_and_l3_passes_cover_across_threads_and_channels() {
    let config = TreeConfig {
        child_count_trigger: 2,
        ..TreeConfig::default()
    };
    let path = temp_db("levels");
    let store = TreeStore::with_config(path.to_str().unwrap(), config).unwrap();
    let thread_b = "alex@example.com/invoice-question";

    // Thread A on email: two batches, each forced into an L1; the
    // second batch's run cascades into L2 by the count trigger.
    for batch in 0..2 {
        for i in 0..3 {
            leaf(
                &store,
                "email",
                THREAD,
                &format!("a-{batch}-{i}"),
                sized("leaf-", 600),
            );
        }
        store
            .summarize_now(
                &SummarizeScope::Thread(THREAD.to_string()),
                "orchestrator",
                &FakeSummarizer,
                &HashEmbedder,
            )
            .unwrap();
    }
    // Thread B on whatsapp, same contact.
    for batch in 0..2 {
        for i in 0..3 {
            leaf(
                &store,
                "whatsapp",
                thread_b,
                &format!("b-{batch}-{i}"),
                sized("leaf-", 600),
            );
        }
        store
            .summarize_now(
                &SummarizeScope::Thread(thread_b.to_string()),
                "orchestrator",
                &FakeSummarizer,
                &HashEmbedder,
            )
            .unwrap();
    }

    // L3 over the two open L2 nodes.
    let report = store
        .summarize_now(
            &SummarizeScope::Contact(CONTACT.to_string()),
            "orchestrator",
            &FakeSummarizer,
            &HashEmbedder,
        )
        .unwrap();
    assert_eq!(report.summaries_created.len(), 1);
    let l3 = report.summaries_created[0];
    assert_eq!(store.node(l3).unwrap().unwrap().level, 3);
    store.assert_partition().unwrap();

    // Drill the L3 down to leaves: both channels must appear.
    let mut origins = HashSet::new();
    let mut frontier = vec![l3];
    let mut leaf_count = 0;
    while let Some(id) = frontier.pop() {
        let drill = store.drill_down(id, 1_000_000).unwrap();
        for child in drill.children {
            if child.level == Some(0) {
                leaf_count += 1;
                origins.insert(child.origin.clone());
            } else {
                frontier.push(child.memory_id);
            }
        }
    }
    assert_eq!(leaf_count, 12);
    assert!(origins.iter().any(|o| o.starts_with("channel:email:")));
    assert!(origins.iter().any(|o| o.starts_with("channel:whatsapp:")));
    cleanup(&path);
}

// T7 (token arm): L2 fires on the open-child token sum alone. The
// L1s are built under the default config, whose child token trigger
// (8,000) their combined ~430 tokens cannot reach; a second instance
// with a low trigger then finds the L2 pass due on tokens alone.
#[test]
fn l2_fires_on_child_token_sum() {
    let path = temp_db("l2tokens");
    let store = TreeStore::open(path.to_str().unwrap()).unwrap();
    for batch in 0..2 {
        for i in 0..3 {
            leaf(
                &store,
                "email",
                THREAD,
                &format!("x-{batch}-{i}"),
                sized("leaf-", 600),
            );
        }
        let report = store
            .summarize_now(
                &SummarizeScope::Thread(THREAD.to_string()),
                "orchestrator",
                &FakeSummarizer,
                &HashEmbedder,
            )
            .unwrap();
        assert_eq!(report.summaries_created.len(), 1, "L1 only, no cascade");
    }
    let config = TreeConfig {
        child_token_trigger: 100,
        ..TreeConfig::default()
    };
    let passer = TreeStore::with_config(path.to_str().unwrap(), config).unwrap();
    // Two open L1 nodes, count 2 < 8: only the token trigger can fire.
    let report = passer
        .run_due_passes("orchestrator", &FakeSummarizer, &HashEmbedder)
        .unwrap();
    assert_eq!(report.summaries_created.len(), 1);
    let l2 = report.summaries_created[0];
    assert_eq!(passer.node(l2).unwrap().unwrap().level, 2);
    cleanup(&path);
}

// T8 + T9: budget bands are respected and no record is ever cut
// mid-record.
#[test]
fn read_context_respects_bands_and_never_truncates() {
    let (store, path) = open("budget");
    for i in 0..4 {
        leaf(
            &store,
            "email",
            THREAD,
            &format!("old-{i}"),
            sized("leaf-", 60),
        );
    }
    let report = store
        .summarize_now(
            &SummarizeScope::Thread(THREAD.to_string()),
            "orchestrator",
            &FakeSummarizer,
            &HashEmbedder,
        )
        .unwrap();
    let l1 = report.summaries_created[0];
    let l1_tokens = store.node(l1).unwrap().unwrap().token_estimate;
    for i in 0..4 {
        leaf(
            &store,
            "email",
            THREAD,
            &format!("new-{i}"),
            sized("leaf-", 60),
        );
    }

    // Generous budget: everything fits, byte for byte.
    let mut request = ContextRequest::for_thread(CONTACT, THREAD);
    request.token_budget = 2000;
    let bundle = store.read_context(&HashEmbedder, &request).unwrap();
    assert_eq!(bundle.recent_leaves.len(), 8);
    assert_eq!(bundle.summaries.len(), 1);
    assert!(bundle.recall.is_empty());
    assert_eq!(bundle.tokens_used, 8 * 60 + l1_tokens);
    assert!(bundle.tokens_used <= 2000);
    assert!(!bundle.overflow);
    for band in [&bundle.recent_leaves, &bundle.summaries, &bundle.recall] {
        for record in band {
            let stored = store.memory_store().get(record.memory_id).unwrap().unwrap();
            assert_eq!(record.content, stored.content);
        }
    }

    // Tight budget: caps are 180 / 160 / 60 of the 400 total.
    request.token_budget = 400;
    request.query = Some("deploy invoice".to_string());
    let bundle = store.read_context(&HashEmbedder, &request).unwrap();
    let band_sum = |records: &[adam_tree::ContextRecord]| -> usize {
        records.iter().map(|r| r.token_estimate).sum()
    };
    assert_eq!(bundle.recent_leaves.len(), 3);
    assert!(band_sum(&bundle.recent_leaves) <= 180);
    assert_eq!(bundle.summaries.len(), 1);
    assert!(band_sum(&bundle.summaries) <= 160);
    assert_eq!(bundle.recall.len(), 1);
    assert!(band_sum(&bundle.recall) <= 60);
    assert!(bundle.tokens_used <= 400);
    // The recall hit must not duplicate a band 1 or 2 record.
    let mut seen: HashSet<MemoryId> = HashSet::new();
    for band in [&bundle.recent_leaves, &bundle.summaries, &bundle.recall] {
        for record in band {
            assert!(seen.insert(record.memory_id), "duplicate across bands");
        }
    }
    cleanup(&path);
}

// T8 (exception arm): a single summary larger than its whole band is
// returned alone, flagged as overflow.
#[test]
fn oversize_summary_is_returned_alone_with_overflow_flag() {
    let (store, path) = open("oversize-band");
    let record = MemoryRecord::new(
        MemoryKind::Semantic,
        "x".repeat(20_000),
        vec![1.0, 0.0],
        0.9,
        Provenance {
            origin: "summary:tree:L1".to_string(),
            evidence: vec![],
        },
        0.0,
    );
    store.memory_store().store(&record).unwrap();
    plant_node(&path, &record, 1, 3, Some(5_000));

    let mut request = ContextRequest::for_thread(CONTACT, THREAD);
    request.token_budget = 1000;
    let bundle = store.read_context(&HashEmbedder, &request).unwrap();
    assert!(bundle.overflow);
    assert_eq!(bundle.summaries.len(), 1);
    assert_eq!(bundle.summaries[0].memory_id, record.id);
    assert!(bundle.tokens_used >= 5_000);
    cleanup(&path);
}

// T10: drill-down from an L2 returns, recursively, exactly the
// original leaves in original order. Summarization is lossless in
// the only sense that matters operationally.
#[test]
fn drill_down_recovers_verbatim_leaves_in_order() {
    let config = TreeConfig {
        child_count_trigger: 2,
        ..TreeConfig::default()
    };
    let path = temp_db("drill");
    let store = TreeStore::with_config(path.to_str().unwrap(), config).unwrap();
    let base = Utc::now() - Duration::hours(2);
    let mut contents = Vec::new();
    for batch in 0..2 {
        for i in 0..3 {
            let n = batch * 3 + i;
            let content = sized(&format!("marker-{n}-"), 200);
            contents.push(content.clone());
            leaf_at(
                &store,
                THREAD,
                &format!("d-{n}"),
                content,
                base + Duration::seconds(n * 10),
            );
        }
        store
            .summarize_now(
                &SummarizeScope::Thread(THREAD.to_string()),
                "orchestrator",
                &FakeSummarizer,
                &HashEmbedder,
            )
            .unwrap();
    }
    // Find the L2: the open level-2 node of the thread.
    let l2 = store
        .search_tree(&HashEmbedder, "marker", None, 50)
        .unwrap()
        .into_iter()
        .find(|hit| hit.record.level == Some(2))
        .map(|hit| hit.record.memory_id)
        .expect("an L2 summary exists");

    let level1 = store.drill_down(l2, 1_000_000).unwrap();
    assert_eq!(level1.children.len(), 2);
    let mut recovered = Vec::new();
    for child in &level1.children {
        let leaves = store.drill_down(child.memory_id, 1_000_000).unwrap();
        assert_eq!(leaves.children.len(), 3);
        recovered.extend(leaves.children.iter().map(|c| c.content.clone()));
    }
    assert_eq!(recovered, contents);
    cleanup(&path);
}

// T11: at equal similarity, a covering summary ranks above its
// covered leaves in search.
#[test]
fn search_ranks_summary_above_covered_leaves_at_equal_similarity() {
    let (store, path) = open("search-rank");
    let embedder = adam_tree::ConstEmbedder(vec![1.0, 0.0]);
    let mut leaf_ids = Vec::new();
    for i in 0..4 {
        let outcome = store
            .write_leaf(
                &embedder,
                &LeafWrite::new(
                    "email",
                    CONTACT,
                    THREAD,
                    format!("s-{i}"),
                    sized("leaf-", 200),
                ),
            )
            .unwrap();
        leaf_ids.push(outcome.memory_id);
    }
    let report = store
        .summarize_now(
            &SummarizeScope::Thread(THREAD.to_string()),
            "orchestrator",
            &FakeSummarizer,
            &embedder,
        )
        .unwrap();
    let l1 = report.summaries_created[0];

    let hits = store.search_tree(&embedder, "anything", None, 10).unwrap();
    assert_eq!(hits.len(), 5);
    let position = |id: MemoryId| hits.iter().position(|h| h.record.memory_id == id).unwrap();
    for leaf_id in &leaf_ids {
        assert!(position(l1) < position(*leaf_id));
    }
    cleanup(&path);
}

// T13: the never-summarize set is excluded from pass input even when
// a pass is forced.
#[test]
fn never_summarize_set_is_excluded_from_forced_passes() {
    let (store, path) = open("never");
    let mut normal = Vec::new();
    for i in 0..3 {
        normal.push(
            leaf(
                &store,
                "email",
                THREAD,
                &format!("n-{i}"),
                sized("leaf-", 150),
            )
            .memory_id,
        );
    }
    // A pinned leaf.
    let mut pinned_input = LeafWrite::new("email", CONTACT, THREAD, "pin-1", sized("pinned-", 150));
    pinned_input.pinned = true;
    let pinned = store
        .write_leaf(&HashEmbedder, &pinned_input)
        .unwrap()
        .memory_id;
    // A leaf carrying a credential-shaped string.
    let credential = leaf(
        &store,
        "email",
        THREAD,
        "cred-1",
        "contact: here is the key AKIAIOSFODNN7EXAMPLE for the deploy".to_string(),
    )
    .memory_id;
    // An approval record planted with a sidecar row.
    let approval = MemoryRecord::new(
        MemoryKind::Episodic,
        "approval granted for payment 42",
        vec![0.5, 0.5],
        1.0,
        Provenance {
            origin: "approval:payment:42".to_string(),
            evidence: vec![],
        },
        0.0,
    );
    store.memory_store().store(&approval).unwrap();
    plant_node(&path, &approval, 0, 0, None);
    // A SelfKnowledge record planted with a sidecar row.
    let self_knowledge = MemoryRecord::new(
        MemoryKind::SelfKnowledge,
        "I am strong at systems engineering.",
        vec![0.5, 0.5],
        0.95,
        Provenance {
            origin: "reflection:self".to_string(),
            evidence: vec![],
        },
        0.0,
    );
    store.memory_store().store(&self_knowledge).unwrap();
    plant_node(&path, &self_knowledge, 0, 0, None);

    let report = store
        .summarize_now(
            &SummarizeScope::Thread(THREAD.to_string()),
            "orchestrator",
            &FakeSummarizer,
            &HashEmbedder,
        )
        .unwrap();
    assert_eq!(report.summaries_created.len(), 1);
    assert_eq!(report.nodes_covered, 3);
    for id in &normal {
        assert_eq!(store.node(*id).unwrap().unwrap().state, NodeState::Covered);
    }
    for id in [pinned, credential, approval.id, self_knowledge.id] {
        assert_eq!(
            store.node(id).unwrap().unwrap().state,
            NodeState::Open,
            "protected node {id} must stay open"
        );
    }
    let summary = store
        .memory_store()
        .get(report.summaries_created[0])
        .unwrap()
        .unwrap();
    assert!(!summary.content.contains("AKIA"));
    store.assert_partition().unwrap();
    cleanup(&path);
}

// T14: an oversize summary is rejected, retried once, and on second
// failure the children stay open and the failure is reported.
#[test]
fn oversize_summary_is_rejected_after_one_retry() {
    use std::cell::RefCell;

    struct StubbornSummarizer {
        calls: RefCell<usize>,
    }
    impl adam_tree::Summarizer for StubbornSummarizer {
        fn summarize(&self, _request: &adam_tree::SummaryRequest) -> Result<String, TreeError> {
            *self.calls.borrow_mut() += 1;
            Ok("x".repeat(100_000))
        }
    }

    let (store, path) = open("oversize-pass");
    let mut ids = Vec::new();
    for i in 0..3 {
        ids.push(
            leaf(
                &store,
                "email",
                THREAD,
                &format!("o-{i}"),
                sized("leaf-", 200),
            )
            .memory_id,
        );
    }
    let stubborn = StubbornSummarizer {
        calls: RefCell::new(0),
    };
    let report = store
        .summarize_now(
            &SummarizeScope::Thread(THREAD.to_string()),
            "orchestrator",
            &stubborn,
            &HashEmbedder,
        )
        .unwrap();
    assert_eq!(*stubborn.calls.borrow(), 2, "one attempt plus one retry");
    assert!(report.summaries_created.is_empty());
    assert_eq!(
        report.skipped_reason.as_deref(),
        Some("summary oversize after retry")
    );
    for id in &ids {
        assert_eq!(store.node(*id).unwrap().unwrap().state, NodeState::Open);
    }
    cleanup(&path);
}

// T4: a crash at any pass step converges on the next run through
// adoption: exactly one summary, every child covered once.
#[test]
fn crash_mid_pass_converges_through_adoption() {
    let (store, path) = open("crash");
    let base = Utc::now() - Duration::hours(3);
    let mut leaf_ids = Vec::new();
    for i in 0..3 {
        leaf_ids.push(
            leaf_at(
                &store,
                THREAD,
                &format!("k-{i}"),
                sized("leaf-", 150),
                base + Duration::seconds(i * 10),
            )
            .memory_id,
        );
    }
    // Simulate the crashed pass: summary stored, one edge written,
    // no sidecar row for the summary, children still open.
    let summary = MemoryRecord::new(
        MemoryKind::Semantic,
        "crashed pass summary",
        vec![0.3, 0.7],
        0.9,
        Provenance {
            origin: "summary:tree:L1".to_string(),
            evidence: leaf_ids.iter().map(|id| id.to_string()).collect(),
        },
        0.0,
    );
    store.memory_store().store(&summary).unwrap();
    store
        .memory_store()
        .relate(
            leaf_ids[0],
            summary.id,
            adam_memory::RelationKind::Generalizes,
        )
        .unwrap();

    let adopted = store.adopt_orphans().unwrap();
    assert_eq!(adopted.summaries_adopted, 1);
    assert_eq!(adopted.children_flipped, 3);
    let node = store.node(summary.id).unwrap().unwrap();
    assert_eq!(node.level, 1);
    assert_eq!(node.covered_count, 3);
    assert_eq!(node.span_start, base);
    for id in &leaf_ids {
        assert_eq!(store.node(*id).unwrap().unwrap().state, NodeState::Covered);
    }
    store.assert_partition().unwrap();

    // A later pass must not summarize those leaves a second time.
    let report = store
        .summarize_now(
            &SummarizeScope::Thread(THREAD.to_string()),
            "orchestrator",
            &FakeSummarizer,
            &HashEmbedder,
        )
        .unwrap();
    assert!(report.summaries_created.is_empty());
    cleanup(&path);
}

// T15: the summarizer lease is single-holder; expiry frees it.
#[test]
fn summarizer_lease_is_single_holder() {
    let path = temp_db("lease");
    let a = TreeStore::open(path.to_str().unwrap()).unwrap();
    let b = TreeStore::open(path.to_str().unwrap()).unwrap();

    let lease = a.try_acquire_lease("orchestrator").unwrap().unwrap();
    assert!(b.try_acquire_lease("other").unwrap().is_none());
    let report = b
        .summarize_now(
            &SummarizeScope::Thread(THREAD.to_string()),
            "other",
            &FakeSummarizer,
            &HashEmbedder,
        )
        .unwrap();
    assert_eq!(report.skipped_reason.as_deref(), Some("lease held"));
    a.release_lease(&lease).unwrap();
    let lease_b = b.try_acquire_lease("other").unwrap().unwrap();
    b.release_lease(&lease_b).unwrap();
    cleanup(&path);

    // Expiry: a lease nobody releases lapses on its TTL.
    let config = TreeConfig {
        lease_ttl: std::time::Duration::from_millis(80),
        ..TreeConfig::default()
    };
    let path = temp_db("lease-expiry");
    let c = TreeStore::with_config(path.to_str().unwrap(), config.clone()).unwrap();
    let d = TreeStore::with_config(path.to_str().unwrap(), config).unwrap();
    assert!(c.try_acquire_lease("first").unwrap().is_some());
    std::thread::sleep(std::time::Duration::from_millis(150));
    assert!(d.try_acquire_lease("second").unwrap().is_some());
    cleanup(&path);
}

// T18: decay never touches tree leaves (decay_rate 0.0) or sidecar
// state, while ADAM's own decaying records still decay.
#[test]
fn decay_pass_leaves_the_tree_untouched() {
    let (store, path) = open("decay");
    let old = Utc::now() - Duration::days(365);
    let mut ids = Vec::new();
    for i in 0..3 {
        ids.push(leaf_at(&store, THREAD, &format!("z-{i}"), sized("leaf-", 60), old).memory_id);
    }
    let mut decaying = MemoryRecord::new(
        MemoryKind::Episodic,
        "a stale non-tree episode",
        vec![0.2, 0.8],
        1.0,
        Provenance {
            origin: "tool:test".to_string(),
            evidence: vec![],
        },
        1.0,
    );
    decaying.last_accessed_at = Utc::now() - Duration::days(10);
    store.memory_store().store(&decaying).unwrap();

    let updated = store.memory_store().decay_pass(Utc::now()).unwrap();
    assert_eq!(updated, 1, "only the non-tree record decays");
    for id in &ids {
        let record = store.memory_store().get(*id).unwrap().unwrap();
        assert_eq!(record.confidence, 1.0);
        assert_eq!(store.node(*id).unwrap().unwrap().state, NodeState::Open);
    }
    cleanup(&path);
}

// set_pinned: the one deliberate sidecar mutation outside a pass.
#[test]
fn set_pinned_round_trips_and_rejects_unknown_ids() {
    let (store, path) = open("pin");
    let outcome = leaf(&store, "email", THREAD, "p-1", sized("leaf-", 60));
    store.set_pinned(outcome.memory_id, true).unwrap();
    assert!(store.node(outcome.memory_id).unwrap().unwrap().pinned);
    store.set_pinned(outcome.memory_id, false).unwrap();
    assert!(!store.node(outcome.memory_id).unwrap().unwrap().pinned);
    let err = store.set_pinned(MemoryId::new_v4(), true).unwrap_err();
    assert!(matches!(err, TreeError::NotInTree(_)));
    cleanup(&path);
}

// ---------------------------------------------------------------------
// Mutation-gate strengthening (PR #24): boundary and contract tests
// for the summarizer's report, ratios, and retry discipline, the
// read path's bands, search ranking, drill-down budgets, and the
// partition checker. Each test pins behavior the spec requires, at
// the exact boundary where an off-by-one or flipped operator would
// change the outcome.
// ---------------------------------------------------------------------

struct FixedSummarizer(usize);

impl Summarizer for FixedSummarizer {
    fn summarize(&self, _request: &SummaryRequest) -> Result<String, TreeError> {
        Ok(sized("fixed summary ", self.0))
    }
}

struct RecordingSummarizer {
    requests: RefCell<Vec<SummaryRequest>>,
    sizes: Vec<usize>,
}

impl Summarizer for RecordingSummarizer {
    fn summarize(&self, request: &SummaryRequest) -> Result<String, TreeError> {
        self.requests.borrow_mut().push(request.clone());
        let tokens = self.sizes[(request.attempt as usize - 1).min(self.sizes.len() - 1)];
        Ok(sized("recorded summary ", tokens))
    }
}

fn trigger_config() -> TreeConfig {
    TreeConfig {
        leaf_count_trigger: 3,
        child_count_trigger: 2,
        ..TreeConfig::default()
    }
}

/// Two batches of three leaves in `thread`, summarized after each
/// batch. Returns (first L1 id, L2 id). Requires a store whose
/// child_count_trigger is 2.
fn build_l2(store: &TreeStore, thread: &str, tag: &str) -> (MemoryId, MemoryId) {
    let mut l1 = None;
    let mut l2 = None;
    for batch in 0..2 {
        for i in 0..3 {
            leaf(
                store,
                "email",
                thread,
                &format!("{tag}-{batch}-{i}"),
                sized("leaf-", 40),
            );
        }
        let report = store
            .summarize_now(
                &SummarizeScope::Thread(thread.to_string()),
                "orchestrator",
                &FakeSummarizer,
                &HashEmbedder,
            )
            .unwrap();
        if batch == 0 {
            l1 = Some(report.summaries_created[0]);
        }
        l2 = Some(*report.summaries_created.last().unwrap());
    }
    (l1.unwrap(), l2.unwrap())
}

#[test]
fn pass_report_is_exact_on_success_and_on_nothing_due() {
    let (store, path) = open("report");
    let report = store
        .summarize_now(
            &SummarizeScope::Thread(THREAD.to_string()),
            "orchestrator",
            &FakeSummarizer,
            &HashEmbedder,
        )
        .unwrap();
    assert_eq!(report.passes_run, 0);
    assert_eq!(report.skipped_reason.as_deref(), Some("nothing due"));

    let base = Utc::now() - Duration::hours(2);
    for i in 0..3 {
        leaf_at(
            &store,
            THREAD,
            &format!("r-{i}"),
            sized("leaf-", 60),
            base + Duration::seconds(i * 10),
        );
    }
    let report = store
        .summarize_now(
            &SummarizeScope::Thread(THREAD.to_string()),
            "orchestrator",
            &FakeSummarizer,
            &HashEmbedder,
        )
        .unwrap();
    assert_eq!(report.passes_run, 1);
    assert_eq!(report.summaries_created.len(), 1);
    assert_eq!(report.nodes_covered, 3);
    assert_eq!(report.skipped_reason, None);
    cleanup(&path);
}

#[test]
fn oversize_summary_retries_once_with_a_note_then_skips() {
    let (store, path) = open("oversize");
    let base = Utc::now() - Duration::hours(2);
    let mut leaf_ids = Vec::new();
    for i in 0..3 {
        leaf_ids.push(
            leaf_at(
                &store,
                THREAD,
                &format!("o-{i}"),
                sized("leaf-", 100),
                base + Duration::seconds(i * 10),
            )
            .memory_id,
        );
    }
    // Covered total is 300 tokens, so the L1 target is 36; the
    // summarizer insists on 50.
    let recorder = RecordingSummarizer {
        requests: RefCell::new(Vec::new()),
        sizes: vec![50, 50],
    };
    let report = store
        .summarize_now(
            &SummarizeScope::Thread(THREAD.to_string()),
            "orchestrator",
            &recorder,
            &HashEmbedder,
        )
        .unwrap();
    assert!(report.summaries_created.is_empty());
    assert_eq!(
        report.skipped_reason.as_deref(),
        Some("summary oversize after retry")
    );
    let requests = recorder.requests.borrow();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].attempt, 1);
    assert!(requests[0].retry_note.is_none());
    assert_eq!(requests[1].attempt, 2);
    assert!(requests[1].retry_note.is_some());
    drop(requests);
    for id in &leaf_ids {
        assert_eq!(store.node(*id).unwrap().unwrap().state, NodeState::Open);
    }
    cleanup(&path);
}

#[test]
fn l1_pass_uses_the_l1_target_ratio() {
    let (store, path) = open("ratio");
    let base = Utc::now() - Duration::hours(2);
    for i in 0..3 {
        leaf_at(
            &store,
            THREAD,
            &format!("t-{i}"),
            sized("leaf-", 100),
            base + Duration::seconds(i * 10),
        );
    }
    // 50 tokens fits the above-L1 target (20% of 300 = 60) but not
    // the L1 target (12% of 300 = 36): the pass must reject it.
    let report = store
        .summarize_now(
            &SummarizeScope::Thread(THREAD.to_string()),
            "orchestrator",
            &FixedSummarizer(50),
            &HashEmbedder,
        )
        .unwrap();
    assert!(report.summaries_created.is_empty());
    assert_eq!(
        report.skipped_reason.as_deref(),
        Some("summary oversize after retry")
    );
    cleanup(&path);
}

#[test]
fn summary_children_are_covered_in_span_order() {
    let (store, path) = open("span-order");
    let base = Utc::now() - Duration::hours(3);
    let late = leaf_at(
        &store,
        THREAD,
        "s-late",
        sized("late-", 60),
        base + Duration::seconds(200),
    );
    let early = leaf_at(&store, THREAD, "s-early", sized("early-", 60), base);
    let mid = leaf_at(
        &store,
        THREAD,
        "s-mid",
        sized("mid-", 60),
        base + Duration::seconds(100),
    );
    let report = store
        .summarize_now(
            &SummarizeScope::Thread(THREAD.to_string()),
            "orchestrator",
            &FakeSummarizer,
            &HashEmbedder,
        )
        .unwrap();
    let summary = store
        .memory_store()
        .get(report.summaries_created[0])
        .unwrap()
        .unwrap();
    assert_eq!(
        summary.provenance.evidence,
        vec![
            early.memory_id.to_string(),
            mid.memory_id.to_string(),
            late.memory_id.to_string(),
        ],
        "children are summarized in span order, not insertion order"
    );
    cleanup(&path);
}

#[test]
fn partition_check_catches_a_second_parent_and_ignores_non_tree_edges() {
    let path = temp_db("partition");
    let store = TreeStore::with_config(path.to_str().unwrap(), trigger_config()).unwrap();
    const THREAD_B: &str = "alex@example.com/other-thread";
    let (l1a, _l2a) = build_l2(&store, THREAD, "pa");
    let (_l1b, l2b) = build_l2(&store, THREAD_B, "pb");
    store.assert_partition().unwrap();

    // A Generalizes edge to a record with no sidecar row is not a
    // tree parent and must not disturb the partition.
    let stray = MemoryRecord::new(
        MemoryKind::Semantic,
        "stray record",
        vec![0.1],
        0.5,
        Provenance {
            origin: "manual:note".to_string(),
            evidence: vec![],
        },
        0.0,
    );
    store.memory_store().store(&stray).unwrap();
    store
        .memory_store()
        .relate(l1a, stray.id, RelationKind::Generalizes)
        .unwrap();
    store.assert_partition().unwrap();

    // A second tree parent for an already-covered node breaks it.
    store
        .memory_store()
        .relate(l1a, l2b, RelationKind::Generalizes)
        .unwrap();
    assert!(store.assert_partition().is_err());
    cleanup(&path);
}

#[test]
fn l3_summary_of_a_single_thread_keeps_the_thread_key() {
    let path = temp_db("l3-thread");
    let store = TreeStore::with_config(path.to_str().unwrap(), trigger_config()).unwrap();
    let (_l1a, _l2a) = build_l2(&store, THREAD, "la");
    let (_l1c, _l2b) = build_l2(&store, THREAD, "lb");
    let report = store
        .summarize_now(
            &SummarizeScope::Contact(CONTACT.to_string()),
            "orchestrator",
            &FakeSummarizer,
            &HashEmbedder,
        )
        .unwrap();
    assert_eq!(report.summaries_created.len(), 1);
    let node = store.node(report.summaries_created[0]).unwrap().unwrap();
    assert_eq!(node.level, 3);
    assert_eq!(
        node.thread_key, THREAD,
        "an L3 over one thread's summaries keeps that thread, not the contact bucket"
    );
    cleanup(&path);
}

#[test]
fn orphan_summary_adopts_at_its_own_level() {
    let (store, path) = open("orphan-l2");
    let base = Utc::now() - Duration::hours(4);
    let mut l1_ids = Vec::new();
    for batch in 0..2 {
        for i in 0..3 {
            leaf_at(
                &store,
                THREAD,
                &format!("ol-{batch}-{i}"),
                sized("leaf-", 40),
                base + Duration::seconds((batch * 3 + i) * 10),
            );
        }
        let report = store
            .summarize_now(
                &SummarizeScope::Thread(THREAD.to_string()),
                "orchestrator",
                &FakeSummarizer,
                &HashEmbedder,
            )
            .unwrap();
        l1_ids.push(report.summaries_created[0]);
    }
    // An L2 summary record whose sidecar row never landed.
    let planted = MemoryRecord::new(
        MemoryKind::Semantic,
        "planted L2 summary",
        vec![0.4, 0.6],
        0.9,
        Provenance {
            origin: "summary:tree:L2".to_string(),
            evidence: l1_ids.iter().map(|id| id.to_string()).collect(),
        },
        0.0,
    );
    store.memory_store().store(&planted).unwrap();
    let adopted = store.adopt_orphans().unwrap();
    assert_eq!(adopted.summaries_adopted, 1);
    let node = store.node(planted.id).unwrap().unwrap();
    assert_eq!(node.level, 2);
    cleanup(&path);
}

#[test]
fn band1_includes_a_leaf_that_exactly_fits() {
    let (store, path) = open("band1-fit");
    let base = Utc::now() - Duration::hours(1);
    leaf_at(&store, THREAD, "b-small", sized("small-", 10), base);
    let big = leaf_at(
        &store,
        THREAD,
        "b-big",
        sized("big-", 100),
        base + Duration::seconds(60),
    );
    // Budget 223: band 1 cap is exactly 100 tokens. The newest
    // leaf (100 tokens) fits exactly; the older small leaf then
    // no longer fits.
    let mut request = ContextRequest::for_thread(CONTACT, THREAD);
    request.token_budget = 223;
    let bundle = store.read_context(&HashEmbedder, &request).unwrap();
    assert_eq!(bundle.recent_leaves.len(), 1);
    assert_eq!(bundle.recent_leaves[0].memory_id, big.memory_id);
    assert_eq!(bundle.tokens_used, 100);
    cleanup(&path);
}

#[test]
fn band2_single_oversize_summary_overflows_and_exact_fit_does_not() {
    let (store, path) = open("band2-overflow");
    let base = Utc::now() - Duration::hours(2);
    for i in 0..3 {
        leaf_at(
            &store,
            THREAD,
            &format!("v-{i}"),
            sized("leaf-", 120),
            base + Duration::seconds(i * 10),
        );
    }
    let report = store
        .summarize_now(
            &SummarizeScope::Thread(THREAD.to_string()),
            "orchestrator",
            &FakeSummarizer,
            &HashEmbedder,
        )
        .unwrap();
    let summary_id = report.summaries_created[0];
    // The summary is 44 tokens (12% of 360, ceiling). The leaves
    // (120 each) never fit band 1 at these budgets.
    let mut request = ContextRequest::for_thread(CONTACT, THREAD);
    request.token_budget = 100; // band 2 cap 40 < 44
    let bundle = store.read_context(&HashEmbedder, &request).unwrap();
    assert_eq!(bundle.summaries.len(), 1);
    assert_eq!(bundle.summaries[0].memory_id, summary_id);
    assert!(bundle.overflow);

    let mut request = ContextRequest::for_thread(CONTACT, THREAD);
    request.token_budget = 110; // band 2 cap exactly 44
    let bundle = store.read_context(&HashEmbedder, &request).unwrap();
    assert_eq!(bundle.summaries.len(), 1);
    assert!(!bundle.overflow);
    cleanup(&path);
}

#[test]
fn summary_chain_includes_sibling_threads_and_contact_scope_stays_summaries() {
    let path = temp_db("chain");
    let store = TreeStore::with_config(path.to_str().unwrap(), trigger_config()).unwrap();
    const THREAD_B: &str = "alex@example.com/other-thread";
    let (_l1a, l2a) = build_l2(&store, THREAD, "ca");
    let (_l1b, l2b) = build_l2(&store, THREAD_B, "cb");

    let mut request = ContextRequest::for_thread(CONTACT, THREAD);
    request.token_budget = 1_000_000;
    let bundle = store.read_context(&HashEmbedder, &request).unwrap();
    let chain_ids: Vec<MemoryId> = bundle.summaries.iter().map(|r| r.memory_id).collect();
    assert_eq!(
        chain_ids,
        vec![l2a, l2b],
        "the active thread's L2 first, then the sibling thread's L2"
    );

    let mut request = ContextRequest::for_thread(CONTACT, THREAD);
    request.thread_key = None;
    request.token_budget = 1_000_000;
    let bundle = store.read_context(&HashEmbedder, &request).unwrap();
    assert!(bundle
        .summaries
        .iter()
        .all(|r| r.level.is_some_and(|l| l >= 1)));
    let mut ids: HashSet<MemoryId> = bundle.summaries.iter().map(|r| r.memory_id).collect();
    assert!(ids.remove(&l2a) && ids.remove(&l2b) && ids.is_empty());
    cleanup(&path);
}

#[test]
fn search_tree_filters_by_contact() {
    let (store, path) = open("search-contact");
    leaf(&store, "email", THREAD, "s-a", sized("alpha leaf ", 40));
    let bob = LeafWrite::new(
        "email",
        "bob@example.com",
        "bob@example.com/chat",
        "s-b",
        sized("alpha leaf ", 40),
    );
    store.write_leaf(&HashEmbedder, &bob).unwrap();
    let hits = store
        .search_tree(&HashEmbedder, "alpha", Some(CONTACT), 10)
        .unwrap();
    assert!(!hits.is_empty());
    assert!(hits.iter().all(|h| h.record.origin.ends_with(CONTACT)));
    cleanup(&path);
}

#[test]
fn search_tree_ranks_covering_summary_before_its_covered_leaves() {
    let (store, path) = open("search-demote");
    let emb = ConstEmbedder(vec![0.6, 0.8]);
    for i in 0..3 {
        store
            .write_leaf(
                &emb,
                &LeafWrite::new(
                    "email",
                    CONTACT,
                    THREAD,
                    format!("d-{i}"),
                    sized("leaf-", 40),
                ),
            )
            .unwrap();
    }
    store
        .summarize_now(
            &SummarizeScope::Thread(THREAD.to_string()),
            "orchestrator",
            &FakeSummarizer,
            &emb,
        )
        .unwrap();
    // Every record scores identically, so only the demotion rule
    // can put the summary first.
    let hits = store
        .search_tree(&emb, "anything", Some(CONTACT), 10)
        .unwrap();
    assert_eq!(hits.len(), 4);
    assert_eq!(hits[0].record.level, Some(1));
    cleanup(&path);
}

#[test]
fn drill_down_respects_the_budget_exactly() {
    let (store, path) = open("drill");
    let base = Utc::now() - Duration::hours(2);
    for i in 0..3 {
        leaf_at(
            &store,
            THREAD,
            &format!("dd-{i}"),
            sized("leaf-", 10),
            base + Duration::seconds(i * 10),
        );
    }
    let report = store
        .summarize_now(
            &SummarizeScope::Thread(THREAD.to_string()),
            "orchestrator",
            &FakeSummarizer,
            &HashEmbedder,
        )
        .unwrap();
    let summary_id = report.summaries_created[0];

    let out = store.drill_down(summary_id, 20).unwrap();
    assert_eq!(out.children.len(), 2);
    assert_eq!(out.tokens_used, 20);

    let out = store.drill_down(summary_id, 15).unwrap();
    assert_eq!(out.children.len(), 1);
    assert_eq!(out.tokens_used, 10);

    let out = store.drill_down(summary_id, 30).unwrap();
    assert_eq!(out.children.len(), 3);
    assert_eq!(out.tokens_used, 30);
    cleanup(&path);
}

#[test]
fn band3_pinned_leaf_of_the_contact_wins_the_cap() {
    let (store, path) = open("band3-pinned");
    let emb = ConstEmbedder(vec![0.6, 0.8]);
    const THREAD_B: &str = "alex@example.com/other-thread";
    let write = |input: &LeafWrite| store.write_leaf(&emb, input).unwrap();

    // A pinned leaf of ANOTHER contact, written first.
    let mut bob = LeafWrite::new(
        "email",
        "bob@example.com",
        "bob@example.com/chat",
        "pb-1",
        sized("pinned bob ", 10),
    );
    bob.pinned = true;
    write(&bob);

    // A summary in the other thread, pinned after creation.
    for i in 0..3 {
        write(&LeafWrite::new(
            "email",
            CONTACT,
            THREAD_B,
            format!("ps-{i}"),
            sized("t2 leaf ", 10),
        ));
    }
    let report = store
        .summarize_now(
            &SummarizeScope::Thread(THREAD_B.to_string()),
            "orchestrator",
            &FakeSummarizer,
            &emb,
        )
        .unwrap();
    store.set_pinned(report.summaries_created[0], true).unwrap();

    // An unpinned leaf of the contact, then the pinned leaf.
    write(&LeafWrite::new(
        "email",
        CONTACT,
        THREAD_B,
        "u-1",
        sized("plain leaf ", 10),
    ));
    let mut pinned = LeafWrite::new("email", CONTACT, THREAD_B, "p-1", sized("pinned leaf ", 10));
    pinned.pinned = true;
    let pinned_outcome = write(&pinned);

    // Budget 67: bands are 30 / 26 / 11. The active thread is
    // empty and has no summaries, so only band 3 can fill, and its
    // 11 tokens fit exactly one 10-token record: the contact's
    // pinned leaf must win it, ahead of the other contact's pinned
    // leaf, the pinned summary, and the plain leaf.
    let mut request = ContextRequest::for_thread(CONTACT, THREAD);
    request.token_budget = 67;
    request.query = Some("anything".to_string());
    let bundle = store.read_context(&emb, &request).unwrap();
    let recall_ids: Vec<MemoryId> = bundle.recall.iter().map(|r| r.memory_id).collect();
    assert_eq!(recall_ids, vec![pinned_outcome.memory_id]);
    cleanup(&path);
}
