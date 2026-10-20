//! Concurrency tests for adam-tree (spec sections 6 and 8, T16/T17):
//! two TreeStore instances over one SQLite file, mirroring the
//! deployment where the orchestrator writes and passes while channel
//! processes write leaves.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use adam_tree::{
    ContextRequest, FakeSummarizer, HashEmbedder, LeafWrite, NodeState, SummarizeScope, TreeStore,
};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

fn temp_db(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let path = std::env::temp_dir().join(format!(
        "adam-tree-conc-{tag}-{}-{n}.db",
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

fn sized(marker: &str, tokens: usize) -> String {
    let mut s = marker.to_string();
    while s.chars().count() < tokens * 4 {
        s.push('x');
    }
    s
}

const CONTACT: &str = "alex@example.com";
const THREAD: &str = "alex@example.com/contract-review";

/// The spec's retry discipline for writers: a surfaced busy error is
/// safe to retry because the (channel, external_id) key makes the
/// write idempotent.
fn write_with_retry(store: &TreeStore, input: &LeafWrite) {
    let mut attempts = 0;
    loop {
        match store.write_leaf(&HashEmbedder, input) {
            Ok(_) => return,
            Err(error) if attempts < 100 => {
                attempts += 1;
                eprintln!("write_leaf busy, retry {attempts}: {error}");
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(error) => panic!("write_leaf failed after retries: {error}"),
        }
    }
}

// T16: concurrent writers and passes keep exactly-once state.
#[test]
fn concurrent_writes_and_passes_keep_exactly_once() {
    let path = temp_db("exactly-once");
    let store = TreeStore::open(path.to_str().unwrap()).unwrap();
    let writer_path = path.clone();
    let writer = std::thread::spawn(move || {
        let store = TreeStore::open(writer_path.to_str().unwrap()).unwrap();
        for i in 0..20 {
            write_with_retry(
                &store,
                &LeafWrite::new(
                    "email",
                    CONTACT,
                    THREAD,
                    format!("ext-{i}"),
                    sized(&format!("leaf-{i}-"), 60),
                ),
            );
        }
        // A duplicate retry of an earlier message must stay a no-op.
        write_with_retry(
            &store,
            &LeafWrite::new("email", CONTACT, THREAD, "ext-7", sized("leaf-7-", 60)),
        );
    });

    while !writer.is_finished() {
        // Passes racing the writer may surface busy errors; the next
        // pass converges, which is the spec's failure model.
        let _ = store.summarize_now(
            &SummarizeScope::Thread(THREAD.to_string()),
            "orchestrator",
            &FakeSummarizer,
            &HashEmbedder,
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    writer.join().unwrap();

    // Adoption must not resurrect the duplicate as a second leaf.
    store.adopt_orphans().unwrap();
    let mut request = ContextRequest::for_thread(CONTACT, THREAD);
    request.token_budget = 1_000_000;
    let bundle = store.read_context(&HashEmbedder, &request).unwrap();
    assert_eq!(bundle.recent_leaves.len(), 20);
    store.assert_partition().unwrap();

    // Drain: passes after the writer finishes cover every leaf.
    for _ in 0..3 {
        store
            .summarize_now(
                &SummarizeScope::Thread(THREAD.to_string()),
                "orchestrator",
                &FakeSummarizer,
                &HashEmbedder,
            )
            .unwrap();
    }
    for record in &bundle.recent_leaves {
        assert_eq!(
            store.node(record.memory_id).unwrap().unwrap().state,
            NodeState::Covered,
            "leaf {} must be covered exactly once",
            record.memory_id
        );
    }
    store.assert_partition().unwrap();
    cleanup(&path);
}

// T17: reads from a second instance while the first writes and
// passes: reads never observe partial state.
#[test]
fn second_instance_reads_while_first_writes() {
    let path = temp_db("read-write");
    let reader = TreeStore::open(path.to_str().unwrap()).unwrap();
    let writer_path = path.clone();
    let writer = std::thread::spawn(move || {
        let store = TreeStore::open(writer_path.to_str().unwrap()).unwrap();
        for i in 0..12 {
            write_with_retry(
                &store,
                &LeafWrite::new(
                    "email",
                    CONTACT,
                    THREAD,
                    format!("rw-{i}"),
                    sized(&format!("leaf-{i}-"), 120),
                ),
            );
            let _ = store.summarize_now(
                &SummarizeScope::Thread(THREAD.to_string()),
                "orchestrator",
                &FakeSummarizer,
                &HashEmbedder,
            );
        }
    });

    let mut successful_reads = 0;
    while !writer.is_finished() {
        let mut request = ContextRequest::for_thread(CONTACT, THREAD);
        request.token_budget = 4000;
        request.query = Some("leaf".to_string());
        // A read that hits a busy window retries, per spec section 6;
        // a read that lands must be internally consistent.
        for _ in 0..50 {
            match reader.read_context(&HashEmbedder, &request) {
                Ok(bundle) => {
                    assert!(bundle.tokens_used <= 4000 || bundle.overflow);
                    successful_reads += 1;
                    break;
                }
                Err(_) => std::thread::sleep(Duration::from_millis(10)),
            }
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    writer.join().unwrap();
    assert!(successful_reads > 0, "reader completed no reads");

    let mut request = ContextRequest::for_thread(CONTACT, THREAD);
    request.token_budget = 1_000_000;
    let bundle = reader.read_context(&HashEmbedder, &request).unwrap();
    assert_eq!(bundle.recent_leaves.len(), 12);
    reader.assert_partition().unwrap();
    cleanup(&path);
}
