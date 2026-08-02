//! ADAM Memory System — biologically-inspired persistent memory.
//!
//! Four memory kinds share one SQLite-backed store: episodic (specific
//! experiences), semantic (generalized knowledge), procedural (skills),
//! and self-knowledge (identity). Every memory carries provenance, supports
//! vector similarity retrieval, participates in a typed relationship graph,
//! decays with disuse, consolidates from repeated episodes into semantic
//! knowledge, and resolves contradictions without ever deleting history.

mod ann;
mod conflict;
mod decay;
mod graph;
mod record;
mod retrieval;
mod store;

pub use ann::AnnIndex;
pub use record::{MemoryId, MemoryKind, MemoryRecord, MemoryRelation, Provenance, RelationKind};
pub use retrieval::cosine_similarity;
pub use store::{MemoryError, MemoryStore};

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};

    fn embedding(seed: f32) -> Vec<f32> {
        vec![seed, seed * 0.5, 1.0 - seed]
    }

    fn provenance(origin: &str) -> Provenance {
        Provenance {
            origin: origin.to_string(),
            evidence: vec!["test-evidence".to_string()],
        }
    }

    #[test]
    fn store_and_get_round_trips_a_memory() {
        let store = MemoryStore::open(":memory:").unwrap();
        let record = MemoryRecord::new(
            MemoryKind::Episodic,
            "Failed cargo build because dependency missing.",
            embedding(0.9),
            0.8,
            provenance("tool:cargo_build"),
            0.1,
        );
        store.store(&record).unwrap();

        let fetched = store.get(record.id).unwrap().unwrap();
        assert_eq!(fetched.content, record.content);
        assert_eq!(fetched.kind, MemoryKind::Episodic);
        assert_eq!(fetched.embedding, record.embedding);
    }

    #[test]
    fn access_bumps_count_and_refreshes_timestamp() {
        let store = MemoryStore::open(":memory:").unwrap();
        let record = MemoryRecord::new(
            MemoryKind::SelfKnowledge,
            "I am strong at systems engineering.",
            embedding(0.1),
            0.9,
            provenance("reflection:self_assessment"),
            0.0,
        );
        store.store(&record).unwrap();

        let accessed = store.access(record.id).unwrap();
        assert_eq!(accessed.access_count, 1);
        assert!(accessed.last_accessed_at >= record.last_accessed_at);
    }

    #[test]
    fn query_by_kind_filters_correctly() {
        let store = MemoryStore::open(":memory:").unwrap();
        let episodic = MemoryRecord::new(
            MemoryKind::Episodic,
            "Build failed.",
            embedding(0.9),
            0.7,
            provenance("tool:cargo_build"),
            0.1,
        );
        let semantic = MemoryRecord::new(
            MemoryKind::Semantic,
            "Rust projects require dependency validation.",
            embedding(0.2),
            0.6,
            provenance("reflection:consolidation"),
            0.0,
        );
        store.store(&episodic).unwrap();
        store.store(&semantic).unwrap();

        let episodics = store.query_by_kind(MemoryKind::Episodic).unwrap();
        assert_eq!(episodics.len(), 1);
        assert_eq!(episodics[0].id, episodic.id);
    }

    #[test]
    fn query_similar_ranks_by_cosine_similarity() {
        let store = MemoryStore::open(":memory:").unwrap();
        let close = MemoryRecord::new(
            MemoryKind::Procedural,
            "How to debug Rust compilation failures.",
            embedding(0.91),
            0.7,
            provenance("skill:rust_debugging"),
            0.05,
        );
        let far = MemoryRecord::new(
            MemoryKind::Procedural,
            "How to bake sourdough bread.",
            embedding(-0.9),
            0.7,
            provenance("skill:baking"),
            0.05,
        );
        store.store(&close).unwrap();
        store.store(&far).unwrap();

        let results = store
            .query_similar(&embedding(0.9), Some(MemoryKind::Procedural), 1)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.id, close.id);
        assert!(results[0].1 > 0.9);
    }

    #[test]
    fn relate_and_related_outgoing_traverse_the_graph() {
        let store = MemoryStore::open(":memory:").unwrap();
        let episode = MemoryRecord::new(
            MemoryKind::Episodic,
            "Build failed due to missing dependency.",
            embedding(0.9),
            0.7,
            provenance("tool:cargo_build"),
            0.1,
        );
        let generalization = MemoryRecord::new(
            MemoryKind::Semantic,
            "Rust projects require dependency validation.",
            embedding(0.85),
            0.75,
            provenance("reflection:consolidation"),
            0.0,
        );
        store.store(&episode).unwrap();
        store.store(&generalization).unwrap();
        store
            .relate(episode.id, generalization.id, RelationKind::Generalizes)
            .unwrap();

        let outgoing = store.related_outgoing(episode.id).unwrap();
        assert_eq!(outgoing.len(), 1);
        assert_eq!(outgoing[0].0.id, generalization.id);
        assert_eq!(outgoing[0].1, RelationKind::Generalizes);

        let incoming = store.related_incoming(generalization.id).unwrap();
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0].0.id, episode.id);
    }

    #[test]
    fn decay_pass_reduces_confidence_of_stale_memories() {
        let store = MemoryStore::open(":memory:").unwrap();
        let mut record = MemoryRecord::new(
            MemoryKind::Episodic,
            "One-off event nobody revisited.",
            embedding(0.5),
            1.0,
            provenance("tool:test"),
            1.0,
        );
        record.last_accessed_at = Utc::now() - Duration::days(10);
        store.store(&record).unwrap();

        let updated = store.decay_pass(Utc::now()).unwrap();
        assert_eq!(updated, 1);

        let after = store.get(record.id).unwrap().unwrap();
        assert!(after.confidence < 1.0);
    }

    #[test]
    fn self_knowledge_with_zero_decay_rate_never_decays() {
        let store = MemoryStore::open(":memory:").unwrap();
        let mut record = MemoryRecord::new(
            MemoryKind::SelfKnowledge,
            "I am strong at systems engineering.",
            embedding(0.5),
            0.95,
            provenance("reflection:self_assessment"),
            0.0,
        );
        record.last_accessed_at = Utc::now() - Duration::days(365);
        store.store(&record).unwrap();

        let updated = store.decay_pass(Utc::now()).unwrap();
        assert_eq!(updated, 0);
        let after = store.get(record.id).unwrap().unwrap();
        assert_eq!(after.confidence, 0.95);
    }

    #[test]
    fn consolidate_generalizes_similar_episodes_into_semantic_memory() {
        let store = MemoryStore::open(":memory:").unwrap();
        for i in 0..3 {
            let record = MemoryRecord::new(
                MemoryKind::Episodic,
                format!("Cargo build failed attempt {i}."),
                embedding(0.9),
                0.6,
                provenance("tool:cargo_build"),
                0.1,
            );
            store.store(&record).unwrap();
        }

        let created = store.consolidate(MemoryKind::Episodic, 2, 0.99).unwrap();
        assert_eq!(created.len(), 1);

        let semantic = store.get(created[0]).unwrap().unwrap();
        assert_eq!(semantic.kind, MemoryKind::Semantic);
        assert_eq!(semantic.provenance.evidence.len(), 3);

        let sources = store.related_incoming(created[0]).unwrap();
        assert_eq!(sources.len(), 3);
        for (_, relation) in sources {
            assert_eq!(relation, RelationKind::Generalizes);
        }
    }

    #[test]
    fn resolve_conflict_keeps_higher_confidence_and_marks_loser_superseded() {
        let store = MemoryStore::open(":memory:").unwrap();
        let strong = MemoryRecord::new(
            MemoryKind::Semantic,
            "The API requires an auth header.",
            embedding(0.5),
            0.9,
            provenance("tool:integration_test"),
            0.05,
        );
        let weak = MemoryRecord::new(
            MemoryKind::Semantic,
            "The API does not require auth.",
            embedding(-0.5),
            0.3,
            provenance("tool:stale_docs"),
            0.05,
        );
        store.store(&strong).unwrap();
        store.store(&weak).unwrap();

        let winner = store.resolve_conflict(strong.id, weak.id).unwrap();
        assert_eq!(winner, strong.id);

        let loser_after = store.get(weak.id).unwrap().unwrap();
        assert_eq!(loser_after.superseded_by, Some(strong.id));

        let outgoing = store.related_outgoing(weak.id).unwrap();
        assert_eq!(outgoing[0].0.id, strong.id);
        assert_eq!(outgoing[0].1, RelationKind::SupersededBy);
    }
}
