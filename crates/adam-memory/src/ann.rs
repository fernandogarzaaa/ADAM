//! Approximate nearest-neighbor retrieval via an HNSW index, as a
//! complement to [`MemoryStore::query_similar`]'s exact O(n) cosine scan.
//!
//! The exact scan is correct and simple at organism-scale memory volumes
//! (see DESIGN.md), but degrades linearly as record counts grow into the
//! millions. `AnnIndex` is a snapshot built explicitly by the caller from
//! a set of records — it does not update incrementally — so callers
//! choose when to pay the (one-time, O(n log n)) build cost, typically on
//! startup or after a batch of writes, and reuse the built index across
//! many queries.

use instant_distance::{Builder, HnswMap, Point, Search};

use crate::record::{MemoryId, MemoryRecord};
use crate::retrieval::cosine_similarity;

#[derive(Clone)]
struct EmbeddingPoint(Vec<f32>);

impl Point for EmbeddingPoint {
    fn distance(&self, other: &Self) -> f32 {
        // HNSW searches for minimal distance; cosine *similarity* is
        // maximized for closeness, so invert it into a distance measure.
        1.0 - cosine_similarity(&self.0, &other.0)
    }
}

/// A queryable snapshot of memory embeddings, indexed for approximate
/// nearest-neighbor search.
pub struct AnnIndex {
    map: HnswMap<EmbeddingPoint, MemoryId>,
}

impl AnnIndex {
    /// Build an index over `records`. `Builder::default()` handles the
    /// empty-records case safely (queries just return nothing).
    pub fn build(records: &[MemoryRecord]) -> Self {
        let points: Vec<EmbeddingPoint> = records
            .iter()
            .map(|r| EmbeddingPoint(r.embedding.clone()))
            .collect();
        let ids: Vec<MemoryId> = records.iter().map(|r| r.id).collect();
        let map = Builder::default().build(points, ids);
        Self { map }
    }

    /// The `top_k` memory ids most similar to `query_embedding`, with
    /// their cosine similarity score, ranked nearest-first.
    pub fn query(&self, query_embedding: &[f32], top_k: usize) -> Vec<(MemoryId, f32)> {
        let mut search = Search::default();
        let point = EmbeddingPoint(query_embedding.to_vec());
        self.map
            .search(&point, &mut search)
            .take(top_k)
            .map(|item| (*item.value, 1.0 - item.distance))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{MemoryKind, Provenance};

    fn record(embedding: Vec<f32>) -> MemoryRecord {
        MemoryRecord::new(
            MemoryKind::Semantic,
            "test content",
            embedding,
            0.9,
            Provenance {
                origin: "test".to_string(),
                evidence: vec![],
            },
            0.0,
        )
    }

    #[test]
    fn finds_the_closest_point_by_cosine_similarity() {
        let close = record(vec![0.9, 0.1, 0.0]);
        let far = record(vec![-0.9, -0.1, 0.0]);
        let index = AnnIndex::build(&[close.clone(), far]);

        let results = index.query(&[0.9, 0.1, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, close.id);
        assert!(results[0].1 > 0.9);
    }

    #[test]
    fn an_empty_index_returns_no_results() {
        let index = AnnIndex::build(&[]);
        assert!(index.query(&[1.0, 0.0], 5).is_empty());
    }

    #[test]
    fn top_k_limits_the_number_of_results() {
        let records: Vec<MemoryRecord> =
            (0..10).map(|i| record(vec![i as f32, 1.0, 0.0])).collect();
        let index = AnnIndex::build(&records);

        let results = index.query(&[5.0, 1.0, 0.0], 3);
        assert_eq!(results.len(), 3);
    }
}
