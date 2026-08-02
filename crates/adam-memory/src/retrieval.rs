//! Cosine-similarity vector retrieval over stored memory embeddings.

use crate::ann::AnnIndex;
use crate::record::{MemoryKind, MemoryRecord};
use crate::store::{MemoryError, MemoryStore};

impl MemoryStore {
    /// Return the `top_k` memories most similar to `query_embedding` by
    /// cosine similarity, optionally restricted to a single [`MemoryKind`].
    ///
    /// This scores every candidate in-process rather than using an ANN
    /// index — correct and simple for the organism-scale memory volumes
    /// ADAM targets, at the cost of O(n) per query. See
    /// [`MemoryStore::build_ann_index`] for an approximate alternative
    /// that scales better past organism scale.
    pub fn query_similar(
        &self,
        query_embedding: &[f32],
        kind: Option<MemoryKind>,
        top_k: usize,
    ) -> Result<Vec<(MemoryRecord, f32)>, MemoryError> {
        let candidates = match kind {
            Some(k) => self.query_by_kind(k)?,
            None => self.all()?,
        };

        let mut scored: Vec<(MemoryRecord, f32)> = candidates
            .into_iter()
            .map(|record| {
                let score = cosine_similarity(query_embedding, &record.embedding);
                (record, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        Ok(scored)
    }

    /// Build an [`AnnIndex`] snapshot over every stored memory, or only
    /// those matching `kind` when given. Intended for callers with memory
    /// volumes large enough that [`MemoryStore::query_similar`]'s O(n)
    /// scan becomes a bottleneck — build once (e.g. on startup or after a
    /// batch of writes) and reuse the index across many queries rather
    /// than rebuilding per query.
    pub fn build_ann_index(&self, kind: Option<MemoryKind>) -> Result<AnnIndex, MemoryError> {
        let records = match kind {
            Some(k) => self.query_by_kind(k)?,
            None => self.all()?,
        };
        Ok(AnnIndex::build(&records))
    }
}

/// Cosine similarity between two vectors. Returns `0.0` for mismatched
/// lengths or zero-magnitude vectors rather than panicking, since
/// embeddings from different models or empty placeholders should be
/// scored as "unrelated" rather than crash retrieval.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot / (mag_a * mag_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_have_similarity_one() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn orthogonal_vectors_have_similarity_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn mismatched_lengths_return_zero_instead_of_panicking() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }
}
