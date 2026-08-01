//! Confidence decay over time and episodic -> semantic consolidation.

use chrono::{DateTime, Utc};

use crate::record::{MemoryId, MemoryKind, MemoryRecord, Provenance, RelationKind};
use crate::retrieval::cosine_similarity;
use crate::store::{MemoryError, MemoryStore};

impl MemoryStore {
    /// Apply exponential confidence decay to every memory with a nonzero
    /// `decay_rate` that hasn't been superseded, based on days elapsed
    /// since `last_accessed_at`. Frequently-accessed memories resist
    /// decay because each [`MemoryStore::access`] call resets the clock.
    /// Returns the number of memories whose confidence changed.
    pub fn decay_pass(&self, now: DateTime<Utc>) -> Result<usize, MemoryError> {
        let mut updated = 0usize;
        for record in self.all()? {
            if record.decay_rate <= 0.0 || record.superseded_by.is_some() {
                continue;
            }
            let elapsed_days = (now - record.last_accessed_at).num_seconds() as f32 / 86_400.0;
            if elapsed_days <= 0.0 {
                continue;
            }
            let decayed = record.confidence * (-record.decay_rate * elapsed_days).exp();
            if (decayed - record.confidence).abs() > f32::EPSILON {
                self.update_confidence(record.id, decayed)?;
                updated += 1;
            }
        }
        Ok(updated)
    }

    /// Cluster memories of `kind` (typically [`MemoryKind::Episodic`]) by
    /// embedding similarity and, for every cluster of at least
    /// `min_group_size` members whose pairwise similarity to the cluster's
    /// seed meets `similarity_threshold`, commit one new
    /// [`MemoryKind::Semantic`] memory generalizing them. Every source
    /// memory is linked to the new semantic memory via
    /// [`RelationKind::Generalizes`]. Returns the ids of newly created
    /// semantic memories.
    pub fn consolidate(
        &self,
        kind: MemoryKind,
        min_group_size: usize,
        similarity_threshold: f32,
    ) -> Result<Vec<MemoryId>, MemoryError> {
        let mut pool = self.query_by_kind(kind)?;
        let mut created = Vec::new();

        while let Some(seed) = pool.pop() {
            let mut cluster = vec![seed.clone()];
            pool.retain(|candidate| {
                let similarity = cosine_similarity(&seed.embedding, &candidate.embedding);
                if similarity >= similarity_threshold {
                    cluster.push(candidate.clone());
                    false
                } else {
                    true
                }
            });

            if cluster.len() < min_group_size {
                continue;
            }

            let new_id = self.commit_generalization(&cluster)?;
            created.push(new_id);
        }

        Ok(created)
    }

    fn commit_generalization(&self, cluster: &[MemoryRecord]) -> Result<MemoryId, MemoryError> {
        let dims = cluster[0].embedding.len();
        let mut averaged_embedding = vec![0.0f32; dims];
        let mut confidence_sum = 0.0f32;
        let mut contents = Vec::new();
        let mut evidence = Vec::new();

        for record in cluster {
            for (i, value) in record.embedding.iter().enumerate() {
                if i < dims {
                    averaged_embedding[i] += value;
                }
            }
            confidence_sum += record.confidence;
            contents.push(record.content.clone());
            evidence.push(record.id.to_string());
        }

        let n = cluster.len() as f32;
        for value in averaged_embedding.iter_mut() {
            *value /= n;
        }

        let generalized = MemoryRecord::new(
            MemoryKind::Semantic,
            format!(
                "Generalized from {} experiences: {}",
                cluster.len(),
                contents.join("; ")
            ),
            averaged_embedding,
            confidence_sum / n,
            Provenance {
                origin: "reflection:consolidation".to_string(),
                evidence,
            },
            0.0,
        );
        let new_id = generalized.id;
        self.store(&generalized)?;

        for record in cluster {
            self.relate(record.id, new_id, RelationKind::Generalizes)?;
        }

        Ok(new_id)
    }
}
