//! Lightweight deterministic text embedding.
//!
//! A real production organism would delegate to an embedding model; this
//! crate has no ML dependency, so it uses a deterministic bag-of-hashed-
//! tokens vector instead. It is intentionally simple (same tradeoff
//! `adam-memory` documents for its O(n) retrieval): good enough to make
//! cosine similarity meaningfully cluster related text at organism scale,
//! without pulling in a model runtime. Swapping in a real embedder later
//! only requires changing this one function.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const DIMENSIONS: usize = 64;

pub fn embed(text: &str) -> Vec<f32> {
    let mut vector = vec![0f32; DIMENSIONS];
    for token in text.to_lowercase().split_whitespace() {
        let mut hasher = DefaultHasher::new();
        token.hash(&mut hasher);
        let bucket = (hasher.finish() as usize) % DIMENSIONS;
        vector[bucket] += 1.0;
    }
    vector
}

#[cfg(test)]
mod tests {
    use super::*;
    use adam_memory::cosine_similarity;

    #[test]
    fn identical_text_embeds_identically() {
        assert_eq!(embed("the build failed"), embed("the build failed"));
    }

    #[test]
    fn shared_vocabulary_scores_higher_than_disjoint_vocabulary() {
        let a = embed("rust cargo build failure");
        let b = embed("rust cargo build succeeded");
        let c = embed("banana smoothie recipe");

        let sim_related = cosine_similarity(&a, &b);
        let sim_unrelated = cosine_similarity(&a, &c);
        assert!(sim_related > sim_unrelated);
    }
}
