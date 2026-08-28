//! Text embedding, backed by a local ONNX sentence-transformer model
//! (`all-MiniLM-L6-v2`, run through `fastembed`/`ort`) rather than the
//! deterministic hashed bag-of-tokens vector this crate used previously.
//!
//! `adam-memory` already treats embeddings as opaque `Vec<f32>` (see
//! `cosine_similarity`'s doc comment on mismatched lengths), so this is the
//! only file that needed to change per DESIGN.md's "Lightweight
//! deterministic embeddings instead of an ML model" note. The model is
//! fetched from Hugging Face on first use and cached under
//! `ADAM_EMBEDDING_CACHE_DIR` (or `fastembed`'s own default cache dir), then
//! held for the rest of the process behind a lazily-initialized,
//! mutex-guarded instance — `TextEmbedding::embed` takes `&mut self`, and
//! every `Organism` method that calls `embed` takes `&self`.
//!
//! Loading (and any concurrent `embed` call) is serialized behind one
//! process-wide mutex rather than a `OnceLock`, for two reasons: a
//! `OnceLock::get_or_init` would cache a failed load (e.g. a network hiccup
//! fetching the model on first use) forever, permanently breaking every
//! later call in a long-running `adam-mcp` process instead of retrying; and
//! it would let every thread racing to initialize at once independently
//! download and load the model in parallel, which is exactly the kind of
//! concurrent first-use load this module cannot assume won't happen (many
//! `Organism`s under one `OrganismPool`, many test threads in one binary).
//! Holding the mutex across the whole "load if absent, then use" sequence
//! means only one attempt — and only one download — ever happens at a time,
//! and a failed attempt simply leaves the slot empty for the next caller to
//! retry.

use std::sync::Mutex;

use fastembed::{EmbeddingModel, InitOptionsWithLength, TextEmbedding};
use thiserror::Error;

/// `all-MiniLM-L6-v2`'s output width. Any query embedding is compared only
/// against embeddings this same function produced, so this constant is not
/// load-bearing outside this module — but callers with no records yet still
/// need a defined width for a zero vector when inference fails.
const DIMENSIONS: usize = 384;

#[derive(Debug, Error)]
pub enum EmbeddingError {
    #[error("failed to load embedding model: {0}")]
    ModelLoad(String),
    #[error("failed to embed text: {0}")]
    Inference(String),
}

/// Embed `text` with the real model. Model load or inference failure is
/// propagated as [`EmbeddingError`] (via `?` into
/// [`crate::organism::OrganismError`]) rather than silently substituting a
/// placeholder embedding for stored content — the only fallback is an
/// all-zero vector for the degenerate case of `embed` returning no vector
/// for one input text, which `cosine_similarity` scores as unrelated to
/// everything, including itself.
pub fn embed(text: &str) -> Result<Vec<f32>, EmbeddingError> {
    static MODEL: Mutex<Option<TextEmbedding>> = Mutex::new(None);
    let mut guard = MODEL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    if guard.is_none() {
        let mut options = InitOptionsWithLength::new(EmbeddingModel::AllMiniLML6V2Q)
            .with_show_download_progress(false);
        if let Ok(dir) = std::env::var("ADAM_EMBEDDING_CACHE_DIR") {
            options = options.with_cache_dir(dir.into());
        }
        let instance = TextEmbedding::try_new(options)
            .map_err(|err| EmbeddingError::ModelLoad(err.to_string()))?;
        *guard = Some(instance);
    }

    let model = guard
        .as_mut()
        .expect("just initialized above if it was empty");
    let mut vectors = model
        .embed(vec![text], None)
        .map_err(|err| EmbeddingError::Inference(err.to_string()))?;
    Ok(vectors.pop().unwrap_or_else(|| vec![0.0; DIMENSIONS]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use adam_memory::cosine_similarity;

    // These tests exercise the real model and therefore need network access
    // on first run (to fetch and cache the ONNX weights) or a warm
    // `ADAM_EMBEDDING_CACHE_DIR`. They are marked `#[ignore]` so `cargo test
    // --workspace` stays hermetic by default; run them explicitly with
    // `cargo test -p adam-organism -- --ignored` when the model is
    // reachable.

    #[test]
    #[ignore]
    fn identical_text_embeds_identically() {
        assert_eq!(
            embed("the build failed").unwrap(),
            embed("the build failed").unwrap()
        );
    }

    #[test]
    #[ignore]
    fn shared_vocabulary_scores_higher_than_disjoint_vocabulary() {
        let a = embed("rust cargo build failure").unwrap();
        let b = embed("rust cargo build succeeded").unwrap();
        let c = embed("banana smoothie recipe").unwrap();

        let sim_related = cosine_similarity(&a, &b);
        let sim_unrelated = cosine_similarity(&a, &c);
        assert!(sim_related > sim_unrelated);
    }
}
