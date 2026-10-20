//! Deterministic backends for tests and demos. Not for production
//! use: the embedder is a hash projection, not a semantic model, and
//! the summarizer truncates instead of abstracting — but both honour
//! the contracts the layer enforces, which is what tests need.

use crate::error::TreeError;
use crate::model::{token_estimate, Embedder, Summarizer, SummaryRequest};

/// An embedder that returns the same vector for every input, making
/// every similarity score equal. Useful for ranking tests.
#[derive(Debug, Clone)]
pub struct ConstEmbedder(pub Vec<f32>);

impl Embedder for ConstEmbedder {
    fn embed(&self, _text: &str) -> Result<Vec<f32>, TreeError> {
        Ok(self.0.clone())
    }
}

/// A deterministic stand-in embedder: an FNV-1a hash of the text,
/// expanded into an 8-dimension vector by a xorshift walk. Same
/// text, same vector; no semantic claims.
#[derive(Debug, Default)]
pub struct HashEmbedder;

impl Embedder for HashEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, TreeError> {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in text.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        let mut dims = Vec::with_capacity(8);
        let mut state = hash;
        for _ in 0..8 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            dims.push(((state % 2000) as f32 / 1000.0) - 1.0);
        }
        Ok(dims)
    }
}

/// A deterministic summarizer that honours the target: it emits the
/// spec's form header (dated span, channels, message count), then
/// the children's contents in span order, truncated on a char
/// boundary so the result's [`token_estimate`] fits
/// `target_max_tokens`. It never invents content.
#[derive(Debug, Default)]
pub struct FakeSummarizer;

impl Summarizer for FakeSummarizer {
    fn summarize(&self, request: &SummaryRequest) -> Result<String, TreeError> {
        let mut channels: Vec<&str> = request
            .children
            .iter()
            .filter_map(|c| c.channel.as_deref())
            .collect();
        channels.sort_unstable();
        channels.dedup();
        let channel_part = if channels.is_empty() {
            "multi-channel".to_string()
        } else {
            channels.join(" + ")
        };
        let mut text = format!(
            "{} to {}, {}, {} messages\n",
            request.span_start.format("%Y-%m-%d"),
            request.span_end.format("%Y-%m-%d"),
            channel_part,
            request.children.len(),
        );
        for child in &request.children {
            text.push_str(&child.content);
            text.push('\n');
        }
        // token_estimate(text) <= target  <=>  chars <= target * 4.
        let max_chars = request.target_max_tokens.saturating_mul(4);
        if text.chars().count() > max_chars {
            text = text.chars().take(max_chars).collect();
        }
        debug_assert!(token_estimate(&text) <= request.target_max_tokens);
        Ok(text)
    }
}
