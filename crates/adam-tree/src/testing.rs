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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn const_embedder_returns_its_vector_verbatim() {
        let embedder = ConstEmbedder(vec![0.25, -0.5, 0.75]);
        assert_eq!(
            embedder.embed("anything at all").unwrap(),
            vec![0.25, -0.5, 0.75]
        );
        assert_eq!(embedder.embed("").unwrap(), vec![0.25, -0.5, 0.75]);
    }

    /// The hash embedder's contract is exact determinism: same
    /// text, same vector, byte for byte. These vectors are the
    /// FNV-1a + xorshift walk computed independently of the Rust
    /// code; any change to the hash arithmetic breaks them.
    #[test]
    fn hash_embedder_matches_reference_vectors() {
        assert_eq!(
            HashEmbedder.embed("").unwrap(),
            vec![
                -0.05299997329711914,
                0.6790000200271606,
                0.7300000190734863,
                0.003000020980834961,
                0.9570000171661377,
                -0.12800002098083496,
                -0.14499998092651367,
                0.18299996852874756,
            ]
        );
        assert_eq!(
            HashEmbedder.embed("hello").unwrap(),
            vec![
                0.07299995422363281,
                -0.5480000078678131,
                0.878000020980835,
                0.5570000410079956,
                0.2059999704360962,
                -0.812000036239624,
                0.13499999046325684,
                -0.880000002682209,
            ]
        );
        assert_eq!(
            HashEmbedder.embed("cookie").unwrap(),
            vec![
                -0.20099997520446777,
                -0.5149999856948853,
                -0.8110000044107437,
                0.7990000247955322,
                -0.6089999973773956,
                0.6059999465942383,
                0.2799999713897705,
                -0.1769999861717224,
            ]
        );
    }
}
