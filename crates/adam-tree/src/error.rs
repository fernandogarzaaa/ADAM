//! Errors raised by the tree-of-summaries layer.

use adam_memory::{MemoryError, MemoryId};
use thiserror::Error;

/// Errors raised by [`crate::TreeStore`] operations.
#[derive(Debug, Error)]
pub enum TreeError {
    /// An error from the underlying ADAM memory store.
    #[error(transparent)]
    Memory(#[from] MemoryError),
    /// An error from the layer's own SQLite sidecar connection.
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    /// Stored sidecar data could not be parsed, or a sidecar row
    /// references an ADAM record that does not exist.
    #[error("corrupt tree data: {0}")]
    Corrupt(String),
    /// The memory has no `tree_nodes` row.
    #[error("memory {0} is not part of the tree")]
    NotInTree(MemoryId),
    /// A summary stayed over its token target after one retry
    /// (spec section 5). The covered children stay open.
    #[error("summary of {tokens} tokens exceeds target of {target} tokens")]
    SummaryTooLarge {
        /// Token estimate of the rejected summary.
        tokens: usize,
        /// The target it had to fit.
        target: usize,
    },
    /// A produced summary contained credential-shaped content and was
    /// rejected outright (spec section 5).
    #[error("summary contains credential-shaped content and was rejected")]
    CredentialContent,
    /// A node ended up with the wrong number of tree parents; the
    /// coverage partition is the one corruption this design forbids,
    /// so it fails loudly instead of being silently repaired.
    #[error("partition violation: memory {0} has {1} tree parents")]
    PartitionViolation(MemoryId, usize),
    /// A summarizer or embedder backend failed.
    #[error("backend error: {0}")]
    Backend(String),
    /// Caller input failed validation.
    #[error("invalid input: {0}")]
    InvalidInput(String),
}
