//! Tree-of-summaries memory layer on ADAM.
//!
//! ADAM stores records; this layer adds the conversational memory
//! discipline the Cookie agent runs on: verbatim episodic leaves per
//! exchange, abstractive summary nodes in levels (L1 thread chunk, L2
//! thread, L3 contact) linked by ADAM's own `Generalizes` relation, a
//! strict coverage partition, a three-band token-budgeted read path,
//! and a lease-gated summarizer that only the orchestrator runs.
//!
//! The layer adds no new `MemoryKind` or `RelationKind`. Everything
//! ADAM cannot express lives in one sidecar table (`tree_nodes`) in
//! the same SQLite file. The crate is model-agnostic: embeddings and
//! summary text come from injected [`Embedder`] / [`Summarizer`]
//! backends supplied by the caller.
//!
//! Design: `memory-layer-SPEC.md` (cookie-identity program). Section
//! references in the docs below point at that spec.

mod config;
mod detect;
mod error;
mod model;
mod read;
mod store;
mod summarize;
mod testing;
mod write;

pub use adam_memory::{MemoryId, MemoryKind, MemoryRecord, Provenance, RelationKind};
pub use config::TreeConfig;
pub use detect::contains_credential_shaped;
pub use error::TreeError;
pub use model::{
    token_estimate, AdoptionReport, ContextBundle, ContextRecord, ContextRequest, DrillResult,
    Embedder, LeafOutcome, LeafWrite, Lease, NodeState, SearchHit, SummarizeScope, Summarizer,
    SummaryChild, SummaryReport, SummaryRequest, TreeNode,
};
pub use store::TreeStore;
pub use testing::{ConstEmbedder, FakeSummarizer, HashEmbedder};
