//! Core memory record types shared across the ADAM memory system.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier for a memory record.
pub type MemoryId = Uuid;

/// The four biologically-inspired memory types ADAM tracks.
///
/// - `Episodic`: a specific experience ("failed cargo build because
///   dependency missing").
/// - `Semantic`: generalized knowledge distilled from experience ("Rust
///   projects require dependency validation").
/// - `Procedural`: a skill or how-to ("how to debug Rust compilation
///   failures").
/// - `SelfKnowledge`: identity-level self-assessment ("I am strong at
///   systems engineering").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    Episodic,
    Semantic,
    Procedural,
    SelfKnowledge,
}

impl MemoryKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryKind::Episodic => "episodic",
            MemoryKind::Semantic => "semantic",
            MemoryKind::Procedural => "procedural",
            MemoryKind::SelfKnowledge => "self_knowledge",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "episodic" => Some(MemoryKind::Episodic),
            "semantic" => Some(MemoryKind::Semantic),
            "procedural" => Some(MemoryKind::Procedural),
            "self_knowledge" => Some(MemoryKind::SelfKnowledge),
            _ => None,
        }
    }
}

/// Origin and justification for a memory's existence — every memory must be
/// traceable to where it came from and why it should be trusted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    /// Where the memory came from, e.g. `"tool:cargo_build"`,
    /// `"reflection:consolidation"`, `"user:conversation"`.
    pub origin: String,
    /// Supporting evidence strings (log excerpts, references to other
    /// memory ids, quoted user statements, etc).
    pub evidence: Vec<String>,
}

/// A single stored memory: content plus provenance, confidence, an
/// embedding for similarity retrieval, and access/decay bookkeeping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: MemoryId,
    pub kind: MemoryKind,
    pub content: String,
    pub embedding: Vec<f32>,
    pub confidence: f32,
    pub provenance: Provenance,
    pub created_at: DateTime<Utc>,
    pub last_accessed_at: DateTime<Utc>,
    pub access_count: u32,
    /// Exponential decay rate applied per day of inactivity. `0.0` means
    /// the memory never decays (appropriate for `SelfKnowledge`).
    pub decay_rate: f32,
    /// Set when this memory has been superseded by a higher-confidence
    /// or corrected memory, rather than being deleted.
    pub superseded_by: Option<MemoryId>,
}

impl MemoryRecord {
    /// Construct a new memory record. `confidence` is clamped to `[0, 1]`.
    pub fn new(
        kind: MemoryKind,
        content: impl Into<String>,
        embedding: Vec<f32>,
        confidence: f32,
        provenance: Provenance,
        decay_rate: f32,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            kind,
            content: content.into(),
            embedding,
            confidence: confidence.clamp(0.0, 1.0),
            provenance,
            created_at: now,
            last_accessed_at: now,
            access_count: 0,
            decay_rate,
            superseded_by: None,
        }
    }
}

/// A typed edge between two memories in the memory graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationKind {
    /// `to` generalizes `from` (episodic -> semantic consolidation).
    Generalizes,
    /// `to` supports/reinforces `from`.
    Supports,
    /// `to` contradicts `from`.
    Contradicts,
    /// `to` supersedes `from` after conflict resolution.
    SupersededBy,
    /// `to` is a procedural dependency of `from`.
    DependsOn,
}

impl RelationKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            RelationKind::Generalizes => "generalizes",
            RelationKind::Supports => "supports",
            RelationKind::Contradicts => "contradicts",
            RelationKind::SupersededBy => "superseded_by",
            RelationKind::DependsOn => "depends_on",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "generalizes" => Some(RelationKind::Generalizes),
            "supports" => Some(RelationKind::Supports),
            "contradicts" => Some(RelationKind::Contradicts),
            "superseded_by" => Some(RelationKind::SupersededBy),
            "depends_on" => Some(RelationKind::DependsOn),
            _ => None,
        }
    }
}

/// A stored graph edge between two memory records.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryRelation {
    pub id: Uuid,
    pub from: MemoryId,
    pub to: MemoryId,
    pub kind: RelationKind,
    pub created_at: DateTime<Utc>,
}
