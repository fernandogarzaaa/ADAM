//! Genome system: the versioned identity/values/goals/beliefs/capabilities/
//! skills/preferences/policies substrate for an ADAM organism.
//!
//! History is append-only and hash-linked (`GenomeVersion::content_hash`
//! commits to the exact genome payload). Rollback never rewrites history —
//! it commits a *new* version whose content equals a prior version's, so the
//! version graph always grows forward and nothing is ever deleted.

use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use uuid::Uuid;

/// Unique identifier for a single point in the genome's version history.
pub type VersionId = Uuid;

/// Core organism identity: name and a free-form self-description.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Identity {
    pub name: String,
    pub description: String,
}

/// The organism genome: the full evolvable state of an ADAM instance.
///
/// This is intentionally data-only (no behavior) — it is the payload that
/// gets versioned, diffed, hashed, and rolled back by [`GenomeHistory`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Genome {
    pub identity: Identity,
    pub values: Vec<String>,
    pub goals: Vec<String>,
    pub beliefs: Vec<String>,
    pub capabilities: Vec<String>,
    pub skills: Vec<String>,
    pub preferences: HashMap<String, String>,
    pub policies: Vec<String>,
}

impl Genome {
    /// Construct a minimal genome with only an identity set. All other
    /// fields start empty and are populated through subsequent commits.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            identity: Identity {
                name: name.into(),
                description: description.into(),
            },
            values: Vec::new(),
            goals: Vec::new(),
            beliefs: Vec::new(),
            capabilities: Vec::new(),
            skills: Vec::new(),
            preferences: HashMap::new(),
            policies: Vec::new(),
        }
    }

    /// Deterministic SHA-256 hash of the canonical JSON encoding of this
    /// genome. Two genomes with identical content always hash identically
    /// regardless of insertion order, because `preferences` (a `HashMap`)
    /// is serialized through `serde_json::to_value` and re-sorted by key
    /// before hashing.
    pub fn content_hash(&self) -> String {
        let value = serde_json::to_value(self).expect("Genome always serializes");
        let canonical = canonicalize(&value);
        let mut hasher = Sha256::new();
        hasher.update(canonical.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

/// Recursively renders a `serde_json::Value` with object keys sorted, so the
/// resulting string is stable regardless of `HashMap` iteration order.
fn canonicalize(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let entries: Vec<String> = keys
                .into_iter()
                .map(|k| format!("{:?}:{}", k, canonicalize(&map[k])))
                .collect();
            format!("{{{}}}", entries.join(","))
        }
        serde_json::Value::Array(items) => {
            let entries: Vec<String> = items.iter().map(canonicalize).collect();
            format!("[{}]", entries.join(","))
        }
        other => other.to_string(),
    }
}

/// A single immutable snapshot in the genome's version history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenomeVersion {
    pub id: VersionId,
    pub parent: Option<VersionId>,
    pub label: String,
    pub genome: Genome,
    pub reason: String,
    pub created_at: DateTime<Utc>,
    pub content_hash: String,
}

/// Errors raised while operating on a [`GenomeHistory`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum GenomeError {
    #[error("version {0} not found in history")]
    VersionNotFound(VersionId),
}

/// Append-only, hash-linked version graph for a genome's evolution.
///
/// Internally this is a linear chain (each version has exactly one parent,
/// except the root) rather than a branching DAG, which matches ADAM's
/// single-organism evolution model: one lineage per organism instance,
/// extended by `commit` and folded back on by `rollback`.
#[derive(Debug, Clone)]
pub struct GenomeHistory {
    versions: Vec<GenomeVersion>,
    index: HashMap<VersionId, usize>,
    head: VersionId,
}

impl GenomeHistory {
    /// Start a new history with `genome` as version `1.0` (the root, with
    /// no parent).
    pub fn init(genome: Genome, reason: impl Into<String>) -> Self {
        let id = Uuid::new_v4();
        let version = GenomeVersion {
            id,
            parent: None,
            label: "1.0".to_string(),
            content_hash: genome.content_hash(),
            genome,
            reason: reason.into(),
            created_at: Utc::now(),
        };
        let mut index = HashMap::new();
        index.insert(id, 0usize);
        Self {
            versions: vec![version],
            index,
            head: id,
        }
    }

    /// Append a new version on top of the current head. The version label
    /// bumps the minor component of the head's label (`1.0` -> `1.1`).
    /// Returns the new version's id, which becomes the new head.
    pub fn commit(&mut self, genome: Genome, reason: impl Into<String>) -> VersionId {
        let parent_label = self.head().label.clone();
        let label = bump_minor(&parent_label);
        let id = Uuid::new_v4();
        let version = GenomeVersion {
            id,
            parent: Some(self.head),
            label,
            content_hash: genome.content_hash(),
            genome,
            reason: reason.into(),
            created_at: Utc::now(),
        };
        self.versions.push(version);
        self.index.insert(id, self.versions.len() - 1);
        self.head = id;
        id
    }

    /// The current head version.
    pub fn head(&self) -> &GenomeVersion {
        self.get(self.head)
            .expect("head always points at an existing version")
    }

    /// The id of the current head version.
    pub fn head_id(&self) -> VersionId {
        self.head
    }

    /// Look up a version by id.
    pub fn get(&self, id: VersionId) -> Result<&GenomeVersion, GenomeError> {
        self.index
            .get(&id)
            .map(|&i| &self.versions[i])
            .ok_or(GenomeError::VersionNotFound(id))
    }

    /// Full linear history from root to head, oldest first.
    pub fn all(&self) -> &[GenomeVersion] {
        &self.versions
    }

    /// Walk the parent chain of `id` back to the root, returning ids
    /// ordered from `id` itself back to the root (inclusive of `id`).
    pub fn ancestors(&self, id: VersionId) -> Result<Vec<VersionId>, GenomeError> {
        let mut current = self.get(id)?;
        let mut chain = vec![current.id];
        while let Some(parent_id) = current.parent {
            current = self.get(parent_id)?;
            chain.push(current.id);
        }
        Ok(chain)
    }

    /// Roll back to a prior version's *content* without deleting history.
    /// This commits a brand-new version whose genome equals `target`'s
    /// genome, parented on the current head, so `GenomeHistory` remains
    /// strictly append-only and fully auditable.
    pub fn rollback(
        &mut self,
        target: VersionId,
        reason: impl Into<String>,
    ) -> Result<VersionId, GenomeError> {
        let target_genome = self.get(target)?.genome.clone();
        Ok(self.commit(target_genome, reason.into()))
    }

    /// Structural diff between two versions' genomes.
    pub fn diff(&self, from: VersionId, to: VersionId) -> Result<GenomeDiff, GenomeError> {
        let from_genome = &self.get(from)?.genome;
        let to_genome = &self.get(to)?.genome;
        Ok(GenomeDiff::compute(from_genome, to_genome))
    }
}

/// Bumps the minor component of a `"major.minor"` label, e.g. `"1.0"` ->
/// `"1.1"`. Falls back to appending `.1` if the label isn't in that shape.
fn bump_minor(label: &str) -> String {
    if let Some((major, minor)) = label.split_once('.') {
        if let Ok(minor_num) = minor.parse::<u32>() {
            return format!("{}.{}", major, minor_num + 1);
        }
    }
    format!("{}.1", label)
}

/// Structural, field-level diff between two [`Genome`] snapshots.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct GenomeDiff {
    pub identity_changed: bool,
    pub values_added: Vec<String>,
    pub values_removed: Vec<String>,
    pub goals_added: Vec<String>,
    pub goals_removed: Vec<String>,
    pub beliefs_added: Vec<String>,
    pub beliefs_removed: Vec<String>,
    pub capabilities_added: Vec<String>,
    pub capabilities_removed: Vec<String>,
    pub skills_added: Vec<String>,
    pub skills_removed: Vec<String>,
    pub policies_added: Vec<String>,
    pub policies_removed: Vec<String>,
    pub preferences_changed: HashMap<String, (Option<String>, Option<String>)>,
}

impl GenomeDiff {
    fn compute(from: &Genome, to: &Genome) -> Self {
        let (values_added, values_removed) = diff_sets(&from.values, &to.values);
        let (goals_added, goals_removed) = diff_sets(&from.goals, &to.goals);
        let (beliefs_added, beliefs_removed) = diff_sets(&from.beliefs, &to.beliefs);
        let (capabilities_added, capabilities_removed) =
            diff_sets(&from.capabilities, &to.capabilities);
        let (skills_added, skills_removed) = diff_sets(&from.skills, &to.skills);
        let (policies_added, policies_removed) = diff_sets(&from.policies, &to.policies);

        let mut preferences_changed = HashMap::new();
        let pref_keys: HashSet<&String> = from
            .preferences
            .keys()
            .chain(to.preferences.keys())
            .collect();
        for key in pref_keys {
            let before = from.preferences.get(key).cloned();
            let after = to.preferences.get(key).cloned();
            if before != after {
                preferences_changed.insert(key.clone(), (before, after));
            }
        }

        Self {
            identity_changed: from.identity != to.identity,
            values_added,
            values_removed,
            goals_added,
            goals_removed,
            beliefs_added,
            beliefs_removed,
            capabilities_added,
            capabilities_removed,
            skills_added,
            skills_removed,
            policies_added,
            policies_removed,
            preferences_changed,
        }
    }

    /// True if `from` and `to` were structurally identical.
    pub fn is_empty(&self) -> bool {
        !self.identity_changed
            && self.values_added.is_empty()
            && self.values_removed.is_empty()
            && self.goals_added.is_empty()
            && self.goals_removed.is_empty()
            && self.beliefs_added.is_empty()
            && self.beliefs_removed.is_empty()
            && self.capabilities_added.is_empty()
            && self.capabilities_removed.is_empty()
            && self.skills_added.is_empty()
            && self.skills_removed.is_empty()
            && self.policies_added.is_empty()
            && self.policies_removed.is_empty()
            && self.preferences_changed.is_empty()
    }
}

/// Set difference between two orderless string lists: `(added, removed)`
/// relative to `from -> to`.
fn diff_sets(from: &[String], to: &[String]) -> (Vec<String>, Vec<String>) {
    let from_set: HashSet<&String> = from.iter().collect();
    let to_set: HashSet<&String> = to.iter().collect();
    let mut added: Vec<String> = to_set
        .difference(&from_set)
        .map(|s| s.to_string())
        .collect();
    let mut removed: Vec<String> = from_set
        .difference(&to_set)
        .map(|s| s.to_string())
        .collect();
    added.sort();
    removed.sort();
    (added, removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_genome() -> Genome {
        Genome::new("ADAM", "A persistent cognitive evolution layer")
    }

    #[test]
    fn init_creates_root_version_with_no_parent() {
        let history = GenomeHistory::init(base_genome(), "genesis");
        let head = history.head();
        assert_eq!(head.label, "1.0");
        assert!(head.parent.is_none());
        assert_eq!(head.reason, "genesis");
        assert_eq!(history.all().len(), 1);
    }

    #[test]
    fn commit_bumps_minor_version_and_links_parent() {
        let mut history = GenomeHistory::init(base_genome(), "genesis");
        let root_id = history.head_id();

        let mut next = base_genome();
        next.capabilities.push("rust-debugging".to_string());
        let v2 = history.commit(next, "Added Rust debugging capability");

        let version = history.get(v2).unwrap();
        assert_eq!(version.label, "1.1");
        assert_eq!(version.parent, Some(root_id));
        assert_eq!(history.head_id(), v2);
    }

    #[test]
    fn content_hash_is_stable_and_order_independent() {
        let mut a = base_genome();
        a.preferences
            .insert("tone".to_string(), "direct".to_string());
        a.preferences
            .insert("verbosity".to_string(), "low".to_string());

        let mut b = base_genome();
        b.preferences
            .insert("verbosity".to_string(), "low".to_string());
        b.preferences
            .insert("tone".to_string(), "direct".to_string());

        assert_eq!(a.content_hash(), b.content_hash());
    }

    #[test]
    fn content_hash_changes_when_genome_changes() {
        let a = base_genome();
        let mut b = base_genome();
        b.goals.push("ship phase 1".to_string());
        assert_ne!(a.content_hash(), b.content_hash());
    }

    #[test]
    fn history_is_immutable_and_rollback_appends_forward() {
        let mut history = GenomeHistory::init(base_genome(), "genesis");
        let v1 = history.head_id();

        let mut v2_genome = base_genome();
        v2_genome.capabilities.push("rust-debugging".to_string());
        let v2 = history.commit(v2_genome, "add rust debugging");

        let mut v3_genome = history.get(v2).unwrap().genome.clone();
        v3_genome.capabilities.push("go-debugging".to_string());
        let v3 = history.commit(v3_genome, "add go debugging");

        let rolled_back = history
            .rollback(v1, "regression: go-debugging broke tests")
            .unwrap();

        // History must never shrink or mutate prior entries.
        assert_eq!(history.all().len(), 4);
        assert_eq!(history.get(v1).unwrap().genome, base_genome());
        assert_eq!(
            history.get(v2).unwrap().genome.capabilities,
            vec!["rust-debugging".to_string()]
        );
        assert_eq!(
            history.get(v3).unwrap().genome.capabilities,
            vec!["rust-debugging".to_string(), "go-debugging".to_string()]
        );

        // The rollback version is new, forward-appended, and content-equal to v1.
        let rolled_back_version = history.get(rolled_back).unwrap();
        assert_eq!(rolled_back_version.genome, base_genome());
        assert_eq!(rolled_back_version.parent, Some(v3));
        assert_eq!(rolled_back_version.label, "1.3");
        assert_eq!(history.head_id(), rolled_back);
    }

    #[test]
    fn rollback_to_unknown_version_errors() {
        let mut history = GenomeHistory::init(base_genome(), "genesis");
        let bogus = Uuid::new_v4();
        assert_eq!(
            history.rollback(bogus, "oops"),
            Err(GenomeError::VersionNotFound(bogus))
        );
    }

    #[test]
    fn ancestors_walks_full_chain_to_root() {
        let mut history = GenomeHistory::init(base_genome(), "genesis");
        let v1 = history.head_id();
        let v2 = history.commit(base_genome(), "noop change 1");
        let v3 = history.commit(base_genome(), "noop change 2");

        let chain = history.ancestors(v3).unwrap();
        assert_eq!(chain, vec![v3, v2, v1]);
    }

    #[test]
    fn diff_reports_added_and_removed_fields() {
        let mut history = GenomeHistory::init(base_genome(), "genesis");
        let v1 = history.head_id();

        let mut next = base_genome();
        next.capabilities.push("rust-debugging".to_string());
        next.goals.push("ship phase 1".to_string());
        next.identity.description = "An evolved organism".to_string();
        let v2 = history.commit(next, "Added Rust debugging capability");

        let diff = history.diff(v1, v2).unwrap();
        assert!(diff.identity_changed);
        assert_eq!(diff.capabilities_added, vec!["rust-debugging".to_string()]);
        assert!(diff.capabilities_removed.is_empty());
        assert_eq!(diff.goals_added, vec!["ship phase 1".to_string()]);
        assert!(!diff.is_empty());
    }

    #[test]
    fn diff_between_identical_versions_is_empty() {
        let history = GenomeHistory::init(base_genome(), "genesis");
        let v1 = history.head_id();
        let diff = history.diff(v1, v1).unwrap();
        assert!(diff.is_empty());
    }

    #[test]
    fn diff_reports_preference_changes() {
        let mut history = GenomeHistory::init(base_genome(), "genesis");
        let v1 = history.head_id();

        let mut next = base_genome();
        next.preferences
            .insert("tone".to_string(), "direct".to_string());
        let v2 = history.commit(next, "set tone preference");

        let diff = history.diff(v1, v2).unwrap();
        assert_eq!(
            diff.preferences_changed.get("tone"),
            Some(&(None, Some("direct".to_string())))
        );
    }
}
