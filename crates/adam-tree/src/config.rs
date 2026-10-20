//! Tunable thresholds for the tree layer (spec section 3). The
//! defaults are the spec's v1 starting values, chosen so a normal
//! thread summarizes roughly weekly.

use std::time::Duration;

/// Configuration for [`crate::TreeStore`]. Every trigger threshold is
/// configuration, not a constant.
#[derive(Debug, Clone)]
pub struct TreeConfig {
    /// Open leaves in one thread at or above this count make an L1
    /// pass due.
    pub leaf_count_trigger: usize,
    /// Open leaves in one thread whose token sum reaches this make an
    /// L1 pass due.
    pub leaf_token_trigger: usize,
    /// The oldest open leaf being older than this many days (with at
    /// least `leaf_age_min_count` open leaves) makes an L1 pass due.
    pub leaf_age_trigger_days: i64,
    /// Minimum open leaves before the age trigger can fire, so quiet
    /// threads never produce one-leaf summaries.
    pub leaf_age_min_count: usize,
    /// Open child nodes at or above this count make an L(n+1) pass due.
    pub child_count_trigger: usize,
    /// Open child nodes whose token sum reaches this make an L(n+1)
    /// pass due.
    pub child_token_trigger: usize,
    /// How long a summarizer lease lasts without renewal.
    pub lease_ttl: Duration,
    /// SQLite busy timeout applied to the layer's sidecar connection.
    pub busy_timeout: Duration,
    /// Target summary size as a fraction of the covered children's
    /// token sum, at L1 (spec section 5).
    pub l1_target_ratio: f32,
    /// Target summary size as a fraction of the covered children's
    /// token sum, above L1 (spec section 5).
    pub ln_target_ratio: f32,
    /// Default `read_context` token budget when the caller asks for
    /// none (spec section 4).
    pub default_token_budget: usize,
}

impl Default for TreeConfig {
    fn default() -> Self {
        Self {
            leaf_count_trigger: 24,
            leaf_token_trigger: 6_000,
            leaf_age_trigger_days: 7,
            leaf_age_min_count: 6,
            child_count_trigger: 8,
            child_token_trigger: 8_000,
            lease_ttl: Duration::from_secs(600),
            busy_timeout: Duration::from_secs(5),
            l1_target_ratio: 0.12,
            ln_target_ratio: 0.20,
            default_token_budget: 8_000,
        }
    }
}
