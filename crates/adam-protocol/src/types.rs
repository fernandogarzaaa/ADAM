//! CP/1 canonical types that ADAM puts on the wire.
//!
//! ADAM authors eight canonical types, but only some of them cross a boundary
//! today. This module ships typed structs for exactly those:
//!
//! - [`Mutation`] and [`ValidationRequest`] — sent to EVE for measurement.
//! - [`FitnessResult`] — read back from EVE and used to gate acceptance.
//!
//! The other ADAM-owned types (`Genome`, `Identity`, `Belief`, `Memory`,
//! `Skill`, `Capability`, `Reflection`) appear on the wire only as event
//! *subjects*: referenced by `subject_id` and `subject_type`, never as
//! payloads. A struct for each would be an abstraction with no caller,
//! requiring perpetual maintenance against a schema it never exercises — so
//! they are deliberately absent until something actually transmits them, at
//! which point they belong here.
//!
//! Cross-binding agreement on the absent types is still established: the
//! conformance suite validates every canonical type structurally over the
//! shared fixture corpus, which tests the encoding — the thing that has to
//! match.

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::canonical::{self, CanonicalError};

/// Which repository authored a document. Authorship is exclusive per type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Component {
    Adam,
    Eve,
    Axiom,
}

impl Component {
    pub fn as_str(self) -> &'static str {
        match self {
            Component::Adam => "adam",
            Component::Eve => "eve",
            Component::Axiom => "axiom",
        }
    }
}

/// A ratio in `[0, 1]` carried as an integer in `[0, 10000]`.
///
/// Every fractional quantity ADAM sends crosses a boundary through this type.
/// Conversion is the only place rounding happens, and it is explicit — which
/// matters because ADAM's internals are `f32` throughout and EVE's are IEEE-754
/// doubles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct BasisPoints(u16);

/// Deserialization is hand-written because the derived one would accept any
/// `u16` — including `40000`, which is outside `[0, 10000]` and which the
/// schema rejects. A document that parses into a value the schema forbids is
/// worse than a parse error: it flows through the organism looking valid and
/// only fails when something else re-validates it, far from the boundary that
/// let it in.
impl<'de> Deserialize<'de> for BasisPoints {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = u16::deserialize(deserializer)?;
        if raw > 10_000 {
            return Err(serde::de::Error::custom(format!(
                "{raw} is outside the basis-point range [0, 10000]"
            )));
        }
        Ok(BasisPoints(raw))
    }
}

impl BasisPoints {
    pub const ZERO: BasisPoints = BasisPoints(0);
    pub const ONE: BasisPoints = BasisPoints(10_000);

    /// Convert a ratio to basis points, clamping to `[0, 1]` and rounding half
    /// away from zero.
    ///
    /// Clamping rather than erroring is right here: ADAM's confidences and
    /// fitness scores are already clamped `f32`s that can land a hair outside
    /// the range through accumulated error, and failing a whole lifecycle turn
    /// over a `1.0000001` would be worse than recording `10000`.
    pub fn from_ratio(ratio: f64) -> Self {
        if !ratio.is_finite() {
            return BasisPoints::ZERO;
        }
        BasisPoints((ratio.clamp(0.0, 1.0) * 10_000.0).round() as u16)
    }

    /// Convenience for ADAM's pervasive `f32` scores.
    pub fn from_f32(ratio: f32) -> Self {
        Self::from_ratio(f64::from(ratio))
    }

    pub const fn raw(self) -> u16 {
        self.0
    }

    pub fn as_ratio(self) -> f64 {
        f64::from(self.0) / 10_000.0
    }

    pub fn as_f32(self) -> f32 {
        f32::from(self.0) / 10_000.0
    }
}

/// A signed ratio in `[-1, 1]` as an integer in `[-10000, 10000]`.
///
/// Distinct from [`BasisPoints`] because a fitness delta is meaningfully
/// negative: a mutation may make the organism worse, and that is the finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SignedBasisPoints(i32);

impl SignedBasisPoints {
    pub fn new(raw: i32) -> Self {
        Self(raw.clamp(-10_000, 10_000))
    }

    pub const fn raw(self) -> i32 {
        self.0
    }
}

/// RFC 3339 UTC with exactly millisecond precision.
///
/// Fixed precision is a hashing requirement: a timestamp rendered with
/// microseconds in one binding and seconds in another produces two different
/// content hashes for the same document.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Timestamp(String);

impl Timestamp {
    pub fn now() -> Self {
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            // A clock before the epoch is a misconfigured host, not a
            // recoverable condition for a provenance record; the epoch is the
            // only defensible stand-in and is obviously wrong on sight.
            .unwrap_or(0);
        Self::from_unix_millis(millis)
    }

    /// Render a Unix millisecond timestamp in CP/1's fixed format.
    ///
    /// ADAM depends on `chrono` already, but this deliberately does not: the
    /// protocol crate is vendored and must stay dependency-light so a future
    /// consumer can copy it without inheriting ADAM's dependency graph.
    pub fn from_unix_millis(millis: i64) -> Self {
        let days = millis.div_euclid(86_400_000);
        let ms_of_day = millis.rem_euclid(86_400_000);
        let (year, month, dom) = civil_from_days(days);
        let seconds = ms_of_day / 1000;
        Self(format!(
            "{year:04}-{month:02}-{dom:02}T{:02}:{:02}:{:02}.{:03}Z",
            seconds / 3600,
            (seconds / 60) % 60,
            seconds % 60,
            ms_of_day % 1000,
        ))
    }

    /// Accept a string only if it matches CP/1's fixed shape exactly.
    pub fn parse(value: &str) -> Result<Self, String> {
        let bytes = value.as_bytes();
        let shape_ok = bytes.len() == 24
            && bytes[4] == b'-'
            && bytes[7] == b'-'
            && bytes[10] == b'T'
            && bytes[13] == b':'
            && bytes[16] == b':'
            && bytes[19] == b'.'
            && bytes[23] == b'Z'
            && [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 22]
                .iter()
                .all(|&i| bytes[i].is_ascii_digit());
        if shape_ok {
            Ok(Self(value.to_string()))
        } else {
            Err(format!(
                "expected RFC 3339 UTC with millisecond precision (YYYY-MM-DDTHH:MM:SS.sssZ), got {value:?}"
            ))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Days since the Unix epoch to a civil `(year, month, day)`. Howard Hinnant's
/// `civil_from_days`, exact for the proleptic Gregorian calendar.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Chain of custody. Mandatory on every CP/1 document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    pub authored_by: Component,
    pub produced_at: Timestamp,
    pub origin: String,
    pub evidence: Vec<String>,
    /// Ids of the documents this one was computed from, making the organism's
    /// knowledge a walkable DAG rather than a pile of assertions.
    pub derived_from: Vec<String>,
    /// Written by [`crate::canonical::seal`]. Empty until sealed; a document
    /// that crosses a boundary unsealed is malformed.
    #[serde(default)]
    pub content_hash: String,
}

impl Provenance {
    pub fn now(authored_by: Component, origin: impl Into<String>) -> Self {
        Self {
            authored_by,
            produced_at: Timestamp::now(),
            origin: origin.into(),
            evidence: Vec::new(),
            derived_from: Vec::new(),
            content_hash: String::new(),
        }
    }

    pub fn with_evidence(mut self, evidence: impl IntoIterator<Item = String>) -> Self {
        self.evidence.extend(evidence);
        self
    }

    pub fn derived_from(mut self, ids: impl IntoIterator<Item = String>) -> Self {
        self.derived_from.extend(ids);
        self
    }
}

/// The category of change a mutation proposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MutationKind {
    AmendGenome,
    RetireSkill,
    ReconcileBelief,
    InvestigateConflict,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MutationStatus {
    Proposed,
    Validating,
    Accepted,
    Rejected,
}

/// A proposed change to genome, skills or beliefs.
///
/// `risk_bp` is intrinsic to the kind and target, never derived from
/// `confidence_bp`. Letting a proposal lower its own risk by asserting
/// confidence in itself would defeat the point of measuring it externally.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mutation {
    pub cp: String,
    #[serde(rename = "type")]
    pub doc_type: String,
    pub id: String,
    pub kind: MutationKind,
    pub target: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_value: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proposed_value: Option<String>,
    pub rationale: String,
    pub confidence_bp: BasisPoints,
    pub risk_bp: BasisPoints,
    pub status: MutationStatus,
    pub provenance: Provenance,
}

impl Mutation {
    /// Assemble a mutation. `cp` and `type` are set here rather than accepted,
    /// so a caller cannot mint a document claiming to be something else.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: impl Into<String>,
        kind: MutationKind,
        target: impl Into<String>,
        current_value: Option<String>,
        proposed_value: Option<String>,
        rationale: impl Into<String>,
        confidence_bp: BasisPoints,
        risk_bp: BasisPoints,
        status: MutationStatus,
        provenance: Provenance,
    ) -> Self {
        Self {
            cp: crate::CP.to_string(),
            doc_type: "Mutation".to_string(),
            id: id.into(),
            kind,
            target: target.into(),
            current_value,
            proposed_value,
            rationale: rationale.into(),
            confidence_bp,
            risk_bp,
            status,
            provenance,
        }
    }

    /// Serialize to a sealed `serde_json::Value` ready for transport.
    pub fn seal(&self) -> Result<serde_json::Value, CanonicalError> {
        let mut value = serde_json::to_value(self).expect("Mutation always serializes");
        canonical::seal(&mut value)?;
        Ok(value)
    }
}

/// ADAM asks EVE to measure a mutation.
///
/// Carries genome *hashes* rather than genomes: EVE does not need the
/// organism's state to run scenarios, and shipping it across the boundary would
/// leak ADAM-owned data into a component with no business authoring it. The
/// hashes still pin the resulting measurement to the exact genome pair.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationRequest {
    pub cp: String,
    #[serde(rename = "type")]
    pub doc_type: String,
    pub id: String,
    pub mutation: Mutation,
    pub genome_before_hash: String,
    pub genome_after_hash: String,
    pub scenario_ids: Vec<String>,
    /// The seed both of EVE's runs use. Determinism depends on it, so it is
    /// required rather than defaulted.
    pub seed: u32,
    pub trials: u32,
    pub provenance: Provenance,
}

/// The genome pair a measurement is pinned to.
///
/// Grouped rather than passed as two loose strings because they are only
/// meaningful together — a before-hash without its after-hash does not identify
/// a change — and because two adjacent parameters of the same type are exactly
/// the shape that gets transposed at a call site without the compiler noticing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenomePair {
    pub before_hash: String,
    pub after_hash: String,
}

impl GenomePair {
    pub fn new(before_hash: impl Into<String>, after_hash: impl Into<String>) -> Self {
        Self {
            before_hash: before_hash.into(),
            after_hash: after_hash.into(),
        }
    }
}

/// How many times, and against what, a measurement runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeasurementPlan {
    pub scenario_ids: Vec<String>,
    /// The seed both of EVE's runs use.
    pub seed: u32,
    pub trials: u32,
}

impl ValidationRequest {
    pub fn new(
        id: impl Into<String>,
        mutation: Mutation,
        genomes: GenomePair,
        plan: MeasurementPlan,
        provenance: Provenance,
    ) -> Self {
        Self {
            cp: crate::CP.to_string(),
            doc_type: "ValidationRequest".to_string(),
            id: id.into(),
            mutation,
            genome_before_hash: genomes.before_hash,
            genome_after_hash: genomes.after_hash,
            scenario_ids: plan.scenario_ids,
            seed: plan.seed,
            // A measurement over zero trials is not a measurement.
            trials: plan.trials.max(1),
            provenance,
        }
    }

    pub fn seal(&self) -> Result<serde_json::Value, CanonicalError> {
        let mut value = serde_json::to_value(self).expect("ValidationRequest always serializes");
        canonical::seal(&mut value)?;
        Ok(value)
    }
}

/// One side of a counterfactual fitness comparison.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Measurement {
    pub composite_bp: BasisPoints,
    pub task_success_bp: BasisPoints,
    pub frustration_bp: BasisPoints,
    pub trust_bp: BasisPoints,
    pub cognitive_load_bp: BasisPoints,
    pub runs: u32,
}

/// What EVE recommends doing with a mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Recommendation {
    Approve,
    NeedsReview,
    Reject,
}

/// Evidence-backed measurement of a mutation, authored by EVE.
///
/// ADAM reads this and must never mint one. The check is
/// [`FitnessResult::is_authentic`], and it is what makes the acceptance gate
/// mean something: a component scoring its own proposed changes is not
/// measuring, it is asserting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FitnessResult {
    pub cp: String,
    #[serde(rename = "type")]
    pub doc_type: String,
    pub id: String,
    pub mutation_id: String,
    pub seed: u32,
    pub scenario_ids: Vec<String>,
    pub trials: u32,
    pub baseline: Measurement,
    pub candidate: Measurement,
    pub delta_bp: SignedBasisPoints,
    pub recommendation: Recommendation,
    pub reason: String,
    pub provenance: Provenance,
}

impl FitnessResult {
    /// Whether this result is a measurement ADAM may rely on.
    ///
    /// Three independent conditions, each of which has to hold for the number
    /// to mean what it claims:
    ///
    /// - It was authored by EVE. ADAM minting its own fitness would be exactly
    ///   the self-scoring this design removes.
    /// - It concerns `mutation_id`. A result pinned to a different proposal is
    ///   evidence about something else.
    /// - Baseline and candidate were measured over the same number of runs.
    ///   The comparison is only counterfactual if both sides saw the same work;
    ///   differing run counts mean one side was measured differently.
    pub fn is_authentic(&self, mutation_id: &str) -> bool {
        self.provenance.authored_by == Component::Eve
            && self.mutation_id == mutation_id
            && self.baseline.runs == self.candidate.runs
            && self.doc_type == "FitnessResult"
    }

    /// Why [`FitnessResult::is_authentic`] returned false, for an error message.
    pub fn authenticity_failure(&self, mutation_id: &str) -> Option<String> {
        if self.doc_type != "FitnessResult" {
            return Some(format!(
                "document is a {}, not a FitnessResult",
                self.doc_type
            ));
        }
        if self.provenance.authored_by != Component::Eve {
            return Some(format!(
                "authored by {}, but only EVE may author a FitnessResult",
                self.provenance.authored_by.as_str()
            ));
        }
        if self.mutation_id != mutation_id {
            return Some(format!(
                "measures mutation {}, not {mutation_id}",
                self.mutation_id
            ));
        }
        if self.baseline.runs != self.candidate.runs {
            return Some(format!(
                "baseline ran {} times and candidate {}, so the comparison is not counterfactual",
                self.baseline.runs, self.candidate.runs
            ));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provenance(component: Component) -> Provenance {
        Provenance::now(component, "test")
    }

    fn measurement(runs: u32) -> Measurement {
        Measurement {
            composite_bp: BasisPoints::from_ratio(0.64),
            task_success_bp: BasisPoints::from_ratio(0.7),
            frustration_bp: BasisPoints::from_ratio(0.3),
            trust_bp: BasisPoints::from_ratio(0.6),
            cognitive_load_bp: BasisPoints::from_ratio(0.4),
            runs,
        }
    }

    fn fitness(
        author: Component,
        mutation_id: &str,
        baseline_runs: u32,
        candidate_runs: u32,
    ) -> FitnessResult {
        FitnessResult {
            cp: crate::CP.to_string(),
            doc_type: "FitnessResult".to_string(),
            id: "cccccccc-cccc-4ccc-8ccc-cccccccccccc".to_string(),
            mutation_id: mutation_id.to_string(),
            seed: 1337,
            scenario_ids: vec!["excellent".to_string()],
            trials: 3,
            baseline: measurement(baseline_runs),
            candidate: measurement(candidate_runs),
            delta_bp: SignedBasisPoints::new(700),
            recommendation: Recommendation::Approve,
            reason: "improved".to_string(),
            provenance: provenance(author),
        }
    }

    #[test]
    fn basis_points_round_half_away_from_zero_and_clamp() {
        assert_eq!(BasisPoints::from_ratio(0.00005).raw(), 1);
        assert_eq!(BasisPoints::from_ratio(0.5).raw(), 5_000);
        assert_eq!(BasisPoints::from_ratio(1.5).raw(), 10_000);
        assert_eq!(BasisPoints::from_ratio(-0.2).raw(), 0);
        assert_eq!(BasisPoints::from_ratio(f64::NAN).raw(), 0);
    }

    #[test]
    fn basis_points_convert_adams_f32_scores() {
        assert_eq!(BasisPoints::from_f32(0.82f32).raw(), 8_200);
        assert!((BasisPoints::from_f32(0.82f32).as_f32() - 0.82).abs() < 1e-6);
    }

    #[test]
    fn signed_basis_points_clamp_to_range() {
        assert_eq!(SignedBasisPoints::new(-99_999).raw(), -10_000);
        assert_eq!(SignedBasisPoints::new(99_999).raw(), 10_000);
        assert_eq!(SignedBasisPoints::new(-700).raw(), -700);
    }

    #[test]
    fn timestamp_renders_known_instants() {
        assert_eq!(
            Timestamp::from_unix_millis(0).as_str(),
            "1970-01-01T00:00:00.000Z"
        );
        assert_eq!(
            Timestamp::from_unix_millis(1_767_225_600_000).as_str(),
            "2026-01-01T00:00:00.000Z"
        );
        // A leap day, to exercise civil_from_days properly.
        assert_eq!(
            Timestamp::from_unix_millis(1_709_164_800_123).as_str(),
            "2024-02-29T00:00:00.123Z"
        );
    }

    #[test]
    fn timestamp_parse_rejects_other_precisions() {
        assert!(Timestamp::parse("2026-01-01T00:00:00Z").is_err());
        assert!(Timestamp::parse("2026-01-01T00:00:00.123456Z").is_err());
        assert!(Timestamp::parse("2026-01-01T00:00:00.123Z").is_ok());
        assert!(Timestamp::parse(Timestamp::now().as_str()).is_ok());
    }

    #[test]
    fn a_sealed_mutation_verifies_and_carries_no_floats() {
        let mutation = Mutation::new(
            "88888888-8888-4888-8888-888888888888",
            MutationKind::AmendGenome,
            "preferences.tone",
            Some("verbose".to_string()),
            Some("concise".to_string()),
            "user prefers concise responses",
            BasisPoints::from_f32(0.9),
            BasisPoints::from_f32(0.1),
            MutationStatus::Proposed,
            provenance(Component::Adam),
        );
        let sealed = mutation.seal().unwrap();
        assert!(canonical::verify_seal(&sealed).unwrap());
        assert_eq!(sealed["type"], serde_json::json!("Mutation"));
        assert_eq!(sealed["confidence_bp"], serde_json::json!(9000));
    }

    #[test]
    fn absent_mutation_values_become_absent_keys_never_nulls() {
        let mutation = Mutation::new(
            "88888888-8888-4888-8888-888888888888",
            MutationKind::ReconcileBelief,
            "some statement",
            None,
            None,
            "belief keeps collapsing",
            BasisPoints::ONE,
            BasisPoints::ZERO,
            MutationStatus::Proposed,
            provenance(Component::Adam),
        );
        let sealed = mutation.seal().unwrap();
        assert!(sealed.get("current_value").is_none());
        assert!(sealed.get("proposed_value").is_none());
        // Would have failed on a null, proving the point.
        assert!(canonical::to_canonical(&sealed).is_ok());
    }

    #[test]
    fn a_validation_request_never_asks_for_zero_trials() {
        let request = ValidationRequest::new(
            "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
            Mutation::new(
                "88888888-8888-4888-8888-888888888888",
                MutationKind::RetireSkill,
                "flaky",
                None,
                None,
                "chronically failing",
                BasisPoints::ONE,
                BasisPoints::ZERO,
                MutationStatus::Proposed,
                provenance(Component::Adam),
            ),
            GenomePair::new("a".repeat(64), "b".repeat(64)),
            MeasurementPlan {
                scenario_ids: vec!["excellent".to_string()],
                seed: 1337,
                trials: 0,
            },
            provenance(Component::Adam),
        );
        assert_eq!(request.trials, 1);
        assert!(canonical::verify_seal(&request.seal().unwrap()).unwrap());
    }

    #[test]
    fn a_basis_point_value_above_the_range_is_refused_at_the_boundary() {
        // The derived impl would accept any u16. A document that parses into a
        // value the schema forbids is worse than a parse error: it flows on
        // looking valid and fails somewhere far from the boundary that let it
        // in.
        let err = serde_json::from_str::<BasisPoints>("10001").unwrap_err();
        assert!(
            err.to_string().contains("outside the basis-point range"),
            "{err}"
        );
        assert_eq!(
            serde_json::from_str::<BasisPoints>("10000").unwrap(),
            BasisPoints::ONE
        );
        assert_eq!(
            serde_json::from_str::<BasisPoints>("0").unwrap(),
            BasisPoints::ZERO
        );
    }

    #[test]
    fn a_negative_basis_point_value_is_refused_too() {
        assert!(serde_json::from_str::<BasisPoints>("-1").is_err());
    }

    #[test]
    fn an_eve_authored_result_for_the_right_mutation_is_authentic() {
        let result = fitness(Component::Eve, "m1", 9, 9);
        assert!(result.is_authentic("m1"));
        assert_eq!(result.authenticity_failure("m1"), None);
    }

    #[test]
    fn a_self_authored_result_is_never_authentic() {
        // The whole point: ADAM may not score its own proposals.
        let result = fitness(Component::Adam, "m1", 9, 9);
        assert!(!result.is_authentic("m1"));
        assert!(result
            .authenticity_failure("m1")
            .unwrap()
            .contains("only EVE may author"));
    }

    #[test]
    fn a_result_for_a_different_mutation_is_not_evidence_about_this_one() {
        let result = fitness(Component::Eve, "other", 9, 9);
        assert!(!result.is_authentic("m1"));
        assert!(result
            .authenticity_failure("m1")
            .unwrap()
            .contains("not m1"));
    }

    #[test]
    fn mismatched_run_counts_break_the_counterfactual() {
        let result = fitness(Component::Eve, "m1", 9, 3);
        assert!(!result.is_authentic("m1"));
        assert!(result
            .authenticity_failure("m1")
            .unwrap()
            .contains("not counterfactual"));
    }
}
