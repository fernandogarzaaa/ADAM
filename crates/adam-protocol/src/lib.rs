//! CP/1 — the Cognitive Protocol, version 1, as ADAM implements it.
//!
//! ADAM, EVE and AXIOM-AETHER are one organism. CP/1 is the stable wire
//! contract between them: twelve canonical types each with exactly one owning
//! repository (`FitnessResult` excepted — two evaluators may author it, and
//! neither of them is ADAM), a closed set of fifteen events, mandatory chained
//! provenance,
//! and a canonical byte encoding with no floating point on the wire.
//!
//! # Vendored, not depended on
//!
//! The normative source lives in AXIOM-AETHER at `protocol/cp1/`. This
//! repository vendors a copy under its own `protocol/cp1/` and implements a
//! hand-written binding here. **There is no build-time dependency on either
//! other repository, in either direction** — ADAM cannot link a TypeScript
//! simulation engine and would not want to link a `candle`-backed inference
//! crate, so the connection between them is data, not symbols.
//!
//! Drift is caught by [`conformance`], which verifies this binding against the
//! shared fixture corpus and the corpus against `MANIFEST.sha256`. A change to
//! the normative source that has not been re-vendored here fails ADAM's test
//! suite loudly, before anything ships.
//!
//! # What this crate deliberately does not contain
//!
//! ADAM owns eight canonical types, but [`types`] declares only the three that
//! actually cross a boundary today: [`types::Mutation`] and
//! [`types::ValidationRequest`] going out to EVE, and [`types::FitnessResult`]
//! coming back. The rest (`Genome`, `Identity`, `Belief`, `Memory`, `Skill`,
//! `Capability`, `Reflection`) appear on the wire only as event *subjects* —
//! referenced by id and type, never as payloads. Declaring a struct for each
//! now would create abstractions with no caller that must nonetheless be kept
//! in step with the schema forever.
//!
//! # The gate this crate exists to make real
//!
//! ADAM's acceptance path refuses any genome amendment beyond `preferences.*`
//! without an approving fitness measurement. Before CP/1, that "measurement"
//! came from a Rust closure the caller supplied — the organism scoring its own
//! proposals. [`types::FitnessResult::is_authentic`] is what replaces it: a
//! result only counts if EVE authored it, it concerns this exact mutation, and
//! both sides of the comparison ran the same number of times.
//!
//! # Example
//!
//! ```
//! use adam_protocol::{
//!     canonical,
//!     types::{BasisPoints, Component, Mutation, MutationKind, MutationStatus, Provenance},
//!     SignedEnvelope,
//! };
//!
//! let mutation = Mutation::new(
//!     "88888888-8888-4888-8888-888888888888",
//!     MutationKind::AmendGenome,
//!     "preferences.thoroughness",
//!     Some("low".to_string()),
//!     Some("high".to_string()),
//!     "the organism keeps missing information it had access to",
//!     BasisPoints::from_f32(0.7),
//!     BasisPoints::from_f32(0.2),
//!     MutationStatus::Proposed,
//!     Provenance::now(Component::Adam, "evolution:analyze"),
//! );
//!
//! let wire = SignedEnvelope::seal(&mutation.seal().unwrap(), None).unwrap();
//! let received = wire.open(None).unwrap();
//! assert!(canonical::verify_seal(&received).unwrap());
//! assert_eq!(received["confidence_bp"], serde_json::json!(7000));
//! ```

pub mod canonical;
pub mod conformance;
pub mod envelope;
pub mod event;
pub mod types;

/// The protocol identifier carried by every CP/1 document.
pub const CP: &str = "cp1";

/// The revision of the normative source this binding implements.
///
/// Kept in step with the vendored `protocol/cp1/VERSION` by
/// [`tests::version_matches_the_vendored_source`].
pub const VERSION: &str = "1.1.0";

pub use canonical::{content_hash, seal, to_canonical, verify_seal, CanonicalError};
pub use envelope::{EnvelopeError, SignedEnvelope};
pub use event::{Event, EventKind, EventSink, NullSink, PayloadValue, RecordingSink, SubjectType};
pub use types::{
    BasisPoints, Component, FitnessResult, GenomePair, Measurement, MeasurementPlan, Mutation,
    MutationKind, MutationStatus, Provenance, Recommendation, SignedBasisPoints, Timestamp,
    ValidationRequest,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_matches_the_vendored_source() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../protocol/cp1/VERSION");
        let declared =
            std::fs::read_to_string(path).unwrap_or_else(|err| panic!("cannot read {path}: {err}"));
        assert_eq!(
            declared.trim(),
            VERSION,
            "this binding claims CP/1 {VERSION} but the vendored source is at {}",
            declared.trim()
        );
    }
}
