//! ADAM's link to EVE — the "validate inside EVE" step of the developmental
//! lifecycle.
//!
//! # What this crate used to be
//!
//! It was named for an integration that did not exist. The crate scored a
//! proposal by calling a Rust closure the *caller* supplied
//! (`TrialFn = dyn Fn(&EvolutionProposal) -> TrialOutcome`), averaging the pass
//! rate, and reporting the result as evidence. It shared nothing with the EVE
//! repository but the letters, and its doc comment glossed the name as
//! "Evaluate Via Experiment" to fit.
//!
//! The consequence was precise and serious. `Organism::accept_mutation` gates
//! every genome amendment beyond `preferences.*` on an approving evaluation —
//! but whoever supplied the closure decided what that evaluation said. The gate
//! was real; its evidence was self-supplied. An organism scoring its own
//! proposed changes is not measuring, it is asserting.
//!
//! # What it is now
//!
//! A client for the real EVE, speaking CP/1 over a process boundary.
//!
//! - [`EveClient`] projects an [`adam_evolution::EvolutionProposal`] onto a
//!   CP/1 `Mutation`, derives a reproducible seed, and asks EVE to measure it.
//! - [`Cp1Subprocess`] is the transport: it spawns EVE's `eve-cp1` endpoint,
//!   writes one request line, reads one response line.
//! - [`measure_and_verify`] enforces, for every provider, that the returned
//!   measurement was authored by EVE, concerns this exact mutation, and
//!   compared baseline against candidate over the same number of runs.
//!
//! There is no code path that produces a fitness result inside ADAM, and no
//! error variant meaning "measurement failed, proceed anyway".
//!
//! # What ADAM still owns
//!
//! Two things, because they are genuinely ADAM's:
//!
//! - [`intrinsic_risk`] — how consequential a change is, given what it touches.
//!   ADAM knows which of its own fields are load-bearing; EVE does not.
//! - [`derive_seed`] — which seed the measurement runs at, so re-validating the
//!   same proposal against the same genome reproduces the same experiment.
//!
//! The verdict itself is EVE's, and this crate's only role in it is to check
//! that it really came from EVE.
//!
//! # Example
//!
//! ```no_run
//! use adam_eve::{Cp1Subprocess, EveClient};
//! use adam_evolution::{EvolutionProposal, ProposalKind};
//!
//! let client = EveClient::new(Box::new(Cp1Subprocess::new()));
//! let proposal = EvolutionProposal::new(
//!     ProposalKind::AmendGenome {
//!         field: "preferences.thoroughness".to_string(),
//!         current_value: "low".to_string(),
//!         suggested_value: "high".to_string(),
//!     },
//!     "the organism keeps missing information it had access to",
//!     vec!["4 recurring conflicts on the same topic".to_string()],
//!     0.7,
//! );
//!
//! // Spawns EVE's endpoint and measures the mutation counterfactually.
//! let fitness = client.validate(&proposal, "genome-hash-before", "genome-hash-after")?;
//! println!("{:?}: {}", fitness.recommendation, fitness.reason);
//! # Ok::<(), adam_eve::FitnessError>(())
//! ```

mod client;
mod provider;

pub use client::{derive_seed, intrinsic_risk, to_mutation, EveClient, ValidationConfig};
pub use provider::{
    decode_response, measure_and_verify, Cp1Subprocess, FitnessError, FitnessProvider, StubProvider,
};

// Re-exported so callers of this crate need not also depend on `adam-protocol`
// just to name the types this crate returns.
pub use adam_protocol::{FitnessResult, Measurement, Recommendation, ValidationRequest};

#[cfg(test)]
mod tests {
    use super::*;
    use adam_protocol::{
        BasisPoints, Component, Measurement as M, Provenance, SignedBasisPoints, SignedEnvelope,
    };

    fn result_line(mutation_id: &str, author: Component, fleet_key: Option<&[u8]>) -> String {
        let result = FitnessResult {
            cp: "cp1".to_string(),
            doc_type: "FitnessResult".to_string(),
            id: "cccccccc-cccc-4ccc-8ccc-cccccccccccc".to_string(),
            mutation_id: mutation_id.to_string(),
            seed: 1337,
            scenario_ids: vec!["excellent".to_string()],
            trials: 3,
            baseline: M::experience(
                BasisPoints::from_ratio(0.64),
                BasisPoints::from_ratio(0.7),
                BasisPoints::from_ratio(0.3),
                BasisPoints::from_ratio(0.6),
                BasisPoints::from_ratio(0.4),
                9,
            ),
            candidate: M::experience(
                BasisPoints::from_ratio(0.71),
                BasisPoints::from_ratio(0.78),
                BasisPoints::from_ratio(0.26),
                BasisPoints::from_ratio(0.65),
                BasisPoints::from_ratio(0.39),
                9,
            ),
            delta_bp: SignedBasisPoints::new(700),
            recommendation: Recommendation::Approve,
            reason: "candidate improved the composite by 700bp".to_string(),
            provenance: Provenance::now(author, "eve:cp1/validate"),
        };
        let sealed = serde_json::to_value(&result).unwrap();
        let mut sealed = sealed;
        adam_protocol::seal(&mut sealed).unwrap();
        SignedEnvelope::seal(&sealed, fleet_key).unwrap().to_line()
    }

    #[test]
    fn a_well_formed_response_decodes_to_a_fitness_result() {
        let result = decode_response(&result_line("m1", Component::Eve, None), None).unwrap();
        assert_eq!(result.mutation_id, "m1");
        assert_eq!(result.recommendation, Recommendation::Approve);
        assert_eq!(result.delta_bp.raw(), 700);
        assert!(result.is_authentic("m1", Component::Eve));
    }

    #[test]
    fn a_signed_response_decodes_under_the_matching_key() {
        let line = result_line("m1", Component::Eve, Some(b"fleet"));
        assert!(decode_response(&line, Some(b"fleet")).is_ok());
        assert!(matches!(
            decode_response(&line, Some(b"other")).unwrap_err(),
            FitnessError::Envelope(_)
        ));
    }

    #[test]
    fn a_protocol_error_is_surfaced_as_a_refusal_not_a_verdict() {
        // EVE frames a request it could not understand as a bare ProtocolError
        // rather than an envelope, because there is no document to wrap.
        let line = r#"{"cp":"cp1","type":"ProtocolError","detail":"unknown scenario \"nope\""}"#;
        let err = decode_response(line, None).unwrap_err();
        assert!(matches!(err, FitnessError::ProtocolError(_)));
        assert!(err.to_string().contains("unknown scenario"));
    }

    #[test]
    fn a_tampered_response_is_refused() {
        let line = result_line("m1", Component::Eve, None).replace("700", "9700");
        assert!(decode_response(&line, None).is_err());
    }

    #[test]
    fn a_response_carrying_the_wrong_document_type_is_refused() {
        let mut document = serde_json::json!({
            "cp": "cp1",
            "type": "Mutation",
            "id": "88888888-8888-4888-8888-888888888888",
            "provenance": {
                "authored_by": "eve",
                "produced_at": "2026-01-01T00:00:00.000Z",
                "origin": "o",
                "evidence": [],
                "derived_from": []
            }
        });
        adam_protocol::seal(&mut document).unwrap();
        let line = SignedEnvelope::seal(&document, None).unwrap().to_line();
        assert!(matches!(
            decode_response(&line, None).unwrap_err(),
            FitnessError::NotAFitnessResult(_)
        ));
    }

    #[test]
    fn a_non_envelope_line_is_refused() {
        assert!(decode_response("not json at all", None).is_err());
    }

    #[test]
    fn measure_and_verify_rejects_an_adam_authored_result_from_any_provider() {
        let request = EveClient::new(Box::new(StubProvider::failing("unused"))).request_for(
            &adam_evolution::EvolutionProposal::new(
                adam_evolution::ProposalKind::RetireSkill {
                    skill_name: "flaky".to_string(),
                },
                "chronically failing",
                vec![],
                0.9,
            ),
            "before",
            "after",
        );
        let forged = decode_response(
            &result_line(&request.mutation.id, Component::Adam, None),
            None,
        )
        .unwrap();
        let provider = StubProvider::returning(forged);
        let err = measure_and_verify(&provider, &request).unwrap_err();
        assert!(
            err.to_string().contains("authored by adam")
                && err.to_string().contains("is not an evaluator"),
            "unexpected reason: {err}"
        );
    }

    #[test]
    fn a_missing_provider_binary_fails_loudly_rather_than_silently_approving() {
        let client = EveClient::new(Box::new(Cp1Subprocess::command(
            "definitely-not-a-real-binary-adam-eve-test",
            vec![],
        )));
        let err = client
            .validate(
                &adam_evolution::EvolutionProposal::new(
                    adam_evolution::ProposalKind::RetireSkill {
                        skill_name: "flaky".to_string(),
                    },
                    "chronically failing",
                    vec![],
                    0.9,
                ),
                "before",
                "after",
            )
            .unwrap_err();
        assert!(matches!(err, FitnessError::Spawn { .. }));
        assert!(err.to_string().contains("could not start"));
    }
}
