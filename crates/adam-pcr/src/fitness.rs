//! Measuring a mutation against the real workspace.
//!
//! # Why this exists
//!
//! EVE answers one question: does the change make a simulated human's
//! experience better? That is a real question, but it is not the only one, and
//! it is not the one the workspace asks. The workspace asks whether more
//! records got processed. A mutation can improve one and harm the other, and an
//! organism that can only hear EVE cannot notice that it happened.
//!
//! So this is a *second* evaluator, not a replacement. It implements the same
//! [`FitnessProvider`] seam, returns the same [`FitnessResult`], and is
//! verified by the same [`adam_eve::measure_and_verify`]. Nothing about the
//! adaptation gate changes; it simply now has a second thing that can speak to
//! it.
//!
//! # What it does not do
//!
//! It reports `composite_bp` and `runs`, and nothing else. There is no
//! simulated human here, so there is no frustration, trust or cognitive-load
//! number to report, and [`Measurement::objective`] gives no way to invent one.
//! Absent means *not measured*, which is the truth, and is not the same as
//! zero.
//!
//! # Honest limits
//!
//! EVE runs in a subprocess, so the boundary that stops ADAM authoring EVE's
//! evidence is enforced by the operating system. This provider runs *in ADAM's
//! own process*. It authors `Component::Pcr` documents from inside the same
//! address space as the organism they judge. That is a genuinely weaker
//! guarantee, not parity, and it is stated here rather than glossed: what
//! remains is that the objective is computed from bytes on disk that ADAM did
//! not write, and that the numbers come from [`Workspace::objective_bp`] rather
//! than from anything the organism says about itself.

use std::path::{Path, PathBuf};

use adam_eve::{FitnessError, FitnessProvider};
use adam_kernel::Genome;
use adam_organism::apply_list_amendment;
use adam_protocol::{
    BasisPoints, Component, Event, EventKind, EventSink, FitnessResult, Measurement, MutationKind,
    PayloadValue, Provenance, Recommendation, SignedBasisPoints, SubjectType, ValidationRequest,
};

use crate::connector::action_for;
use crate::workspace::Workspace;

/// Where measurements from this provider say they came from.
pub const ORIGIN: &str = "pcr:workspace/measure";

/// Convert a workspace objective into the protocol's unsigned basis points.
///
/// `objective_bp` is already basis points and cannot leave `0..=10_000`
/// (`processed * 10_000 / total`, with `processed <= total`), so this is a
/// change of type rather than of value. The clamp is defensive only, and
/// deliberately saturating: a negative objective would be a bug in the
/// workspace, and turning it into a huge positive by casting would hide that
/// bug inside a plausible-looking score.
fn objective_to_bp(objective_bp: i32) -> BasisPoints {
    BasisPoints::from_ratio(f64::from(objective_bp.clamp(0, 10_000)) / 10_000.0)
}

/// One side of a counterfactual, and the runs that produced it.
struct Side {
    objective_bp: i32,
    runs: u32,
    /// The `TaskRunCompleted` event ids, so the measurement can name the runs
    /// it rests on rather than merely asserting they happened.
    run_events: Vec<String>,
}

/// Scores a mutation by running the real workspace with and without it.
///
/// The provider owns the records every run starts from, and rebuilds the
/// workspace from them before each run. That reset is the whole basis of the
/// comparison: acting changes the workspace, so a second run against a used
/// workspace would be measuring the leftovers of the first.
pub struct RealTaskFitnessProvider<'a> {
    /// A directory the provider may create and destroy run workspaces under.
    root: PathBuf,
    /// The records every run starts from, as `(name, body)`.
    ///
    /// Held rather than re-read from somewhere so that both sides provably see
    /// the same input; a comparison whose two halves read different records is
    /// not a counterfactual.
    records: Vec<(String, String)>,
    /// The genome the organism has now — the baseline side of the comparison.
    baseline: Genome,
    sink: Option<&'a dyn EventSink>,
    correlation_id: String,
}

impl<'a> RealTaskFitnessProvider<'a> {
    pub fn new(
        root: impl Into<PathBuf>,
        records: Vec<(String, String)>,
        baseline: Genome,
        correlation_id: impl Into<String>,
    ) -> Self {
        Self {
            root: root.into(),
            records,
            baseline,
            sink: None,
            correlation_id: correlation_id.into(),
        }
    }

    /// Announce each run to an event sink.
    ///
    /// Optional because a measurement is still a measurement with nobody
    /// listening, and requiring a sink would force every test to supply one.
    pub fn announcing_to(mut self, sink: &'a dyn EventSink) -> Self {
        self.sink = Some(sink);
        self
    }

    /// Build the candidate genome by applying `request`'s mutation to the
    /// baseline.
    ///
    /// Only `AmendGenome` is supported, and unsupported kinds are an error
    /// rather than a silently-unchanged genome: measuring a mutation this
    /// provider did not actually apply would report "no difference" for a
    /// change that was never tried, which is the most misleading answer
    /// available.
    fn candidate_genome(&self, request: &ValidationRequest) -> Result<Genome, FitnessError> {
        let mutation = &request.mutation;
        if mutation.kind != MutationKind::AmendGenome {
            return Err(FitnessError::ProtocolError(format!(
                "the workspace can only measure AmendGenome mutations; this one is {:?}",
                mutation.kind
            )));
        }

        let mut candidate = self.baseline.clone();
        let changed = apply_list_amendment(
            &mut candidate,
            &mutation.target,
            mutation.current_value.as_deref().unwrap_or_default(),
            mutation.proposed_value.as_deref().unwrap_or_default(),
        )
        .map_err(|err| FitnessError::ProtocolError(err.to_string()))?;

        if !changed {
            return Err(FitnessError::ProtocolError(format!(
                "the mutation leaves the genome unchanged at {}; there is nothing to compare",
                mutation.target
            )));
        }
        Ok(candidate)
    }

    /// Check that the request is pinned to the genomes this provider actually
    /// ran.
    ///
    /// Without this the provider would happily measure its own idea of the
    /// change while ADAM held a different one. The hashes are the only thing
    /// that ties the two together, since a `ValidationRequest` carries hashes
    /// rather than genomes.
    fn check_pinned(
        &self,
        request: &ValidationRequest,
        candidate: &Genome,
    ) -> Result<(), FitnessError> {
        let before = self.baseline.content_hash();
        if before != request.genome_before_hash {
            return Err(FitnessError::Inauthentic {
                detail: format!(
                    "request is pinned to baseline genome {} but this provider holds {before}",
                    request.genome_before_hash
                ),
            });
        }
        let after = candidate.content_hash();
        if after != request.genome_after_hash {
            return Err(FitnessError::Inauthentic {
                detail: format!(
                    "applying the mutation yields genome {after}, but the request is pinned to {}",
                    request.genome_after_hash
                ),
            });
        }
        Ok(())
    }

    /// Run the workspace `trials` times under `genome`, from a clean start each
    /// time.
    fn run_side(
        &self,
        label: &str,
        genome: &Genome,
        trials: u32,
        mutation_id: &str,
    ) -> Result<Side, FitnessError> {
        let action = action_for(genome);
        let mut objective_bp = 0;
        let mut run_events = Vec::new();

        for trial in 0..trials {
            let dir = self.root.join(format!("{label}-{trial}"));
            let workspace = self.reset(&dir)?;
            workspace
                .act(action)
                .map_err(|err| FitnessError::ProtocolError(err.to_string()))?;
            objective_bp = workspace
                .objective_bp()
                .map_err(|err| FitnessError::ProtocolError(err.to_string()))?;

            if let Some(sink) = self.sink {
                let mut payload = std::collections::BTreeMap::new();
                payload.insert("side".to_string(), PayloadValue::from(label));
                payload.insert("trial".to_string(), PayloadValue::from(trial));
                payload.insert("action".to_string(), PayloadValue::from(action.as_str()));
                payload.insert(
                    "objective_bp".to_string(),
                    PayloadValue::Int(i64::from(objective_bp)),
                );
                let event = Event::new(
                    Component::Pcr,
                    EventKind::TaskRunCompleted,
                    mutation_id,
                    SubjectType::Mutation,
                    &self.correlation_id,
                    payload,
                    ORIGIN,
                );
                sink.emit(&event);
                run_events.push(event.id);
            }
        }

        Ok(Side {
            objective_bp,
            runs: trials,
            run_events,
        })
    }

    /// Rebuild a workspace from the provider's records.
    ///
    /// Removes the directory first. Creating over a used workspace would leave
    /// the previous run's `done/` entries in place and inflate the next run's
    /// objective — the exact contamination this reset exists to prevent.
    fn reset(&self, dir: &Path) -> Result<Workspace, FitnessError> {
        if dir.exists() {
            std::fs::remove_dir_all(dir).map_err(|err| {
                FitnessError::ProtocolError(format!(
                    "could not clear the run workspace at {}: {err}",
                    dir.display()
                ))
            })?;
        }
        let workspace =
            Workspace::create(dir).map_err(|err| FitnessError::ProtocolError(err.to_string()))?;
        let borrowed: Vec<(&str, &str)> = self
            .records
            .iter()
            .map(|(name, body)| (name.as_str(), body.as_str()))
            .collect();
        workspace
            .seed(&borrowed)
            .map_err(|err| FitnessError::ProtocolError(err.to_string()))?;
        Ok(workspace)
    }
}

impl FitnessProvider for RealTaskFitnessProvider<'_> {
    fn describe(&self) -> String {
        format!("real workspace at {}", self.root.display())
    }

    fn evaluator(&self) -> Component {
        Component::Pcr
    }

    fn measure(&self, request: &ValidationRequest) -> Result<FitnessResult, FitnessError> {
        let candidate_genome = self.candidate_genome(request)?;
        self.check_pinned(request, &candidate_genome)?;

        let mutation_id = request.mutation.id.as_str();
        let baseline = self.run_side("baseline", &self.baseline, request.trials, mutation_id)?;
        let candidate = self.run_side("candidate", &candidate_genome, request.trials, mutation_id)?;

        // Both sides ran `request.trials` times by construction. Asserting it
        // here anyway keeps the invariant local to the place that would break
        // it, rather than relying on a reader noticing the loop bounds.
        debug_assert_eq!(baseline.runs, candidate.runs);

        let delta = candidate.objective_bp - baseline.objective_bp;
        // Higher is better, and the objective is already the whole of what this
        // evaluator can see, so the recommendation is just the sign of the
        // change. No threshold is applied here on purpose: deciding what is
        // good enough is governance's job, and an evaluator that pre-judged it
        // would be making the acceptance decision twice.
        let recommendation = match delta {
            d if d > 0 => Recommendation::Approve,
            0 => Recommendation::NeedsReview,
            _ => Recommendation::Reject,
        };

        let mut derived_from = baseline.run_events.clone();
        derived_from.extend(candidate.run_events.iter().cloned());

        let mut provenance = Provenance::now(Component::Pcr, ORIGIN);
        provenance.derived_from = derived_from;
        provenance.evidence = vec![
            format!("baseline_objective_bp={}", baseline.objective_bp),
            format!("candidate_objective_bp={}", candidate.objective_bp),
        ];

        Ok(FitnessResult {
            cp: adam_protocol::CP.to_string(),
            doc_type: "FitnessResult".to_string(),
            id: uuid::Uuid::new_v4().to_string(),
            mutation_id: mutation_id.to_string(),
            seed: request.seed,
            // The workspace has no scenarios. Naming the environment keeps the
            // field honest and non-empty without pretending to a catalogue this
            // evaluator does not have.
            scenario_ids: vec!["workspace".to_string()],
            trials: request.trials,
            baseline: Measurement::objective(objective_to_bp(baseline.objective_bp), baseline.runs),
            candidate: Measurement::objective(
                objective_to_bp(candidate.objective_bp),
                candidate.runs,
            ),
            delta_bp: SignedBasisPoints::new(delta),
            recommendation,
            reason: format!(
                "workspace objective moved {}bp: {} processed {}bp, {} processed {}bp",
                delta,
                action_for(&self.baseline).as_str(),
                baseline.objective_bp,
                action_for(&candidate_genome).as_str(),
                candidate.objective_bp
            ),
            provenance,
        })
    }
}
