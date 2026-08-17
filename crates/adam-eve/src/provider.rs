//! Where a fitness measurement comes from.
//!
//! [`FitnessProvider`] is the seam between ADAM and EVE. ADAM builds a
//! [`ValidationRequest`], hands it to a provider, and receives a
//! [`FitnessResult`]. It never links EVE, never imports an EVE type, and never
//! learns how the measurement was produced.
//!
//! [`Cp1Subprocess`] is the production implementation: it spawns EVE's CP/1
//! endpoint, writes one request line and reads one response line. Chosen over
//! HTTP because it needs no listener, no port allocation and no service
//! discovery for the common single-host case, and because a subprocess is a
//! real isolation boundary for a component whose job is running scenarios.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::Duration;

use adam_protocol::{
    Component, EnvelopeError, FitnessResult, SignedEnvelope, ValidationRequest,
};

/// Why a fitness measurement could not be obtained.
///
/// Every variant is a reason ADAM must *not* treat a proposal as validated.
/// There is no variant meaning "measurement failed, proceed anyway" — that
/// would restore precisely the property this module removes.
#[derive(Debug)]
pub enum FitnessError {
    /// The provider process could not be started.
    Spawn {
        command: String,
        source: std::io::Error,
    },
    /// Writing the request or reading the response failed.
    Transport(std::io::Error),
    /// The provider closed without answering.
    NoResponse { command: String },
    /// The provider answered, but the envelope did not verify.
    Envelope(EnvelopeError),
    /// The provider reported it could not process the request at all.
    ProtocolError(String),
    /// The response verified as an envelope but is not a `FitnessResult`.
    NotAFitnessResult(String),
    /// The response is a `FitnessResult`, but not one ADAM may rely on: wrong
    /// author, wrong mutation, or an asymmetric comparison.
    Inauthentic { detail: String },
    /// The provider exceeded its time budget.
    Timeout { command: String, after: Duration },
}

impl std::fmt::Display for FitnessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FitnessError::Spawn { command, source } => {
                write!(
                    f,
                    "could not start the fitness provider `{command}`: {source}"
                )
            }
            FitnessError::Transport(source) => {
                write!(f, "fitness provider transport failed: {source}")
            }
            FitnessError::NoResponse { command } => {
                write!(f, "fitness provider `{command}` closed without answering")
            }
            FitnessError::Envelope(err) => write!(f, "fitness response did not verify: {err}"),
            FitnessError::ProtocolError(detail) => {
                write!(f, "fitness provider refused the request: {detail}")
            }
            FitnessError::NotAFitnessResult(kind) => {
                write!(f, "expected a FitnessResult, received a {kind}")
            }
            FitnessError::Inauthentic { detail } => {
                write!(f, "fitness result cannot be relied on: {detail}")
            }
            FitnessError::Timeout { command, after } => write!(
                f,
                "fitness provider `{command}` did not answer within {after:?}"
            ),
        }
    }
}

impl std::error::Error for FitnessError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FitnessError::Spawn { source, .. } => Some(source),
            FitnessError::Transport(source) => Some(source),
            _ => None,
        }
    }
}

/// Something that can measure a mutation.
///
/// Implementations must return a result whose
/// [`FitnessResult::is_authentic`] holds for the request's mutation, or an
/// error. Returning an inauthentic result is a contract violation, and
/// [`measure_and_verify`] enforces it for every provider rather than trusting
/// each one to police itself.
pub trait FitnessProvider: Send + Sync {
    /// The provider's identity, for error messages and audit entries.
    fn describe(&self) -> String;

    /// Which component this provider's measurements are authored by.
    ///
    /// Declared by the provider and then *checked against the document it
    /// returns*, which is what makes it more than a label: a provider claiming
    /// to be EVE while returning a PCR-authored result fails verification, and
    /// so does one that names a component which may not author evidence at all.
    fn evaluator(&self) -> Component;

    /// Measure `request`, returning the evaluator's verdict.
    fn measure(&self, request: &ValidationRequest) -> Result<FitnessResult, FitnessError>;
}

/// Call `provider`, then check the result is one ADAM may rely on.
///
/// The check is applied here, once, rather than inside each provider: a
/// provider that skipped it would silently reintroduce self-scored evidence,
/// and that is exactly the failure mode CP/1 exists to close.
///
/// This is also the only place that knows both who was asked and who answered,
/// so it is where those two have to be compared. Neither the provider nor the
/// document can make that comparison alone.
pub fn measure_and_verify(
    provider: &dyn FitnessProvider,
    request: &ValidationRequest,
) -> Result<FitnessResult, FitnessError> {
    let result = provider.measure(request)?;
    match result.authenticity_failure(&request.mutation.id, provider.evaluator()) {
        None => Ok(result),
        Some(detail) => Err(FitnessError::Inauthentic {
            detail: format!("{detail} (provider: {})", provider.describe()),
        }),
    }
}

/// Measures fitness by spawning EVE's CP/1 endpoint as a subprocess.
///
/// The default command is `eve-cp1`, the binary EVE publishes for exactly this
/// purpose. One process is spawned per measurement: a long-lived child would
/// need liveness tracking, restart logic and back-pressure handling to save a
/// process spawn against a workload that already runs browser simulations for
/// seconds at a time.
pub struct Cp1Subprocess {
    command: String,
    args: Vec<String>,
    fleet_key: Option<Vec<u8>>,
    timeout: Duration,
}

impl Cp1Subprocess {
    /// A provider running `eve-cp1` from `PATH`.
    pub fn new() -> Self {
        Self {
            command: "eve-cp1".to_string(),
            args: Vec::new(),
            fleet_key: None,
            timeout: Duration::from_secs(600),
        }
    }

    /// A provider running an explicit command, e.g. `node` with a script path.
    pub fn command(command: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            command: command.into(),
            args,
            ..Self::new()
        }
    }

    /// Sign outgoing envelopes and require signatures on incoming ones.
    ///
    /// Unnecessary over a plain subprocess boundary, where the parent already
    /// controls the child. Meaningful when the command is a shim onto a remote
    /// EVE, where the request crosses a network the parent does not own.
    pub fn with_fleet_key(mut self, key: impl Into<Vec<u8>>) -> Self {
        self.fleet_key = Some(key.into());
        self
    }

    /// How long to wait for a measurement.
    ///
    /// Generous by default: a real measurement runs a scenario suite several
    /// times over and legitimately takes minutes. A tight timeout here would
    /// show up as spurious validation failures under load, which read as the
    /// mutation being bad rather than the budget being wrong.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

impl Default for Cp1Subprocess {
    fn default() -> Self {
        Self::new()
    }
}

impl FitnessProvider for Cp1Subprocess {
    fn describe(&self) -> String {
        if self.args.is_empty() {
            self.command.clone()
        } else {
            format!("{} {}", self.command, self.args.join(" "))
        }
    }

    fn evaluator(&self) -> Component {
        Component::Eve
    }

    fn measure(&self, request: &ValidationRequest) -> Result<FitnessResult, FitnessError> {
        let sealed = request
            .seal()
            .expect("ValidationRequest contains no floats or nulls by construction");
        let envelope = SignedEnvelope::seal(&sealed, self.fleet_key.as_deref())
            .expect("a sealed request always re-encodes");

        let mut child = Command::new(&self.command)
            .args(&self.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // Inherit stderr so EVE's diagnostics reach the operator's logs
            // rather than being swallowed. stdout carries the protocol and
            // must not be polluted; stderr is where EVE puts everything else.
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|source| FitnessError::Spawn {
                command: self.describe(),
                source,
            })?;

        {
            let stdin = child
                .stdin
                .as_mut()
                .expect("stdin was configured as a pipe");
            if let Err(err) = writeln!(stdin, "{}", envelope.to_line()) {
                let _ = child.wait();
                // A broken pipe here means the provider exited before it could
                // read the request — which is the same condition as exiting
                // before answering it, and deserves the same diagnosis.
                // Reporting it as a generic transport error would make one
                // situation surface as two different errors depending on
                // whether the write or the child's exit won the race.
                return Err(if err.kind() == std::io::ErrorKind::BrokenPipe {
                    FitnessError::NoResponse {
                        command: self.describe(),
                    }
                } else {
                    FitnessError::Transport(err)
                });
            }
            // Closing stdin is what tells the endpoint no more requests are
            // coming, so it can exit after answering this one.
        }
        child.stdin.take();

        let stdout = child
            .stdout
            .take()
            .expect("stdout was configured as a pipe");

        // Read on a worker thread so the timeout is enforceable. `read_line`
        // has no deadline, so a hung EVE would otherwise block the organism's
        // lifecycle indefinitely — and an organism that cannot make progress
        // because a measurement never returned is worse than one that reports
        // the measurement failed.
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            let outcome = loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) => break Ok(None),
                    Ok(_) if line.trim().is_empty() => continue,
                    Ok(_) => break Ok(Some(line.trim().to_string())),
                    Err(err) => break Err(err),
                }
            };
            // A send failure means the parent already timed out and stopped
            // listening; there is nothing left to report to.
            let _ = sender.send(outcome);
        });

        let response = match receiver.recv_timeout(self.timeout) {
            Ok(Ok(Some(line))) => line,
            Ok(Ok(None)) => {
                let _ = child.wait();
                return Err(FitnessError::NoResponse {
                    command: self.describe(),
                });
            }
            Ok(Err(err)) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(FitnessError::Transport(err));
            }
            Err(_) => {
                // Kill and reap: leaving a hung child running would leak a
                // process (and a browser) per timed-out measurement.
                let _ = child.kill();
                let _ = child.wait();
                return Err(FitnessError::Timeout {
                    command: self.describe(),
                    after: self.timeout,
                });
            }
        };

        let _ = child.wait();
        self.decode(&response)
    }
}

impl Cp1Subprocess {
    fn decode(&self, line: &str) -> Result<FitnessResult, FitnessError> {
        // A refusal is framed as a bare ProtocolError, not an envelope, because
        // a request that could not be understood has no document to wrap.
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(line) {
            if value.get("type").and_then(|t| t.as_str()) == Some("ProtocolError") {
                return Err(FitnessError::ProtocolError(
                    value
                        .get("detail")
                        .and_then(|d| d.as_str())
                        .unwrap_or("no detail supplied")
                        .to_string(),
                ));
            }
        }

        let envelope = SignedEnvelope::from_line(line).map_err(FitnessError::Envelope)?;
        let document = envelope
            .open(self.fleet_key.as_deref())
            .map_err(FitnessError::Envelope)?;

        let doc_type = document
            .get("type")
            .and_then(|t| t.as_str())
            .unwrap_or("document with no type");
        if doc_type != "FitnessResult" {
            return Err(FitnessError::NotAFitnessResult(doc_type.to_string()));
        }

        serde_json::from_value(document).map_err(|err| {
            FitnessError::NotAFitnessResult(format!("malformed FitnessResult: {err}"))
        })
    }
}

/// A provider that returns a prepared result. For tests only.
///
/// Exposed rather than hidden behind `#[cfg(test)]` so downstream crates —
/// `adam-organism`'s tests in particular — can exercise the acceptance gate
/// without spawning a Node process. It is named to make its use obvious in a
/// stack trace or an audit entry.
pub struct StubProvider {
    result: Result<FitnessResult, String>,
    evaluator: Component,
}

impl StubProvider {
    pub fn returning(result: FitnessResult) -> Self {
        Self {
            result: Ok(result),
            evaluator: Component::Eve,
        }
    }

    pub fn failing(detail: impl Into<String>) -> Self {
        Self {
            result: Err(detail.into()),
            evaluator: Component::Eve,
        }
    }

    /// Stand in for an evaluator other than EVE.
    ///
    /// Defaulting to EVE keeps every existing test unchanged; a test that cares
    /// which evaluator was asked says so here. Note that this sets who the
    /// request was sent *to*, not who authored the prepared result — the two
    /// disagreeing is precisely what [`measure_and_verify`] must catch.
    pub fn as_evaluator(mut self, evaluator: Component) -> Self {
        self.evaluator = evaluator;
        self
    }
}

impl FitnessProvider for StubProvider {
    fn describe(&self) -> String {
        "stub (test double)".to_string()
    }

    fn evaluator(&self) -> Component {
        self.evaluator
    }

    fn measure(&self, _request: &ValidationRequest) -> Result<FitnessResult, FitnessError> {
        self.result.clone().map_err(FitnessError::ProtocolError)
    }
}

/// Verify a raw response line the way [`Cp1Subprocess`] does, without spawning
/// anything.
///
/// The decoding path carries the checks that make a measurement trustworthy, so
/// it needs to be testable on its own. Exposed for that reason and used by the
/// crate's tests.
pub fn decode_response(
    line: &str,
    fleet_key: Option<&[u8]>,
) -> Result<FitnessResult, FitnessError> {
    let provider = Cp1Subprocess {
        command: "test".to_string(),
        args: Vec::new(),
        fleet_key: fleet_key.map(<[u8]>::to_vec),
        timeout: Duration::from_secs(1),
    };
    provider.decode(line)
}

#[cfg(test)]
mod tests {
    use super::*;
    use adam_protocol::{BasisPoints, Component, Provenance};

    fn request() -> ValidationRequest {
        use adam_protocol::{Mutation, MutationKind, MutationStatus};
        ValidationRequest::new(
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
                MutationStatus::Validating,
                Provenance::now(Component::Adam, "adam:evolution/proposal"),
            ),
            adam_protocol::GenomePair::new("a".repeat(64), "b".repeat(64)),
            adam_protocol::MeasurementPlan {
                scenario_ids: vec!["excellent".to_string()],
                seed: 1337,
                trials: 3,
            },
            Provenance::now(Component::Adam, "adam:evolution/validate"),
        )
    }

    /// A shell that never answers, for exercising the timeout path.
    fn silent_command() -> Cp1Subprocess {
        Cp1Subprocess::command("sh", vec!["-c".to_string(), "sleep 30".to_string()])
            .with_timeout(Duration::from_millis(250))
    }

    #[test]
    fn a_provider_that_never_answers_times_out_instead_of_hanging() {
        let start = std::time::Instant::now();
        let err = silent_command().measure(&request()).unwrap_err();
        assert!(
            matches!(err, FitnessError::Timeout { .. }),
            "expected a timeout, got {err:?}"
        );
        // The point of the timeout is that it returns promptly, not eventually.
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "timeout took {:?}, which means it did not fire",
            start.elapsed()
        );
    }

    #[test]
    fn a_provider_that_closes_without_answering_is_reported_as_such() {
        // Distinct from a timeout: the child exited, it just said nothing.
        //
        // `true` exits immediately, so this races the request write against the
        // child's exit — the write may succeed and the read then see EOF, or
        // the write may hit a broken pipe. Both are the same condition, and
        // both must produce the same error, or the diagnosis would depend on
        // scheduling. (CI caught exactly this: it lost the race that a
        // developer machine reliably won.)
        let provider = Cp1Subprocess::command("true", vec![]);
        for attempt in 0..20 {
            let err = provider.measure(&request()).unwrap_err();
            assert!(
                matches!(err, FitnessError::NoResponse { .. }),
                "attempt {attempt}: expected NoResponse, got {err:?}"
            );
        }
    }

    #[test]
    fn a_missing_binary_is_reported_as_a_spawn_failure() {
        let provider = Cp1Subprocess::command("definitely-not-a-real-binary-xyz", vec![]);
        let err = provider.measure(&request()).unwrap_err();
        assert!(matches!(err, FitnessError::Spawn { .. }));
    }

    #[test]
    fn the_provider_describes_itself_with_its_arguments() {
        assert_eq!(Cp1Subprocess::new().describe(), "eve-cp1");
        assert_eq!(
            Cp1Subprocess::command("node", vec!["bin/eve-cp1.js".to_string()]).describe(),
            "node bin/eve-cp1.js"
        );
    }

    #[test]
    fn a_garbage_response_line_is_refused() {
        let provider = Cp1Subprocess::command(
            "sh",
            vec!["-c".to_string(), "echo not-an-envelope".to_string()],
        );
        assert!(provider.measure(&request()).is_err());
    }
}
