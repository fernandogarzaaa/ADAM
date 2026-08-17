//! One small, real, boring environment.
//!
//! # Why a filesystem
//!
//! The experiment needs somewhere the organism can be wrong about the world.
//! A simulated environment cannot provide that — whoever writes the simulator
//! decides what is there, so the organism's failure is authored rather than
//! discovered. A directory of files is genuinely external: the bytes exist
//! whether ADAM believes in them or not, the failure comes from content ADAM
//! did not write, and acting changes what a later `observe` returns.
//!
//! It is also deliberately dull. Directory listings are sorted, nothing
//! consults a clock or a random number generator, and every operation is
//! ordinary file IO — so a difference between two runs is a real difference
//! rather than environment noise pretending to be one.
//!
//! # What it is not
//!
//! It is synthetic. Nobody's actual work depends on these files, and that
//! limitation is stated rather than dressed up: what the environment can show
//! is that a closed loop runs against something outside the organism, not that
//! the loop is useful yet.

use std::path::{Path, PathBuf};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum WorkspaceError {
    #[error("workspace io error at {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("record {name} is not valid utf-8")]
    NotUtf8 { name: String },
}

fn io(path: &Path) -> impl Fn(std::io::Error) -> WorkspaceError + '_ {
    move |source| WorkspaceError::Io {
        path: path.display().to_string(),
        source,
    }
}

/// What the organism can do here.
///
/// Two actions, differing in exactly one respect: whether the records are
/// checked before any of them are processed. That is the point — the
/// difference between them is what an adaptation would have to discover.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Process records in order, stopping at the first malformed one.
    ///
    /// The naive action, and not a strawman: stopping on unexpected input is
    /// what unguarded code does, and everything after the bad record stays
    /// unprocessed as a result.
    ProcessAll,
    /// Check every record first, set the malformed ones aside, then process
    /// the rest.
    VerifyThenProcess,
}

impl Action {
    pub fn as_str(self) -> &'static str {
        match self {
            Action::ProcessAll => "process_all",
            Action::VerifyThenProcess => "verify_then_process",
        }
    }
}

/// What the organism can see. Sorted, so two identical states look identical.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceState {
    pub pending: Vec<String>,
    pub processed: Vec<String>,
    pub quarantined: Vec<String>,
}

impl WorkspaceState {
    pub fn total(&self) -> usize {
        self.pending.len() + self.processed.len() + self.quarantined.len()
    }
}

/// What happened when the organism acted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionOutcome {
    pub action: Action,
    pub processed: usize,
    pub quarantined: usize,
    /// The record the action stopped on, if it stopped.
    pub failed_on: Option<String>,
    /// What was wrong with it, in the words the environment would use.
    pub error: Option<String>,
}

impl ActionOutcome {
    pub fn succeeded(&self) -> bool {
        self.failed_on.is_none()
    }
}

/// A directory of records the organism processes.
pub struct Workspace {
    root: PathBuf,
}

impl Workspace {
    /// Create the directory layout. Existing content is left alone.
    pub fn create(root: impl Into<PathBuf>) -> Result<Self, WorkspaceError> {
        let root = root.into();
        for dir in [INBOX, DONE, QUARANTINE] {
            let path = root.join(dir);
            std::fs::create_dir_all(&path).map_err(io(&path))?;
        }
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Put records in the inbox.
    ///
    /// Called by whoever sets up the experiment, never by the organism — the
    /// content the organism trips over must come from outside it, or the
    /// failure it learns from is one it arranged for itself.
    pub fn seed(&self, records: &[(&str, &str)]) -> Result<(), WorkspaceError> {
        for (name, body) in records {
            let path = self.root.join(INBOX).join(name);
            std::fs::write(&path, body).map_err(io(&path))?;
        }
        Ok(())
    }

    /// Look at the workspace without changing it.
    pub fn observe(&self) -> Result<WorkspaceState, WorkspaceError> {
        Ok(WorkspaceState {
            pending: self.list(INBOX)?,
            processed: self.list(DONE)?,
            quarantined: self.list(QUARANTINE)?,
        })
    }

    /// The objective, in basis points: how much of the work is actually done.
    ///
    /// Quarantined records count against it. Setting a record aside beats
    /// stopping on it, but it is not the same as processing it, and a score
    /// that pretended otherwise would reward avoiding the work.
    pub fn objective_bp(&self) -> Result<i32, WorkspaceError> {
        let state = self.observe()?;
        let total = state.total();
        if total == 0 {
            return Ok(0);
        }
        Ok((state.processed.len() * 10_000 / total) as i32)
    }

    /// Do something, and change what a later `observe` will return.
    pub fn act(&self, action: Action) -> Result<ActionOutcome, WorkspaceError> {
        match action {
            Action::ProcessAll => self.process_all(),
            Action::VerifyThenProcess => self.verify_then_process(),
        }
    }

    fn process_all(&self) -> Result<ActionOutcome, WorkspaceError> {
        let mut processed = 0;
        for name in self.list(INBOX)? {
            match self.check(&name)? {
                Ok(()) => {
                    self.move_to(&name, DONE)?;
                    processed += 1;
                }
                // Stop, leaving the rest of the inbox untouched. The organism
                // discovers this by observing afterwards, not by being told.
                Err(problem) => {
                    return Ok(ActionOutcome {
                        action: Action::ProcessAll,
                        processed,
                        quarantined: 0,
                        failed_on: Some(name),
                        error: Some(problem),
                    })
                }
            }
        }
        Ok(ActionOutcome {
            action: Action::ProcessAll,
            processed,
            quarantined: 0,
            failed_on: None,
            error: None,
        })
    }

    fn verify_then_process(&self) -> Result<ActionOutcome, WorkspaceError> {
        let mut quarantined = 0;
        for name in self.list(INBOX)? {
            if self.check(&name)?.is_err() {
                self.move_to(&name, QUARANTINE)?;
                quarantined += 1;
            }
        }
        let mut processed = 0;
        for name in self.list(INBOX)? {
            self.move_to(&name, DONE)?;
            processed += 1;
        }
        Ok(ActionOutcome {
            action: Action::VerifyThenProcess,
            processed,
            quarantined,
            failed_on: None,
            error: None,
        })
    }

    /// A record is well-formed when every non-empty line is `key=value`.
    fn check(&self, name: &str) -> Result<Result<(), String>, WorkspaceError> {
        let path = self.root.join(INBOX).join(name);
        let bytes = std::fs::read(&path).map_err(io(&path))?;
        let text = String::from_utf8(bytes).map_err(|_| WorkspaceError::NotUtf8 {
            name: name.to_string(),
        })?;
        for (index, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if !line.contains('=') {
                return Ok(Err(format!(
                    "line {} of {name} is not key=value: {line:?}",
                    index + 1
                )));
            }
        }
        Ok(Ok(()))
    }

    fn move_to(&self, name: &str, dir: &str) -> Result<(), WorkspaceError> {
        let from = self.root.join(INBOX).join(name);
        let to = self.root.join(dir).join(name);
        std::fs::rename(&from, &to).map_err(io(&from))
    }

    /// Sorted, so the environment reports the same state the same way twice.
    fn list(&self, dir: &str) -> Result<Vec<String>, WorkspaceError> {
        let path = self.root.join(dir);
        let mut names = Vec::new();
        for entry in std::fs::read_dir(&path).map_err(io(&path))? {
            let entry = entry.map_err(io(&path))?;
            if entry.path().is_file() {
                names.push(entry.file_name().to_string_lossy().into_owned());
            }
        }
        names.sort();
        Ok(names)
    }
}

const INBOX: &str = "inbox";
const DONE: &str = "done";
const QUARANTINE: &str = "quarantine";

#[cfg(test)]
mod tests {
    use super::*;

    struct Temp(PathBuf);

    impl Temp {
        fn new(label: &str) -> Self {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos();
            let dir = std::env::temp_dir().join(format!("adam-ws-{label}-{nonce}"));
            std::fs::create_dir_all(&dir).expect("mkdir");
            Self(dir)
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Four records, the third malformed. The position matters: the naive
    /// action must get some work done before it fails, so the failure is
    /// partial rather than total.
    const RECORDS: &[(&str, &str)] = &[
        ("01-alpha.rec", "name=alpha\ncount=3\n"),
        ("02-bravo.rec", "name=bravo\ncount=1\n"),
        (
            "03-charlie.rec",
            "name=charlie\nthis line has no separator\n",
        ),
        ("04-delta.rec", "name=delta\ncount=7\n"),
    ];

    fn seeded(label: &str) -> (Temp, Workspace) {
        let temp = Temp::new(label);
        let ws = Workspace::create(&temp.0).expect("create");
        ws.seed(RECORDS).expect("seed");
        (temp, ws)
    }

    #[test]
    fn a_fresh_workspace_has_everything_pending() {
        let (_t, ws) = seeded("fresh");
        let state = ws.observe().expect("observe");
        assert_eq!(state.pending.len(), 4);
        assert!(state.processed.is_empty());
        assert_eq!(ws.objective_bp().expect("objective"), 0);
    }

    #[test]
    fn the_naive_action_stops_on_content_it_did_not_expect() {
        let (_t, ws) = seeded("naive");
        let outcome = ws.act(Action::ProcessAll).expect("act");

        assert!(!outcome.succeeded());
        assert_eq!(outcome.processed, 2);
        assert_eq!(outcome.failed_on.as_deref(), Some("03-charlie.rec"));
        assert!(outcome.error.expect("error").contains("not key=value"));

        let state = ws.observe().expect("observe");
        assert_eq!(state.processed.len(), 2);
        assert_eq!(state.pending.len(), 2, "the rest is still waiting");
        assert_eq!(ws.objective_bp().expect("objective"), 5000);
    }

    #[test]
    fn checking_first_gets_more_of_the_work_done() {
        let (_t, ws) = seeded("careful");
        let outcome = ws.act(Action::VerifyThenProcess).expect("act");

        assert!(outcome.succeeded());
        assert_eq!(outcome.processed, 3);
        assert_eq!(outcome.quarantined, 1);
        assert_eq!(ws.objective_bp().expect("objective"), 7500);
    }

    #[test]
    fn acting_changes_what_a_later_observation_says() {
        let (_t, ws) = seeded("changes");
        let before = ws.observe().expect("observe");
        ws.act(Action::ProcessAll).expect("act");
        let after = ws.observe().expect("observe");

        assert_ne!(
            before, after,
            "an action that left observations unchanged would make the loop a mirror"
        );
        assert!(after.pending.len() < before.pending.len());
    }

    #[test]
    fn the_same_action_on_the_same_state_gives_the_same_result() {
        let (_a, first) = seeded("determinism-a");
        let (_b, second) = seeded("determinism-b");
        assert_eq!(
            first.act(Action::ProcessAll).expect("act"),
            second.act(Action::ProcessAll).expect("act")
        );
        assert_eq!(
            first.observe().expect("observe"),
            second.observe().expect("observe")
        );
    }

    #[test]
    fn an_empty_workspace_scores_zero_rather_than_dividing_by_zero() {
        let temp = Temp::new("empty");
        let ws = Workspace::create(&temp.0).expect("create");
        assert_eq!(ws.objective_bp().expect("objective"), 0);
        let outcome = ws.act(Action::ProcessAll).expect("act");
        assert!(outcome.succeeded());
        assert_eq!(outcome.processed, 0);
    }
}
