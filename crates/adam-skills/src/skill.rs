//! Skill lifecycle: discover -> create -> test -> evaluate -> promote -> evolve.
//!
//! A [`Skill`] is a first-class, versioned artifact. Every stage transition
//! is validated — a skill cannot be promoted without being evaluated, and
//! cannot be evaluated without test evidence — so the lifecycle itself
//! enforces "no direct self modification" at the type level.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

pub type SkillId = Uuid;

/// Where a skill currently sits in its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillStage {
    Discovered,
    Created,
    Tested,
    Evaluated,
    Promoted,
    Rejected,
}

/// A single test run recorded against a skill's current procedure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TestResult {
    pub passed: bool,
    pub note: String,
    pub recorded_at: DateTime<Utc>,
}

/// One prior procedure this skill evolved away from, kept for auditability.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Improvement {
    pub previous_version: String,
    pub previous_procedure: String,
    pub reason: String,
    pub recorded_at: DateTime<Utc>,
}

/// Errors raised when a lifecycle transition is attempted out of order.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SkillError {
    #[error("cannot {action} while skill {id} is in stage {stage:?}")]
    InvalidStage {
        id: SkillId,
        stage: SkillStage,
        action: &'static str,
    },
    #[error("skill {0} has no recorded test results to evaluate")]
    NoTestResults(SkillId),
}

/// A first-class, versioned skill artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Skill {
    pub id: SkillId,
    pub name: String,
    pub description: String,
    pub trigger_conditions: Vec<String>,
    pub procedure: String,
    pub dependencies: Vec<String>,
    pub confidence: f32,
    pub fitness_score: f32,
    pub failures: Vec<String>,
    pub improvements: Vec<Improvement>,
    pub test_results: Vec<TestResult>,
    pub stage: SkillStage,
    pub version: String,
    pub created_at: DateTime<Utc>,
}

impl Skill {
    /// **discover**: register the existence of a needed capability before
    /// any procedure has been written for it.
    pub fn discover(
        name: impl Into<String>,
        description: impl Into<String>,
        trigger_conditions: Vec<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            description: description.into(),
            trigger_conditions,
            procedure: String::new(),
            dependencies: Vec::new(),
            confidence: 0.0,
            fitness_score: 0.0,
            failures: Vec::new(),
            improvements: Vec::new(),
            test_results: Vec::new(),
            stage: SkillStage::Discovered,
            version: "0.1".to_string(),
            created_at: Utc::now(),
        }
    }

    /// **create**: attach a concrete execution procedure and dependency
    /// list to a discovered skill.
    pub fn define_procedure(
        &mut self,
        procedure: impl Into<String>,
        dependencies: Vec<String>,
    ) -> Result<(), SkillError> {
        self.require_stage(SkillStage::Discovered, "define a procedure for")?;
        self.procedure = procedure.into();
        self.dependencies = dependencies;
        self.stage = SkillStage::Created;
        Ok(())
    }

    /// **test**: record one sandboxed execution outcome against the
    /// current procedure. A skill may accumulate any number of test runs
    /// while in `Created` or `Tested`.
    pub fn record_test(&mut self, passed: bool, note: impl Into<String>) -> Result<(), SkillError> {
        if self.stage != SkillStage::Created && self.stage != SkillStage::Tested {
            return Err(SkillError::InvalidStage {
                id: self.id,
                stage: self.stage,
                action: "record a test result for",
            });
        }
        self.test_results.push(TestResult {
            passed,
            note: note.into(),
            recorded_at: Utc::now(),
        });
        self.stage = SkillStage::Tested;
        Ok(())
    }

    /// **evaluate**: compute a fitness score from recorded test evidence
    /// (pass rate) and move to `Evaluated` if it clears `threshold`,
    /// otherwise `Rejected` with a failure recorded.
    pub fn evaluate(&mut self, threshold: f32) -> Result<(), SkillError> {
        self.require_stage(SkillStage::Tested, "evaluate")?;
        if self.test_results.is_empty() {
            return Err(SkillError::NoTestResults(self.id));
        }

        let passed = self.test_results.iter().filter(|r| r.passed).count() as f32;
        let total = self.test_results.len() as f32;
        self.fitness_score = passed / total;
        self.confidence = self.fitness_score;

        if self.fitness_score >= threshold {
            self.stage = SkillStage::Evaluated;
        } else {
            self.failures.push(format!(
                "evaluation failed: fitness {:.2} below threshold {:.2}",
                self.fitness_score, threshold
            ));
            self.stage = SkillStage::Rejected;
        }
        Ok(())
    }

    /// **promote**: graduate an evaluated skill into active use.
    pub fn promote(&mut self) -> Result<(), SkillError> {
        self.require_stage(SkillStage::Evaluated, "promote")?;
        self.stage = SkillStage::Promoted;
        Ok(())
    }

    /// **evolve**: replace a promoted skill's procedure with an improved
    /// one, bumping its version, archiving the prior procedure as an
    /// [`Improvement`], and sending it back to `Created` so the new
    /// procedure must earn re-evaluation before being trusted again.
    pub fn evolve(
        &mut self,
        reason: impl Into<String>,
        new_procedure: impl Into<String>,
    ) -> Result<(), SkillError> {
        self.require_stage(SkillStage::Promoted, "evolve")?;

        self.improvements.push(Improvement {
            previous_version: self.version.clone(),
            previous_procedure: self.procedure.clone(),
            reason: reason.into(),
            recorded_at: Utc::now(),
        });

        self.version = bump_minor(&self.version);
        self.procedure = new_procedure.into();
        self.test_results.clear();
        self.fitness_score = 0.0;
        self.stage = SkillStage::Created;
        Ok(())
    }

    fn require_stage(&self, expected: SkillStage, action: &'static str) -> Result<(), SkillError> {
        if self.stage != expected {
            return Err(SkillError::InvalidStage {
                id: self.id,
                stage: self.stage,
                action,
            });
        }
        Ok(())
    }
}

fn bump_minor(label: &str) -> String {
    if let Some((major, minor)) = label.split_once('.') {
        if let Ok(minor_num) = minor.parse::<u32>() {
            return format!("{}.{}", major, minor_num + 1);
        }
    }
    format!("{}.1", label)
}
