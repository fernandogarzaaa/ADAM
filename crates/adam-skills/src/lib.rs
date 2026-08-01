//! ADAM Skill Evolution Engine.
//!
//! Skills are first-class, versioned artifacts that move through a strict
//! lifecycle — discover, create, test, evaluate, promote, evolve — with
//! every transition validated so a skill can never be trusted (promoted)
//! without evidence (tests) and never silently mutated (evolve always
//! archives the prior procedure and forces re-evaluation).

mod registry;
mod skill;

pub use registry::SkillRegistry;
pub use skill::{Improvement, Skill, SkillError, SkillId, SkillStage, TestResult};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_lifecycle_from_discovery_to_promotion() {
        let mut skill = Skill::discover(
            "rust-debugging",
            "Debug Rust compilation failures",
            vec!["rust_build_failed".to_string()],
        );
        assert_eq!(skill.stage, SkillStage::Discovered);

        skill
            .define_procedure(
                "1. Read the compiler error. 2. Check Cargo.toml dependencies. 3. Run cargo check.",
                vec!["cargo".to_string()],
            )
            .unwrap();
        assert_eq!(skill.stage, SkillStage::Created);

        skill.record_test(true, "resolved missing dependency case").unwrap();
        skill.record_test(true, "resolved borrow checker case").unwrap();
        skill.record_test(false, "missed a lifetime elision case").unwrap();
        assert_eq!(skill.stage, SkillStage::Tested);
        assert_eq!(skill.test_results.len(), 3);

        skill.evaluate(0.5).unwrap();
        assert_eq!(skill.stage, SkillStage::Evaluated);
        assert!((skill.fitness_score - (2.0 / 3.0)).abs() < 1e-6);

        skill.promote().unwrap();
        assert_eq!(skill.stage, SkillStage::Promoted);
    }

    #[test]
    fn evaluation_below_threshold_rejects_the_skill() {
        let mut skill = Skill::discover("flaky-skill", "Unreliable procedure", vec![]);
        skill.define_procedure("do the thing", vec![]).unwrap();
        skill.record_test(false, "failed").unwrap();
        skill.record_test(false, "failed again").unwrap();

        skill.evaluate(0.5).unwrap();
        assert_eq!(skill.stage, SkillStage::Rejected);
        assert_eq!(skill.fitness_score, 0.0);
        assert_eq!(skill.failures.len(), 1);
    }

    #[test]
    fn evaluate_without_test_results_errors() {
        let mut skill = Skill::discover("untested", "No tests yet", vec![]);
        skill.define_procedure("do the thing", vec![]).unwrap();
        skill.stage = SkillStage::Tested; // simulate a skill that reached Tested with no runs
        assert_eq!(
            skill.evaluate(0.5),
            Err(SkillError::NoTestResults(skill.id))
        );
    }

    #[test]
    fn promote_requires_evaluation_first() {
        let mut skill = Skill::discover("impatient", "Tries to skip evaluation", vec![]);
        skill.define_procedure("do the thing", vec![]).unwrap();
        let err = skill.promote().unwrap_err();
        assert_eq!(
            err,
            SkillError::InvalidStage {
                id: skill.id,
                stage: SkillStage::Created,
                action: "promote",
            }
        );
    }

    #[test]
    fn evolve_archives_prior_procedure_and_forces_retest() {
        let mut skill = Skill::discover("evolving-skill", "Gets better over time", vec![]);
        skill.define_procedure("v1 procedure", vec![]).unwrap();
        skill.record_test(true, "passed").unwrap();
        skill.evaluate(0.5).unwrap();
        skill.promote().unwrap();

        skill
            .evolve("v1 missed edge cases", "v2 procedure with edge case handling")
            .unwrap();

        assert_eq!(skill.stage, SkillStage::Created);
        assert_eq!(skill.version, "0.2");
        assert_eq!(skill.procedure, "v2 procedure with edge case handling");
        assert!(skill.test_results.is_empty());
        assert_eq!(skill.improvements.len(), 1);
        assert_eq!(skill.improvements[0].previous_procedure, "v1 procedure");
        assert_eq!(skill.improvements[0].previous_version, "0.1");
    }

    #[test]
    fn evolve_requires_promotion_first() {
        let mut skill = Skill::discover("hasty-skill", "Tries to evolve too early", vec![]);
        skill.define_procedure("v1 procedure", vec![]).unwrap();
        let err = skill.evolve("premature", "v2").unwrap_err();
        assert_eq!(
            err,
            SkillError::InvalidStage {
                id: skill.id,
                stage: SkillStage::Created,
                action: "evolve",
            }
        );
    }

    #[test]
    fn registry_finds_promoted_skills_by_trigger_condition() {
        let mut registry = SkillRegistry::new();

        let mut skill = Skill::discover(
            "rust-debugging",
            "Debug Rust compilation failures",
            vec!["rust_build_failed".to_string()],
        );
        skill.define_procedure("check deps", vec![]).unwrap();
        skill.record_test(true, "ok").unwrap();
        skill.evaluate(0.5).unwrap();
        skill.promote().unwrap();
        let id = registry.upsert(skill);

        let discovered_only = Skill::discover("go-debugging", "Not ready yet", vec!["go_build_failed".to_string()]);
        registry.upsert(discovered_only);

        assert_eq!(registry.len(), 2);
        let applicable = registry.find_applicable("rust_build_failed");
        assert_eq!(applicable.len(), 1);
        assert_eq!(applicable[0].id, id);

        assert!(registry.find_applicable("go_build_failed").is_empty());
        assert_eq!(registry.by_stage(SkillStage::Discovered).len(), 1);
        assert_eq!(registry.by_stage(SkillStage::Promoted).len(), 1);
    }
}
