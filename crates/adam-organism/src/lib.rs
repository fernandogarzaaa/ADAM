//! ADAM Organism: the composition root wiring genome, memory, skills,
//! beliefs, and the evolution engine into one stateful unit.

mod embedding;
mod organism;

pub use embedding::embed;
pub use organism::{AppliedEffect, Organism, OrganismError, ReflectionSummary};

#[cfg(test)]
mod tests {
    use super::*;
    use adam_beliefs::{Belief, EvidenceOrigin};
    use adam_evolution::{EvolutionSignals, SkillFailureSignal};
    use adam_memory::MemoryKind;
    use adam_skills::Skill;

    fn new_organism() -> Organism {
        Organism::new("ADAM", "test organism", ":memory:").unwrap()
    }

    #[test]
    fn identity_starts_at_version_one_and_persists_experiences_as_memories() {
        let organism = new_organism();
        assert_eq!(organism.identity().label, "1.0");

        let id = organism
            .memory_store(
                MemoryKind::Episodic,
                "cargo build failed: missing dependency",
                "tool:cargo_build",
                vec!["exit code 101".to_string()],
                0.9,
                0.05,
            )
            .unwrap();

        let results = organism
            .memory_query("cargo build failure", None, 5)
            .unwrap();
        assert!(results.iter().any(|(record, _)| record.id == id));
    }

    #[test]
    fn beliefs_influence_what_the_organism_currently_holds() {
        let mut organism = new_organism();
        let belief = Belief::form(
            "rust prevents data races",
            EvidenceOrigin::Reasoning,
            "type system guarantees",
            0.8,
        )
        .unwrap();
        organism.form_belief(belief);
        assert_eq!(organism.beliefs().all_active().len(), 1);
    }

    #[test]
    fn skills_evolve_through_the_registry() {
        let mut organism = new_organism();
        let mut skill = Skill::discover("rust-debugging", "debug rust builds", vec![]);
        skill.define_procedure("check deps", vec![]).unwrap();
        skill.record_test(true, "ok").unwrap();
        skill.evaluate(0.5).unwrap();
        skill.promote().unwrap();
        organism.register_skill(skill);

        assert_eq!(
            organism
                .skills()
                .by_stage(adam_skills::SkillStage::Promoted)
                .len(),
            1
        );
    }

    #[test]
    fn chronic_failures_become_proposals_that_never_auto_apply() {
        let mut organism = new_organism();
        let signals = EvolutionSignals {
            skill_failures: vec![SkillFailureSignal {
                skill_name: "flaky".to_string(),
                fitness_score: 0.1,
                failure_count: 5,
                failures: vec!["timeout".to_string()],
            }],
            ..Default::default()
        };
        let ids = organism.evolve(&signals);
        assert_eq!(ids.len(), 1);
        assert_eq!(organism.proposals().pending().len(), 1);

        // Nothing changed yet — the skill registry is untouched.
        assert!(organism.skills().is_empty());
    }

    #[test]
    fn accepting_a_retire_skill_proposal_actually_removes_the_skill() {
        let mut organism = new_organism();
        let mut skill = Skill::discover("flaky", "unreliable", vec![]);
        skill.define_procedure("try thing", vec![]).unwrap();
        skill.record_test(false, "failed").unwrap();
        skill.evaluate(0.5).unwrap(); // fitness 0.0 -> Rejected, still present in registry
        organism.register_skill(skill);
        assert_eq!(organism.skills().len(), 1);

        let signals = EvolutionSignals {
            skill_failures: vec![SkillFailureSignal {
                skill_name: "flaky".to_string(),
                fitness_score: 0.0,
                failure_count: 4,
                failures: vec!["failed".to_string()],
            }],
            ..Default::default()
        };
        let ids = organism.evolve(&signals);
        let effect = organism.accept_mutation(ids[0]).unwrap();

        assert!(matches!(effect, AppliedEffect::SkillRetired { .. }));
        assert!(organism.skills().is_empty());
    }

    #[test]
    fn accepting_a_genome_amendment_creates_a_new_version() {
        let mut organism = new_organism();
        let v1 = organism.identity().id;

        let proposal = adam_evolution::EvolutionProposal::new(
            adam_evolution::ProposalKind::AmendGenome {
                field: "preferences.tone".to_string(),
                current_value: "verbose".to_string(),
                suggested_value: "concise".to_string(),
            },
            "user prefers concise responses",
            vec![],
            0.9,
        );
        let id = organism.propose_mutation(proposal);
        let effect = organism.accept_mutation(id).unwrap();

        match effect {
            AppliedEffect::GenomeAmended { new_version, .. } => {
                assert_ne!(new_version, v1);
                assert_eq!(organism.identity().id, new_version);
                assert_eq!(
                    organism.genome().preferences.get("tone"),
                    Some(&"concise".to_string())
                );
            }
            other => panic!("expected GenomeAmended, got {other:?}"),
        }
    }

    #[test]
    fn rollback_restores_prior_genome_content_as_a_new_forward_version() {
        let mut organism = new_organism();
        let v1 = organism.identity().id;

        let proposal = adam_evolution::EvolutionProposal::new(
            adam_evolution::ProposalKind::AmendGenome {
                field: "preferences.tone".to_string(),
                current_value: "verbose".to_string(),
                suggested_value: "concise".to_string(),
            },
            "test",
            vec![],
            0.9,
        );
        let id = organism.propose_mutation(proposal);
        organism.accept_mutation(id).unwrap();
        assert_eq!(
            organism.genome().preferences.get("tone"),
            Some(&"concise".to_string())
        );

        let rolled_back = organism.rollback(v1, "regression").unwrap();
        assert_eq!(organism.identity().id, rolled_back);
        assert_eq!(organism.genome().preferences.get("tone"), None);
        assert_eq!(organism.history().len(), 3);
    }

    #[test]
    fn reflect_summarizes_every_subsystem() {
        let mut organism = new_organism();
        organism
            .memory_store(
                MemoryKind::SelfKnowledge,
                "I debug systematically",
                "seed",
                vec![],
                1.0,
                0.0,
            )
            .unwrap();
        organism.form_belief(
            Belief::form(
                "tests catch regressions",
                EvidenceOrigin::Reasoning,
                "seed",
                0.7,
            )
            .unwrap(),
        );

        let summary = organism.reflect().unwrap();
        assert_eq!(summary.genome_version, "1.0");
        assert_eq!(summary.total_memories, 1);
        assert_eq!(summary.active_beliefs, 1);
        assert_eq!(summary.pending_proposals, 0);
    }

    #[test]
    fn every_accept_reject_and_rollback_is_written_to_the_audit_log() {
        let mut organism = new_organism();
        let v1 = organism.identity().id;

        let accept_proposal = adam_evolution::EvolutionProposal::new(
            adam_evolution::ProposalKind::AmendGenome {
                field: "preferences.tone".to_string(),
                current_value: "verbose".to_string(),
                suggested_value: "concise".to_string(),
            },
            "test",
            vec![],
            0.9,
        );
        let accept_id = organism.propose_mutation(accept_proposal);
        organism.accept_mutation(accept_id).unwrap();

        let reject_proposal = adam_evolution::EvolutionProposal::new(
            adam_evolution::ProposalKind::InvestigateConflict {
                topic: "x".to_string(),
            },
            "test",
            vec![],
            0.5,
        );
        let reject_id = organism.propose_mutation(reject_proposal);
        organism.reject_mutation(reject_id).unwrap();

        organism.rollback(v1, "regression").unwrap();

        assert_eq!(organism.audit_log().len(), 3);
    }

    #[test]
    fn evolution_rate_limit_blocks_excess_acceptances_without_dropping_the_proposal() {
        let mut organism = Organism::new("ADAM", "test", ":memory:").unwrap();
        // Exhaust the default limit (5 per 24h) with cheap retire-skill proposals.
        let mut last_id = None;
        for i in 0..5 {
            let mut skill = Skill::discover(format!("skill-{i}"), "d", vec![]);
            skill.define_procedure("p", vec![]).unwrap();
            skill.record_test(false, "f").unwrap();
            skill.evaluate(0.9).unwrap();
            organism.register_skill(skill);

            let proposal = adam_evolution::EvolutionProposal::new(
                adam_evolution::ProposalKind::RetireSkill {
                    skill_name: format!("skill-{i}"),
                },
                "test",
                vec![],
                0.9,
            );
            let id = organism.propose_mutation(proposal);
            organism.accept_mutation(id).unwrap();
            last_id = Some(id);
        }
        let _ = last_id;

        let mut skill = Skill::discover("skill-5", "d", vec![]);
        skill.define_procedure("p", vec![]).unwrap();
        organism.register_skill(skill);
        let sixth = organism.propose_mutation(adam_evolution::EvolutionProposal::new(
            adam_evolution::ProposalKind::RetireSkill {
                skill_name: "skill-5".to_string(),
            },
            "test",
            vec![],
            0.9,
        ));

        let err = organism.accept_mutation(sixth).unwrap_err();
        assert!(matches!(err, OrganismError::Governance(_)));
        // The proposal is untouched — still pending, not silently dropped.
        assert_eq!(
            organism.proposals().get(sixth).unwrap().status,
            adam_evolution::ProposalStatus::Proposed
        );
    }
}
