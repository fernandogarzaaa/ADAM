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

    #[test]
    fn genome_identity_survives_a_process_restart_via_open() {
        let genome_path =
            std::env::temp_dir().join(format!("adam_test_genome_{}.json", uuid::Uuid::new_v4()));
        let genome_path_str = genome_path.to_str().unwrap();

        let v1 = {
            let mut organism =
                Organism::open("ADAM", "restart test", ":memory:", genome_path_str).unwrap();
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
            v1
        };
        // Organism dropped here — simulates a process restart.

        let reopened = Organism::open("ADAM", "restart test", ":memory:", genome_path_str).unwrap();
        assert_ne!(reopened.identity().id, v1);
        assert_eq!(
            reopened.genome().preferences.get("tone"),
            Some(&"concise".to_string())
        );
        assert_eq!(reopened.history().len(), 2);

        std::fs::remove_file(genome_path).ok();
    }

    #[test]
    fn evolve_auto_derives_signals_from_current_organism_state_without_a_caller_supplying_them() {
        let mut organism = new_organism();

        // A chronically failing skill (3+ failures needed to cross the
        // default RetireSkill threshold).
        let mut skill = Skill::discover("flaky", "unreliable", vec![]);
        skill.define_procedure("try thing", vec![]).unwrap();
        skill.record_test(false, "boom").unwrap();
        skill.record_test(false, "boom again").unwrap();
        skill.record_test(false, "boom a third time").unwrap();
        skill.evaluate(0.9).unwrap();
        organism.register_skill(skill);

        // Two separately-retracted beliefs sharing a statement (2+ needed
        // to cross the default ReconcileBelief threshold).
        for _ in 0..2 {
            let mut belief = Belief::form(
                "the api is stable",
                EvidenceOrigin::Observation,
                "seed",
                0.3,
            )
            .unwrap();
            belief
                .add_evidence(EvidenceOrigin::Observation, "broke again", false, 1.0)
                .unwrap();
            organism.form_belief(belief);
        }

        // A recurring memory conflict on the same topic (3+ needed to
        // cross the default InvestigateConflict threshold).
        let winner_id = organism
            .memory_store(
                MemoryKind::Semantic,
                "the service is reliable",
                "seed",
                vec![],
                0.9,
                0.0,
            )
            .unwrap();
        for i in 0..3 {
            let loser_id = organism
                .memory_store(
                    MemoryKind::Semantic,
                    &format!("the service is unreliable {i}"),
                    "seed",
                    vec![],
                    0.2,
                    0.0,
                )
                .unwrap();
            organism
                .memory()
                .resolve_conflict(winner_id, loser_id)
                .unwrap();
        }

        let signals = organism.collect_signals().unwrap();
        assert_eq!(signals.skill_failures.len(), 1);
        assert_eq!(signals.skill_failures[0].skill_name, "flaky");
        assert_eq!(signals.belief_instabilities.len(), 1);
        assert_eq!(signals.belief_instabilities[0].retraction_count, 2);
        assert_eq!(signals.recurring_conflicts.len(), 1);
        assert_eq!(signals.recurring_conflicts[0].occurrences, 3);

        let ids = organism.evolve_auto().unwrap();
        assert!(!ids.is_empty());
    }

    #[test]
    fn amending_values_beyond_preferences_requires_a_prior_eve_approval() {
        let mut organism = new_organism();
        let proposal = adam_evolution::EvolutionProposal::new(
            adam_evolution::ProposalKind::AmendGenome {
                field: "values.append".to_string(),
                current_value: String::new(),
                suggested_value: "curiosity".to_string(),
            },
            "observed a pattern favoring curiosity",
            vec![],
            0.9,
        );
        let id = organism.propose_mutation(proposal);

        // No EVE evaluation recorded yet — accept must fail closed.
        let err = organism.accept_mutation(id).unwrap_err();
        assert!(matches!(err, OrganismError::EveApprovalRequired { .. }));
        assert!(organism.genome().values.is_empty());
    }

    #[test]
    fn eve_approval_unlocks_appending_to_a_genome_list_field() {
        let mut organism = new_organism();
        let proposal = adam_evolution::EvolutionProposal::new(
            adam_evolution::ProposalKind::AmendGenome {
                field: "values.append".to_string(),
                current_value: String::new(),
                suggested_value: "curiosity".to_string(),
            },
            "observed a pattern favoring curiosity",
            vec![],
            // Confidence 1.0 keeps AmendGenome's baseline risk (0.6) at
            // exactly the default max_acceptable_risk threshold rather
            // than over it, so a clean trial run can still reach Approve
            // instead of being forced into NeedsReview by risk alone.
            1.0,
        );
        let id = organism.propose_mutation(proposal);

        let trials = vec![
            adam_eve::TrialOutcome {
                succeeded: true,
                detail: "sandbox replay ok".to_string(),
            };
            5
        ];
        let evaluation = organism.evaluate_mutation_from_trials(id, trials).unwrap();
        assert_eq!(evaluation.recommendation, adam_eve::Recommendation::Approve);

        let effect = organism.accept_mutation(id).unwrap();
        assert!(matches!(effect, AppliedEffect::GenomeAmended { .. }));
        assert_eq!(organism.genome().values, vec!["curiosity".to_string()]);
    }

    #[test]
    fn eve_needs_review_recommendation_still_blocks_acceptance() {
        let mut organism = new_organism();
        let proposal = adam_evolution::EvolutionProposal::new(
            adam_evolution::ProposalKind::AmendGenome {
                field: "goals.append".to_string(),
                current_value: String::new(),
                suggested_value: "ship faster".to_string(),
            },
            "mixed evidence",
            vec![],
            0.9,
        );
        let id = organism.propose_mutation(proposal);

        // Half the trials fail -> fitness 0.5, in the NeedsReview band.
        let trials = vec![
            adam_eve::TrialOutcome {
                succeeded: true,
                detail: "ok".to_string(),
            },
            adam_eve::TrialOutcome {
                succeeded: false,
                detail: "regressed".to_string(),
            },
        ];
        organism.evaluate_mutation_from_trials(id, trials).unwrap();

        let err = organism.accept_mutation(id).unwrap_err();
        assert!(matches!(err, OrganismError::EveApprovalRequired { .. }));
        assert!(organism.genome().goals.is_empty());
    }

    #[test]
    fn preferences_amendments_remain_ungated_by_eve() {
        let mut organism = new_organism();
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

        // No EVE evaluation recorded, yet preferences.* still succeeds.
        let effect = organism.accept_mutation(id).unwrap();
        assert!(matches!(effect, AppliedEffect::GenomeAmended { .. }));
    }

    #[test]
    fn memory_query_ann_finds_the_same_top_hit_as_the_exact_scan() {
        let organism = new_organism();
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
        organism
            .memory_store(
                MemoryKind::Episodic,
                "the weather is sunny today",
                "tool:weather",
                vec![],
                0.9,
                0.05,
            )
            .unwrap();

        let results = organism.memory_query_ann("cargo build failure", 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.id, id);
    }
}
