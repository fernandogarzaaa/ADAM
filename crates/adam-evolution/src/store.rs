//! In-memory registry of proposals awaiting or having received a
//! governance decision.

use std::collections::HashMap;

use crate::proposal::{EvolutionProposal, ProposalId, ProposalStatus};

#[derive(Debug, Default)]
pub struct ProposalStore {
    proposals: HashMap<ProposalId, EvolutionProposal>,
}

impl ProposalStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, proposal: EvolutionProposal) -> ProposalId {
        let id = proposal.id;
        self.proposals.insert(id, proposal);
        id
    }

    pub fn record_all(&mut self, proposals: Vec<EvolutionProposal>) -> Vec<ProposalId> {
        proposals.into_iter().map(|p| self.record(p)).collect()
    }

    pub fn get(&self, id: ProposalId) -> Option<&EvolutionProposal> {
        self.proposals.get(&id)
    }

    pub fn get_mut(&mut self, id: ProposalId) -> Option<&mut EvolutionProposal> {
        self.proposals.get_mut(&id)
    }

    pub fn pending(&self) -> Vec<&EvolutionProposal> {
        self.proposals
            .values()
            .filter(|p| p.status == ProposalStatus::Proposed)
            .collect()
    }

    pub fn accepted(&self) -> Vec<&EvolutionProposal> {
        self.proposals
            .values()
            .filter(|p| p.status == ProposalStatus::Accepted)
            .collect()
    }

    pub fn len(&self) -> usize {
        self.proposals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.proposals.is_empty()
    }
}
