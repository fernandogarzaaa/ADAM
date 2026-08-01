//! ADAM Kernel — the versioned genome substrate for the ADAM cognitive organism.
//!
//! The kernel owns the organism's genome: identity, values, goals, beliefs,
//! capabilities, skills, preferences, and behavioral policies. Every mutation
//! to the genome is recorded as an immutable, hash-linked version so the full
//! evolutionary history can be diffed, audited, and rolled back.

pub mod genome;

pub use genome::{Genome, GenomeDiff, GenomeHistory, GenomeVersion, VersionId};
