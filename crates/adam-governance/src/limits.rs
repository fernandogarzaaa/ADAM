//! Evolution rate limiting: caps how many mutations the organism may
//! accept within a rolling time window, so a burst of proposals can never
//! silently cascade into runaway self-modification.

use chrono::{DateTime, Duration, Utc};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GovernanceError {
    #[error("evolution limit exceeded: {accepted} mutations already accepted within the last {window_secs}s (max {max})")]
    LimitExceeded {
        accepted: usize,
        max: u32,
        window_secs: i64,
    },
}

#[derive(Debug, Clone)]
pub struct EvolutionLimits {
    pub max_accepted_per_window: u32,
    pub window: Duration,
}

impl Default for EvolutionLimits {
    /// A conservative default: at most 5 accepted mutations per 24 hours.
    fn default() -> Self {
        Self {
            max_accepted_per_window: 5,
            window: Duration::hours(24),
        }
    }
}

/// Tracks accepted-mutation timestamps and enforces [`EvolutionLimits`].
#[derive(Debug)]
pub struct RateLimiter {
    limits: EvolutionLimits,
    accepted_at: Vec<DateTime<Utc>>,
}

impl RateLimiter {
    pub fn new(limits: EvolutionLimits) -> Self {
        Self {
            limits,
            accepted_at: Vec::new(),
        }
    }

    /// Count acceptances within the current window, as of `now`.
    fn accepted_within_window(&self, now: DateTime<Utc>) -> usize {
        let cutoff = now - self.limits.window;
        self.accepted_at.iter().filter(|t| **t > cutoff).count()
    }

    /// Check the limit and, if within bounds, record a new acceptance.
    pub fn check_and_record(&mut self, now: DateTime<Utc>) -> Result<(), GovernanceError> {
        let accepted = self.accepted_within_window(now);
        if accepted >= self.limits.max_accepted_per_window as usize {
            return Err(GovernanceError::LimitExceeded {
                accepted,
                max: self.limits.max_accepted_per_window,
                window_secs: self.limits.window.num_seconds(),
            });
        }
        self.accepted_at.push(now);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_acceptances_up_to_the_limit() {
        let mut limiter = RateLimiter::new(EvolutionLimits {
            max_accepted_per_window: 2,
            window: Duration::hours(1),
        });
        let now = Utc::now();
        assert!(limiter.check_and_record(now).is_ok());
        assert!(limiter.check_and_record(now).is_ok());
        let err = limiter.check_and_record(now).unwrap_err();
        assert_eq!(
            err,
            GovernanceError::LimitExceeded { accepted: 2, max: 2, window_secs: 3600 }
        );
    }

    #[test]
    fn old_acceptances_outside_the_window_do_not_count() {
        let mut limiter = RateLimiter::new(EvolutionLimits {
            max_accepted_per_window: 1,
            window: Duration::hours(1),
        });
        let two_hours_ago = Utc::now() - Duration::hours(2);
        assert!(limiter.check_and_record(two_hours_ago).is_ok());
        assert!(limiter.check_and_record(Utc::now()).is_ok());
    }
}
