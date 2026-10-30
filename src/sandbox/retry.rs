//! Retrying a transient sandbox transport failure. Mirrors `agents.sandbox.util.retry`.
//!
//! A hosted sandbox is reached over the network, so a call can fail for reasons that have
//! nothing to do with what it asked for: a gateway restart, a connection reset, a 503 while the
//! provider scales. Retrying those is the difference between a run that continues and a run that
//! dies on a blip. Retrying anything else — a rejected command, a missing file — only wastes
//! time and can repeat a side effect, so the decision is always the caller's, through
//! `should_retry`.
//!
//! This is not [`crate::retry`], which decides whether to re-ask the *model*.

use std::future::Future;
use std::time::Duration;

use crate::sandbox::errors::{ErrorCode, SandboxError};

/// How the delay grows between attempts. Mirrors Python's `BackoffStrategy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackoffStrategy {
    /// The same delay every time.
    Fixed,
    /// The delay grows by one interval per attempt.
    Linear,
    /// The delay doubles on each attempt.
    #[default]
    Exponential,
}

impl BackoffStrategy {
    /// The wire name of the strategy.
    pub fn as_str(self) -> &'static str {
        match self {
            BackoffStrategy::Fixed => "fixed",
            BackoffStrategy::Linear => "linear",
            BackoffStrategy::Exponential => "exponential",
        }
    }

    /// The delay before the retry that follows attempt `attempt` (1-based).
    pub fn delay(self, interval: Duration, attempt: u32) -> Duration {
        match self {
            BackoffStrategy::Fixed => interval,
            BackoffStrategy::Linear => interval.saturating_mul(attempt),
            BackoffStrategy::Exponential => {
                interval.saturating_mul(2u32.saturating_pow(attempt.saturating_sub(1)))
            }
        }
    }
}

/// The default delay before the first retry. Mirrors `DEFAULT_TRANSIENT_RETRY_INTERVAL_S`.
pub const DEFAULT_TRANSIENT_RETRY_INTERVAL: Duration = Duration::from_millis(250);

/// The default number of attempts. Mirrors `DEFAULT_TRANSIENT_RETRY_MAX_ATTEMPT`.
pub const DEFAULT_TRANSIENT_RETRY_MAX_ATTEMPTS: u32 = 3;

/// HTTP status codes a sandbox transport treats as transient. Mirrors
/// `TRANSIENT_HTTP_STATUS_CODES`.
///
/// Only server-side failures are here. A 429 is deliberately absent: it asks the caller to slow
/// down, which a fixed schedule does not honour.
pub const TRANSIENT_HTTP_STATUS_CODES: &[u16] = &[500, 502, 503, 504];

/// Whether `status` is one a sandbox transport should retry.
pub fn is_transient_http_status(status: u16) -> bool {
    TRANSIENT_HTTP_STATUS_CODES.contains(&status)
}

/// How a retry loop behaves. Mirrors the arguments of Python's `retry_async`.
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    /// The base delay.
    pub interval: Duration,
    /// How many attempts in total; one means no retry.
    pub max_attempts: u32,
    /// How the delay grows.
    pub backoff: BackoffStrategy,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        RetryPolicy {
            interval: DEFAULT_TRANSIENT_RETRY_INTERVAL,
            max_attempts: DEFAULT_TRANSIENT_RETRY_MAX_ATTEMPTS,
            backoff: BackoffStrategy::Exponential,
        }
    }
}

impl RetryPolicy {
    /// A policy with the sandbox defaults.
    pub fn new() -> Self {
        RetryPolicy::default()
    }

    /// Set the base delay.
    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.interval = interval;
        self
    }

    /// Set how many attempts are made in total. Values below one are treated as one.
    pub fn with_max_attempts(mut self, max_attempts: u32) -> Self {
        self.max_attempts = max_attempts.max(1);
        self
    }

    /// Set how the delay grows.
    pub fn with_backoff(mut self, backoff: BackoffStrategy) -> Self {
        self.backoff = backoff;
        self
    }

    /// The delay before the retry that follows attempt `attempt` (1-based).
    pub fn delay(self, attempt: u32) -> Duration {
        self.backoff.delay(self.interval, attempt)
    }
}

/// Run `operation`, retrying while `should_retry` says the error is transient.
///
/// `operation` is a factory because each attempt needs its own future. `should_retry` receives
/// the error and the 1-based attempt number; returning `false`, or exhausting
/// `policy.max_attempts`, returns the error to the caller unchanged.
///
/// ```no_run
/// # use std::time::Duration;
/// # use openai_agents_sdk::sandbox::retry::{RetryPolicy, retry_async};
/// # use openai_agents_sdk::sandbox::{ErrorCode, SandboxError};
/// # async fn example() -> Result<(), SandboxError> {
/// let result = retry_async(
///     RetryPolicy::new(),
///     || async { Err::<(), _>(SandboxError::new(ErrorCode::ExecTransportError, "reset")) },
///     |error, _attempt| error.code == ErrorCode::ExecTransportError,
/// )
/// .await;
/// # let _ = result;
/// # Ok(())
/// # }
/// ```
/// Whether an error is worth retrying the same sandbox operation for.
///
/// Mirrors the `retry_if` predicates Python attaches to its sandbox call sites: an explicit
/// `retryable` flag wins, and otherwise only a transport failure counts — a rejected path or a
/// non-zero command will fail again the same way.
pub fn is_transient_sandbox_error(error: &SandboxError) -> bool {
    if let Some(retryable) = error.retryable {
        return retryable;
    }
    matches!(error.code, ErrorCode::ExecTransportError)
}

pub async fn retry_async<T, E, F, Fut, R>(
    policy: RetryPolicy,
    mut operation: F,
    mut should_retry: R,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    R: FnMut(&E, u32) -> bool,
{
    let max_attempts = policy.max_attempts.max(1);
    let mut attempt = 1;
    loop {
        match operation().await {
            Ok(value) => return Ok(value),
            Err(error) => {
                if attempt >= max_attempts || !should_retry(&error, attempt) {
                    return Err(error);
                }
                let delay = policy.delay(attempt);
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
                attempt += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;

    fn policy() -> RetryPolicy {
        RetryPolicy::new().with_interval(Duration::ZERO)
    }

    #[test]
    fn backoff_schedules_match_python() {
        let interval = Duration::from_millis(100);
        assert_eq!(
            BackoffStrategy::Fixed.delay(interval, 3),
            Duration::from_millis(100)
        );
        assert_eq!(
            BackoffStrategy::Linear.delay(interval, 3),
            Duration::from_millis(300)
        );
        assert_eq!(
            BackoffStrategy::Exponential.delay(interval, 1),
            Duration::from_millis(100)
        );
        assert_eq!(
            BackoffStrategy::Exponential.delay(interval, 3),
            Duration::from_millis(400)
        );
    }

    #[tokio::test]
    async fn retries_until_the_operation_succeeds() {
        let attempts = Arc::new(AtomicU32::new(0));
        let counter = attempts.clone();
        let result: Result<u32, &str> = retry_async(
            policy(),
            move || {
                let counter = counter.clone();
                async move {
                    let attempt = counter.fetch_add(1, Ordering::SeqCst) + 1;
                    if attempt < 3 {
                        Err("transient")
                    } else {
                        Ok(attempt)
                    }
                }
            },
            |_, _| true,
        )
        .await;
        assert_eq!(result, Ok(3));
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn stops_at_the_attempt_limit() {
        let attempts = Arc::new(AtomicU32::new(0));
        let counter = attempts.clone();
        let result: Result<(), &str> = retry_async(
            policy().with_max_attempts(2),
            move || {
                let counter = counter.clone();
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Err("transient")
                }
            },
            |_, _| true,
        )
        .await;
        assert_eq!(result, Err("transient"));
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn a_non_transient_error_is_returned_at_once() {
        let attempts = Arc::new(AtomicU32::new(0));
        let counter = attempts.clone();
        let result: Result<(), &str> = retry_async(
            policy(),
            move || {
                let counter = counter.clone();
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Err("permanent")
                }
            },
            |error, _| *error != "permanent",
        )
        .await;
        assert_eq!(result, Err("permanent"));
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn classifies_transient_http_statuses() {
        assert!(is_transient_http_status(503));
        assert!(is_transient_http_status(500));
        // A rate limit asks for a different response than a fixed retry schedule.
        assert!(!is_transient_http_status(429));
        assert!(!is_transient_http_status(404));
    }
}
