use std::future::Future;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::header::HeaderMap;
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use crate::error::{Error, Result};

/// Parse the HTTP `Retry-After` header.
///
/// The header has two valid forms:
///
/// ```text
/// Retry-After: 120
/// Retry-After: Wed, 21 Oct 2026 07:28:00 GMT
/// ```
///
/// The first form means "wait 120 seconds". The second means "wait until this
/// date". If the date is already in the past, the returned duration is zero.
pub fn parse_retry_after(value: &str) -> Option<Duration> {
    parse_retry_after_at(value, SystemTime::now())
}

/// Deterministic version of [`parse_retry_after`] for tests and custom clocks.
pub fn parse_retry_after_at(value: &str, now: SystemTime) -> Option<Duration> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }

    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }

    let retry_at = httpdate::parse_http_date(value).ok()?;
    Some(
        retry_at
            .duration_since(now)
            .unwrap_or(Duration::from_secs(0)),
    )
}

/// Provider family used to decide which vendor-specific reset headers to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryProvider {
    Generic,
    OpenAi,
    Anthropic,
    Google,
}

/// Where a retry delay came from. This is useful for logs and debugging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrySource {
    RetryAfterMs,
    RetryAfterSeconds,
    RetryAfterDate,
    OpenAiResetRequests,
    OpenAiResetTokens,
    AnthropicResetRequests,
    AnthropicResetTokens,
    AnthropicResetInputTokens,
    AnthropicResetOutputTokens,
}

/// A parsed server-directed retry instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryDirective {
    pub delay: Duration,
    pub source: RetrySource,
}

const RETRY_AFTER_MS: &str = "retry-after-ms";
const RETRY_AFTER: &str = "retry-after";
const OPENAI_RESET_REQUESTS: &str = "x-ratelimit-reset-requests";
const OPENAI_RESET_TOKENS: &str = "x-ratelimit-reset-tokens";
const ANTHROPIC_RESET_REQUESTS: &str = "anthropic-ratelimit-requests-reset";
const ANTHROPIC_RESET_TOKENS: &str = "anthropic-ratelimit-tokens-reset";
const ANTHROPIC_RESET_INPUT_TOKENS: &str = "anthropic-ratelimit-input-tokens-reset";
const ANTHROPIC_RESET_OUTPUT_TOKENS: &str = "anthropic-ratelimit-output-tokens-reset";

/// Read all supported retry headers using the current time.
pub fn parse_retry_headers(provider: RetryProvider, headers: &HeaderMap) -> Option<RetryDirective> {
    parse_retry_headers_at(provider, headers, SystemTime::now())
}

/// Deterministic version of [`parse_retry_headers`].
pub fn parse_retry_headers_at(
    provider: RetryProvider,
    headers: &HeaderMap,
    now: SystemTime,
) -> Option<RetryDirective> {
    // `retry-after-ms` is non-standard, but is supported by the official
    // OpenAI, Anthropic and Google SDKs and is more precise.
    if let Some(value) = header_value(headers, RETRY_AFTER_MS)
        && let Some(delay) = parse_millis(value)
    {
        return Some(RetryDirective {
            delay,
            source: RetrySource::RetryAfterMs,
        });
    }

    // Standard HTTP Retry-After: integer/float seconds or HTTP-date.
    if let Some(value) = header_value(headers, RETRY_AFTER) {
        if let Some(delay) = parse_seconds(value) {
            return Some(RetryDirective {
                delay,
                source: RetrySource::RetryAfterSeconds,
            });
        }
        if let Some(delay) = parse_retry_after_at(value, now) {
            return Some(RetryDirective {
                delay,
                source: RetrySource::RetryAfterDate,
            });
        }
    }

    match provider {
        RetryProvider::OpenAi => {
            if let Some(value) = header_value(headers, OPENAI_RESET_REQUESTS)
                && let Some(delay) = parse_go_duration(value)
            {
                return Some(RetryDirective {
                    delay,
                    source: RetrySource::OpenAiResetRequests,
                });
            }
            if let Some(value) = header_value(headers, OPENAI_RESET_TOKENS)
                && let Some(delay) = parse_go_duration(value)
            {
                return Some(RetryDirective {
                    delay,
                    source: RetrySource::OpenAiResetTokens,
                });
            }
        }
        RetryProvider::Anthropic => {
            let headers_and_sources = [
                (
                    ANTHROPIC_RESET_REQUESTS,
                    RetrySource::AnthropicResetRequests,
                ),
                (ANTHROPIC_RESET_TOKENS, RetrySource::AnthropicResetTokens),
                (
                    ANTHROPIC_RESET_INPUT_TOKENS,
                    RetrySource::AnthropicResetInputTokens,
                ),
                (
                    ANTHROPIC_RESET_OUTPUT_TOKENS,
                    RetrySource::AnthropicResetOutputTokens,
                ),
            ];
            for (header, source) in headers_and_sources {
                if let Some(value) = header_value(headers, header)
                    && let Some(delay) = parse_rfc3339_until(value, now)
                {
                    return Some(RetryDirective { delay, source });
                }
            }
        }
        RetryProvider::Google | RetryProvider::Generic => {}
    }

    None
}

fn header_value<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name)?.to_str().ok()
}

fn parse_millis(value: &str) -> Option<Duration> {
    let milliseconds = value.trim().parse::<f64>().ok()?;
    if !milliseconds.is_finite() || milliseconds < 0.0 {
        return None;
    }
    Some(Duration::from_secs_f64(milliseconds / 1000.0))
}

fn parse_seconds(value: &str) -> Option<Duration> {
    let seconds = value.trim().parse::<f64>().ok()?;
    if !seconds.is_finite() || seconds < 0.0 {
        return None;
    }
    Some(Duration::from_secs_f64(seconds))
}

/// Parse OpenAI's reset headers such as `1s`, `6m0s`, `1h`, `500ms`.
///
/// These headers are informational rate-limit headers; they are used only as
/// a fallback when neither `retry-after-ms` nor `retry-after` is present.
fn parse_go_duration(value: &str) -> Option<Duration> {
    let mut rest = value.trim();
    if rest.is_empty() {
        return None;
    }

    let mut total = 0.0_f64;
    let mut saw_value = false;

    while !rest.is_empty() {
        let number_len = rest
            .find(|character: char| !character.is_ascii_digit() && character != '.')
            .unwrap_or(rest.len());
        if number_len == 0 {
            return None;
        }

        let amount = rest[..number_len].parse::<f64>().ok()?;
        rest = &rest[number_len..];
        saw_value = true;

        if rest.is_empty() {
            // A bare number in a reset header is treated as seconds.
            total += amount;
            break;
        }

        let (unit, seconds) = if rest.starts_with("ns") {
            ("ns", amount / 1_000_000_000.0)
        } else if rest.starts_with("us") {
            ("us", amount / 1_000_000.0)
        } else if rest.starts_with("µs") {
            ("µs", amount / 1_000_000.0)
        } else if rest.starts_with("ms") {
            ("ms", amount / 1_000.0)
        } else if rest.starts_with('s') {
            ("s", amount)
        } else if rest.starts_with('m') {
            ("m", amount * 60.0)
        } else if rest.starts_with('h') {
            ("h", amount * 3600.0)
        } else {
            return None;
        };

        total += seconds;
        rest = &rest[unit.len()..];
    }

    if !saw_value || !total.is_finite() || total < 0.0 {
        return None;
    }
    Some(Duration::from_secs_f64(total))
}

fn parse_rfc3339_until(value: &str, now: SystemTime) -> Option<Duration> {
    let retry_at = OffsetDateTime::parse(value.trim(), &Rfc3339).ok()?;
    let retry_nanos = retry_at.unix_timestamp_nanos();
    let now_nanos = now.duration_since(UNIX_EPOCH).ok()?.as_nanos() as i128;
    let delta = retry_nanos - now_nanos;

    if delta <= 0 {
        return Some(Duration::ZERO);
    }

    u64::try_from(delta)
        .ok()
        .map(Duration::from_nanos)
        .or(Some(Duration::MAX))
}

/// Controls how transient errors are retried.
#[derive(Debug, Clone, PartialEq)]
pub struct RetryPolicy {
    /// Total attempts, including the first request. `4` means 1 initial try
    /// plus up to 3 retries.
    pub max_attempts: u32,
    /// Base delay for exponential backoff.
    pub base_delay: Duration,
    /// Maximum delay for computed backoff. A server-provided `Retry-After`
    /// is handled separately by `max_retry_after`.
    pub max_delay: Duration,
    /// Fractional jitter around the computed delay. `0.2` means ±20%.
    pub jitter_ratio: f32,
    /// If `Retry-After` is larger than this, stop retrying instead of waiting
    /// an unexpectedly long time.
    pub max_retry_after: Duration,
    /// When true, a request without an idempotency key is sent once but is not
    /// automatically replayed.
    pub require_idempotency_key: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 4,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(8),
            jitter_ratio: 0.2,
            max_retry_after: Duration::from_secs(60),
            require_idempotency_key: true,
        }
    }
}

impl RetryPolicy {
    pub fn new(max_attempts: u32) -> Self {
        Self {
            max_attempts,
            ..Self::default()
        }
    }

    pub fn max_attempts(mut self, max_attempts: u32) -> Self {
        self.max_attempts = max_attempts;
        self
    }

    pub fn base_delay(mut self, base_delay: Duration) -> Self {
        self.base_delay = base_delay;
        self
    }

    pub fn max_delay(mut self, max_delay: Duration) -> Self {
        self.max_delay = max_delay;
        self
    }

    pub fn jitter_ratio(mut self, jitter_ratio: f32) -> Self {
        self.jitter_ratio = jitter_ratio;
        self
    }

    pub fn max_retry_after(mut self, max_retry_after: Duration) -> Self {
        self.max_retry_after = max_retry_after;
        self
    }

    pub fn require_idempotency_key(mut self, require_idempotency_key: bool) -> Self {
        self.require_idempotency_key = require_idempotency_key;
        self
    }

    /// Compute the wait before the next attempt.
    ///
    /// `retry_number` is zero-based: the first failure uses `0`.
    /// `None` means "do not retry".
    pub fn delay_for(&self, retry_number: u32, retry_after: Option<Duration>) -> Option<Duration> {
        if let Some(retry_after) = retry_after {
            if retry_after > self.max_retry_after {
                return None;
            }
            return Some(retry_after);
        }

        let exponent = retry_number.min(20);
        let multiplier = 2_u32.pow(exponent);
        let base = self
            .base_delay
            .saturating_mul(multiplier)
            .min(self.max_delay);
        let jitter = self.jitter_ratio.clamp(0.0, 1.0);

        if jitter == 0.0 || base.is_zero() {
            return Some(base);
        }

        let unit = pseudo_random_unit(retry_number);
        let factor = (1.0 - jitter) + (2.0 * jitter * unit);
        Some(base.mul_f32(factor).min(self.max_delay))
    }
}

/// Retry an async operation according to a [`RetryPolicy`].
///
/// The operation closure is called again for every attempt, so each request
/// body and HTTP request can be rebuilt cleanly.
pub async fn retry_async<F, Fut, T>(policy: &RetryPolicy, mut operation: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    if policy.max_attempts == 0 {
        return Err(Error::InvalidRetryPolicy(
            "max_attempts must be at least 1".to_string(),
        ));
    }

    for attempt in 0..policy.max_attempts {
        match operation().await {
            Ok(value) => return Ok(value),
            Err(error) => {
                let Some(delay) = next_delay(policy, attempt, &error) else {
                    return Err(error);
                };
                tokio::time::sleep(delay).await;
            }
        }
    }

    unreachable!("the loop returns on success or on the final error")
}

pub(crate) fn next_delay(policy: &RetryPolicy, attempt: u32, error: &Error) -> Option<Duration> {
    let has_attempt_left = attempt + 1 < policy.max_attempts;
    if !has_attempt_left || !error.is_retryable() {
        return None;
    }
    policy.delay_for(attempt, error.retry_after())
}

fn pseudo_random_unit(seed: u32) -> f32 {
    let time_seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.subsec_nanos())
        .unwrap_or(0);
    let mut value = seed ^ time_seed;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    value as f32 / u32::MAX as f32
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use reqwest::header::HeaderValue;

    use super::*;

    fn api_error(status: u16, retry_after: Option<Duration>) -> Error {
        Error::Api {
            provider: "test",
            status,
            body: "test error".to_string(),
            retry_after,
            retry_source: None,
            request_id: None,
        }
    }

    #[test]
    fn parses_delta_seconds() {
        assert_eq!(parse_retry_after("120"), Some(Duration::from_secs(120)));
        assert_eq!(parse_retry_after(" 5 "), Some(Duration::from_secs(5)));
    }

    #[test]
    fn parses_http_date() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000);
        let retry_at = now + Duration::from_secs(30);
        let value = httpdate::fmt_http_date(retry_at);

        assert_eq!(
            parse_retry_after_at(&value, now),
            Some(Duration::from_secs(30))
        );
    }

    #[test]
    fn past_http_date_becomes_zero() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000);
        let retry_at = now - Duration::from_secs(30);
        let value = httpdate::fmt_http_date(retry_at);

        assert_eq!(parse_retry_after_at(&value, now), Some(Duration::ZERO));
    }

    #[test]
    fn rejects_garbage() {
        assert_eq!(parse_retry_after("not-a-date"), None);
        assert_eq!(parse_retry_after(""), None);
    }

    #[test]
    fn honors_server_retry_after() {
        let policy = RetryPolicy::default();
        assert_eq!(
            policy.delay_for(0, Some(Duration::from_secs(3))),
            Some(Duration::from_secs(3))
        );
    }

    #[test]
    fn refuses_too_long_retry_after() {
        let policy = RetryPolicy::default().max_retry_after(Duration::from_secs(5));
        assert_eq!(policy.delay_for(0, Some(Duration::from_secs(6))), None);
    }

    #[test]
    fn prefers_retry_after_ms() {
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER_MS, HeaderValue::from_static("1500"));
        headers.insert(RETRY_AFTER, HeaderValue::from_static("10"));

        let directive = parse_retry_headers(RetryProvider::Generic, &headers).unwrap();
        assert_eq!(directive.delay, Duration::from_millis(1500));
        assert_eq!(directive.source, RetrySource::RetryAfterMs);
    }

    #[test]
    fn parses_openai_reset_duration_headers() {
        let mut headers = HeaderMap::new();
        headers.insert(OPENAI_RESET_REQUESTS, HeaderValue::from_static("6m0s"));
        headers.insert(OPENAI_RESET_TOKENS, HeaderValue::from_static("1h"));

        let directive = parse_retry_headers(RetryProvider::OpenAi, &headers).unwrap();
        assert_eq!(directive.delay, Duration::from_secs(360));
        assert_eq!(directive.source, RetrySource::OpenAiResetRequests);
    }

    #[test]
    fn parses_anthropic_rfc3339_reset_headers() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000);
        let mut headers = HeaderMap::new();
        // 1970-01-01T00:16:40Z is exactly `now`; the reset is 45 seconds later.
        headers.insert(
            ANTHROPIC_RESET_REQUESTS,
            HeaderValue::from_static("1970-01-01T00:17:25Z"),
        );

        let directive = parse_retry_headers_at(RetryProvider::Anthropic, &headers, now).unwrap();
        assert_eq!(directive.delay, Duration::from_secs(45));
        assert_eq!(directive.source, RetrySource::AnthropicResetRequests);
    }

    #[test]
    fn computes_exponential_backoff_without_jitter() {
        let policy = RetryPolicy::new(4)
            .base_delay(Duration::from_secs(1))
            .max_delay(Duration::from_secs(10))
            .jitter_ratio(0.0);

        assert_eq!(policy.delay_for(0, None), Some(Duration::from_secs(1)));
        assert_eq!(policy.delay_for(1, None), Some(Duration::from_secs(2)));
        assert_eq!(policy.delay_for(2, None), Some(Duration::from_secs(4)));
        assert_eq!(policy.delay_for(3, None), Some(Duration::from_secs(8)));
        assert_eq!(policy.delay_for(4, None), Some(Duration::from_secs(10)));
    }

    #[tokio::test]
    async fn retries_until_success() {
        let policy = RetryPolicy::new(3)
            .base_delay(Duration::ZERO)
            .max_delay(Duration::ZERO)
            .jitter_ratio(0.0);
        let attempts = Arc::new(AtomicUsize::new(0));

        let result = retry_async(&policy, || {
            let attempt = attempts.fetch_add(1, Ordering::SeqCst);
            async move {
                if attempt < 2 {
                    Err(api_error(429, None))
                } else {
                    Ok(attempt)
                }
            }
        })
        .await
        .unwrap();

        assert_eq!(result, 2);
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn stops_on_non_retryable_error() {
        let policy = RetryPolicy::new(3)
            .base_delay(Duration::ZERO)
            .jitter_ratio(0.0);
        let attempts = Arc::new(AtomicUsize::new(0));

        let result: Result<()> = retry_async(&policy, || {
            attempts.fetch_add(1, Ordering::SeqCst);
            async { Err(api_error(401, None)) }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn stops_when_retry_after_is_too_long() {
        let policy = RetryPolicy::new(5)
            .base_delay(Duration::ZERO)
            .max_retry_after(Duration::from_secs(5));
        let attempts = Arc::new(AtomicUsize::new(0));

        let result: Result<()> = retry_async(&policy, || {
            attempts.fetch_add(1, Ordering::SeqCst);
            async { Err(api_error(429, Some(Duration::from_secs(60)))) }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }
}
