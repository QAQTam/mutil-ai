use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::time::{Duration, Instant};

use crate::error::{Error, ErrorKind};
use crate::headers::RequestOptions;
use crate::normalize::NormalizeStats;
use crate::profile::Capabilities;
use crate::types::Usage;

/// Controls which structured transport events are emitted.
///
/// Audit events never contain prompts, messages, reasoning, tool arguments,
/// tool results, request bodies, or response bodies. Correlation identifiers
/// and profile metadata require explicit opt-in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuditConfig {
    /// Master switch: when `false`, no audit events are emitted.
    pub enabled: bool,
    /// Emit [`AuditEvent::AttemptStarted`] and [`AuditEvent::AttemptFinished`].
    pub record_attempts: bool,
    /// Emit retry and stream-reconnect scheduling events.
    pub record_retries: bool,
    /// Emit success outcomes for attempts and requests.
    pub record_success: bool,
    /// Emit failure outcomes for attempts and requests.
    pub record_failure: bool,
    /// Emit a final [`AuditEvent::RequestFinished`] with outcome
    /// [`AuditOutcome::Cancelled`] when a stream is dropped before completion.
    pub record_cancellation: bool,
    /// Emit response-header and first-token timing events.
    pub record_timing: bool,
    /// Emit context-free aggregate counts for normalization repairs.
    pub record_normalization: bool,
    /// Include the profile snapshot in [`AuditEvent::RequestStarted`].
    pub include_profile: bool,
    /// Include the caller-supplied SDK request id in
    /// [`AuditEvent::RequestStarted`].
    pub include_sdk_request_id: bool,
    /// Include the provider's request id in header, attempt, and completion
    /// events.
    pub include_provider_request_id: bool,
    /// Include the session id in [`AuditEvent::RequestStarted`]. Off by
    /// default because session ids can be tenant-sensitive.
    pub include_session_id: bool,
    /// Include token usage in [`AuditEvent::RequestFinished`].
    pub include_usage: bool,
    /// Sample one out of every N logical requests. `1` records every request.
    pub sample_every: u64,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            record_attempts: true,
            record_retries: true,
            record_success: true,
            record_failure: true,
            record_cancellation: true,
            record_timing: true,
            record_normalization: true,
            include_profile: false,
            include_sdk_request_id: false,
            include_provider_request_id: true,
            include_session_id: false,
            include_usage: true,
            sample_every: 1,
        }
    }
}

impl AuditConfig {
    /// Creates a config with auditing enabled and all other defaults kept.
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Self::default()
        }
    }

    /// Sets whether attempt-level events are emitted.
    pub fn record_attempts(mut self, enabled: bool) -> Self {
        self.record_attempts = enabled;
        self
    }

    /// Sets whether retry and reconnect scheduling events are emitted.
    pub fn record_retries(mut self, enabled: bool) -> Self {
        self.record_retries = enabled;
        self
    }

    /// Sets whether success outcomes are emitted.
    pub fn record_success(mut self, enabled: bool) -> Self {
        self.record_success = enabled;
        self
    }

    /// Sets whether failure outcomes are emitted.
    pub fn record_failure(mut self, enabled: bool) -> Self {
        self.record_failure = enabled;
        self
    }

    /// Emit a final event when a stream is dropped before terminal completion.
    pub fn record_cancellation(mut self, enabled: bool) -> Self {
        self.record_cancellation = enabled;
        self
    }

    /// Emit response-header and first-token timing events.
    pub fn record_timing(mut self, enabled: bool) -> Self {
        self.record_timing = enabled;
        self
    }

    /// Emit context-free aggregate counts for normalization repairs.
    pub fn record_normalization(mut self, enabled: bool) -> Self {
        self.record_normalization = enabled;
        self
    }

    /// Include the selected profile id, model-profile flag, and capability
    /// snapshot in `RequestStarted`.
    pub fn include_profile(mut self, enabled: bool) -> Self {
        self.include_profile = enabled;
        self
    }

    /// Sets whether the SDK request id is included in
    /// [`AuditEvent::RequestStarted`].
    pub fn include_sdk_request_id(mut self, enabled: bool) -> Self {
        self.include_sdk_request_id = enabled;
        self
    }

    /// Sets whether the provider request id is included in events.
    pub fn include_provider_request_id(mut self, enabled: bool) -> Self {
        self.include_provider_request_id = enabled;
        self
    }

    /// Session IDs can be tenant-sensitive. They are excluded by default.
    pub fn include_session_id(mut self, enabled: bool) -> Self {
        self.include_session_id = enabled;
        self
    }

    /// Sets whether token usage is included in
    /// [`AuditEvent::RequestFinished`].
    pub fn include_usage(mut self, enabled: bool) -> Self {
        self.include_usage = enabled;
        self
    }

    /// Sets the sampling rate; values below `1` are clamped to `1`.
    pub fn sample_every(mut self, sample_every: u64) -> Self {
        self.sample_every = sample_every.max(1);
        self
    }
}

/// The final outcome of an attempt or request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditOutcome {
    /// The operation completed without error.
    Success,
    /// The operation ended with an error.
    Failure,
    /// The operation was cancelled (including drop-before-completion).
    Cancelled,
}

/// Which model output started the stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FirstTokenKind {
    /// The first token was assistant-visible text.
    Text,
    /// The first token was reasoning output.
    Reasoning,
    /// The first token was tool-call data.
    ToolCall,
}

/// Content-free request and response byte counts.
///
/// `request_body` is the serialized body size for one HTTP attempt. Retries
/// reuse the same logical body. `response_body` is the cumulative decoded body
/// size across retries and stream reconnects, including provider error bodies.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ByteCounts {
    /// Serialized request body size for one HTTP attempt; retries reuse the
    /// same logical body.
    pub request_body: u64,
    /// Cumulative decoded response body size across retries and stream
    /// reconnects, including provider error bodies.
    pub response_body: u64,
}

/// Timing measured from the start of one logical request.
///
/// `time_to_headers` is available after the HTTP response headers arrive.
/// `time_to_first_token` is only available for streaming model output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RequestTiming {
    /// Time from request start until HTTP response headers arrived.
    pub time_to_headers: Option<Duration>,
    /// Time from request start until the first streamed token; `None` for
    /// non-streaming requests.
    pub time_to_first_token: Option<Duration>,
    /// The kind of output that produced the first token.
    pub first_token_kind: Option<FirstTokenKind>,
}

/// Non-content configuration snapshot for diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileAuditSnapshot {
    /// Provider profile id, when an explicit profile was selected.
    pub provider_profile_id: Option<String>,
    /// Whether a model-level profile matched this request.
    pub model_profile_selected: bool,
    /// Effective capability gate used for this request.
    pub capabilities: Option<Capabilities>,
}

/// A structured, context-free audit event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditEvent {
    /// A logical request (including all retries) began.
    RequestStarted {
        /// Provider identifier, e.g. `"openai"`.
        provider: &'static str,
        /// Wire protocol identifier, e.g. `"chat-completions"`.
        protocol: &'static str,
        /// Model name requested.
        model: String,
        /// Caller-supplied SDK request id, when opted in.
        sdk_request_id: Option<String>,
        /// Session id, when opted in.
        session_id: Option<String>,
        /// Profile snapshot, when opted in.
        profile: Option<ProfileAuditSnapshot>,
    },
    /// One HTTP attempt began.
    AttemptStarted {
        /// Attempt number, as tracked by the SDK.
        attempt: u32,
    },
    /// HTTP response headers arrived for an attempt.
    ResponseHeaders {
        /// Attempt this response belongs to.
        attempt: u32,
        /// Time from the start of that attempt until headers arrived.
        elapsed: Duration,
        /// HTTP status code.
        status: u16,
        /// Provider request id, when opted in.
        provider_request_id: Option<String>,
    },
    /// One HTTP attempt ended.
    AttemptFinished {
        /// Attempt that finished.
        attempt: u32,
        /// Duration of the attempt.
        duration: Duration,
        /// Whether the attempt succeeded or failed.
        outcome: AuditOutcome,
        /// HTTP status code, when a response was received.
        status: Option<u16>,
        /// Error classification, when the attempt failed.
        error_kind: Option<ErrorKind>,
        /// Whether the SDK considers the error retryable.
        retryable: bool,
        /// Provider request id, when opted in.
        provider_request_id: Option<String>,
    },
    /// The first streamed token arrived.
    FirstToken {
        /// Time from the start of the logical request.
        elapsed: Duration,
        /// Which kind of output produced the token.
        kind: FirstTokenKind,
    },
    /// A response body required normalization; only aggregate counts are
    /// reported, never content.
    Normalization {
        /// Context-free repair counters.
        stats: NormalizeStats,
    },
    /// A caller-requested response transform was applied.
    TransformApplied,
    /// A retry of the request was scheduled.
    RetryScheduled {
        /// Attempt that failed.
        attempt: u32,
        /// Attempt number that will run next.
        next_attempt: u32,
        /// Delay before the next attempt.
        delay: Duration,
        /// Classification of the error that triggered the retry.
        error_kind: Option<ErrorKind>,
    },
    /// A reconnect of an in-progress SSE stream was scheduled.
    StreamReconnectScheduled {
        /// Reconnect attempt number, starting at 1.
        attempt: u32,
        /// Delay before reconnecting.
        delay: Duration,
        /// Whether a last event id is available to resume from.
        has_last_event_id: bool,
    },
    /// The logical request (including retries) ended. Emitted exactly once
    /// per request.
    RequestFinished {
        outcome: AuditOutcome,
        duration: Duration,
        timing: RequestTiming,
        bytes: ByteCounts,
        status: Option<u16>,
        error_kind: Option<ErrorKind>,
        provider_request_id: Option<String>,
        usage: Option<Usage>,
    },
}

/// Receives context-free audit events.
pub trait AuditSink: Send + Sync {
    /// Handles one audit event. Implementations should be non-blocking; use
    /// [`bounded_audit_channel`] for a queue that never backpressures the
    /// request path.
    fn record(&self, event: AuditEvent);
}

/// Runtime counters for a bounded audit channel.
#[derive(Debug, Default)]
pub struct AuditStats {
    accepted: AtomicU64,
    dropped_full: AtomicU64,
    dropped_disconnected: AtomicU64,
}

impl AuditStats {
    /// Loads the current counters as a plain snapshot.
    pub fn snapshot(&self) -> AuditStatsSnapshot {
        AuditStatsSnapshot {
            accepted: self.accepted.load(Ordering::Relaxed),
            dropped_full: self.dropped_full.load(Ordering::Relaxed),
            dropped_disconnected: self.dropped_disconnected.load(Ordering::Relaxed),
        }
    }
}

/// A point-in-time copy of [`AuditStats`] counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AuditStatsSnapshot {
    /// Events successfully delivered to the channel.
    pub accepted: u64,
    /// Events dropped because the channel was full.
    pub dropped_full: u64,
    /// Events dropped because the receiver was disconnected.
    pub dropped_disconnected: u64,
}

impl AuditStatsSnapshot {
    /// Total number of dropped events.
    pub const fn dropped(&self) -> u64 {
        self.dropped_full + self.dropped_disconnected
    }
}

struct AuditChannelSink {
    sender: SyncSender<AuditEvent>,
    stats: Arc<AuditStats>,
}

impl AuditSink for AuditChannelSink {
    fn record(&self, event: AuditEvent) {
        match self.sender.try_send(event) {
            Ok(()) => {
                self.stats.accepted.fetch_add(1, Ordering::Relaxed);
            }
            Err(TrySendError::Full(_)) => {
                self.stats.dropped_full.fetch_add(1, Ordering::Relaxed);
            }
            Err(TrySendError::Disconnected(_)) => {
                self.stats
                    .dropped_disconnected
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Create a non-blocking, bounded audit channel.
///
/// A full queue drops the event and increments [`AuditStatsSnapshot::dropped_full`].
/// The model request path is never blocked by audit backpressure.
pub fn bounded_audit_channel(
    capacity: usize,
) -> (Arc<dyn AuditSink>, Receiver<AuditEvent>, Arc<AuditStats>) {
    let (sender, receiver) = sync_channel(capacity.max(1));
    let stats = Arc::new(AuditStats::default());
    let sink = Arc::new(AuditChannelSink {
        sender,
        stats: stats.clone(),
    });
    (sink, receiver, stats)
}

#[derive(Default)]
struct AuditState {
    time_to_headers_nanos: AtomicU64,
    time_to_first_token_nanos: AtomicU64,
    first_token_kind: AtomicU64,
    response_body_bytes: AtomicU64,
}

impl AuditState {
    const UNSET: u64 = u64::MAX;
}

#[derive(Clone)]
pub(crate) struct AuditContext {
    sink: Arc<dyn AuditSink>,
    config: AuditConfig,
    provider: &'static str,
    protocol: &'static str,
    model: String,
    profile: Option<ProfileAuditSnapshot>,
    sdk_request_id: Option<String>,
    session_id: Option<String>,
    request_body_bytes: u64,
    started_at: Instant,
    state: Arc<AuditState>,
}

pub(crate) struct AuditRequestMeta<'a> {
    pub provider: &'static str,
    pub protocol: &'static str,
    pub model: &'a str,
    pub profile: Option<ProfileAuditSnapshot>,
    pub request_body_bytes: u64,
}

impl AuditContext {
    pub(crate) fn new(
        sink: Arc<dyn AuditSink>,
        config: AuditConfig,
        options: &RequestOptions,
        meta: AuditRequestMeta<'_>,
    ) -> Self {
        let AuditRequestMeta {
            provider,
            protocol,
            model,
            profile,
            request_body_bytes,
        } = meta;
        Self {
            sink,
            config,
            provider,
            protocol,
            model: model.to_string(),
            profile: config.include_profile.then_some(profile).flatten(),
            sdk_request_id: config
                .include_sdk_request_id
                .then(|| options.context.request_id.clone())
                .flatten(),
            session_id: config
                .include_session_id
                .then(|| options.context.session_id.clone())
                .flatten(),
            request_body_bytes,
            started_at: Instant::now(),
            state: Arc::new(AuditState {
                time_to_headers_nanos: AtomicU64::new(AuditState::UNSET),
                time_to_first_token_nanos: AtomicU64::new(AuditState::UNSET),
                first_token_kind: AtomicU64::new(AuditState::UNSET),
                response_body_bytes: AtomicU64::new(0),
            }),
        }
    }

    pub(crate) fn request_started(&self) {
        if !self.config.enabled {
            return;
        }
        self.sink.record(AuditEvent::RequestStarted {
            provider: self.provider,
            protocol: self.protocol,
            model: self.model.clone(),
            sdk_request_id: self.sdk_request_id.clone(),
            session_id: self.session_id.clone(),
            profile: self.profile.clone(),
        });
    }

    pub(crate) fn attempt_started(&self, attempt: u32) {
        if !self.config.enabled || !self.config.record_attempts {
            return;
        }
        self.sink.record(AuditEvent::AttemptStarted { attempt });
    }

    pub(crate) fn response_headers(
        &self,
        attempt: u32,
        elapsed: Duration,
        status: u16,
        provider_request_id: Option<String>,
        successful: bool,
    ) {
        if !self.config.enabled || !self.config.record_timing {
            return;
        }
        if successful {
            let _ = self.state.time_to_headers_nanos.compare_exchange(
                AuditState::UNSET,
                duration_nanos(elapsed),
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
        }
        self.sink.record(AuditEvent::ResponseHeaders {
            attempt,
            elapsed,
            status,
            provider_request_id: self
                .config
                .include_provider_request_id
                .then_some(provider_request_id)
                .flatten(),
        });
    }

    pub(crate) fn first_token(&self, kind: FirstTokenKind) {
        if !self.config.enabled || !self.config.record_timing {
            return;
        }
        let kind_code = first_token_kind_code(kind);
        if self
            .state
            .first_token_kind
            .compare_exchange(
                AuditState::UNSET,
                kind_code,
                Ordering::Relaxed,
                Ordering::Relaxed,
            )
            .is_err()
        {
            return;
        }

        let elapsed = self.started_at.elapsed();
        self.state
            .time_to_first_token_nanos
            .store(duration_nanos(elapsed), Ordering::Relaxed);
        self.sink.record(AuditEvent::FirstToken { elapsed, kind });
    }

    pub(crate) fn normalization(&self, stats: NormalizeStats) {
        if !self.config.enabled || !self.config.record_normalization || stats.is_clean() {
            return;
        }
        self.sink.record(AuditEvent::Normalization { stats });
    }

    pub(crate) fn transform_applied(&self) {
        if self.config.enabled {
            self.sink.record(AuditEvent::TransformApplied);
        }
    }

    pub(crate) fn add_response_bytes(&self, bytes: usize) {
        if !self.config.enabled || bytes == 0 {
            return;
        }
        self.state
            .response_body_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }

    pub(crate) fn attempt_finished_with_request_id(
        &self,
        attempt: u32,
        duration: Duration,
        status: Option<u16>,
        provider_request_id: Option<String>,
        error: Option<&Error>,
    ) {
        if !self.config.enabled || !self.config.record_attempts {
            return;
        }
        let outcome = if error.is_some() {
            AuditOutcome::Failure
        } else {
            AuditOutcome::Success
        };
        if outcome == AuditOutcome::Success && !self.config.record_success {
            return;
        }
        if outcome == AuditOutcome::Failure && !self.config.record_failure {
            return;
        }
        self.sink.record(AuditEvent::AttemptFinished {
            attempt,
            duration,
            outcome,
            status,
            error_kind: error.map(Error::kind),
            retryable: error.is_some_and(Error::is_retryable),
            provider_request_id: self
                .config
                .include_provider_request_id
                .then_some(provider_request_id)
                .flatten(),
        });
    }

    pub(crate) fn retry_scheduled(
        &self,
        attempt: u32,
        next_attempt: u32,
        delay: Duration,
        error: &Error,
    ) {
        if !self.config.enabled || !self.config.record_retries {
            return;
        }
        self.sink.record(AuditEvent::RetryScheduled {
            attempt,
            next_attempt,
            delay,
            error_kind: Some(error.kind()),
        });
    }

    pub(crate) fn stream_reconnect_scheduled(
        &self,
        attempt: u32,
        delay: Duration,
        has_last_event_id: bool,
    ) {
        if !self.config.enabled || !self.config.record_retries {
            return;
        }
        self.sink.record(AuditEvent::StreamReconnectScheduled {
            attempt,
            delay,
            has_last_event_id,
        });
    }

    pub(crate) fn request_finished_with_request_id(
        &self,
        outcome: AuditOutcome,
        status: Option<u16>,
        provider_request_id: Option<String>,
        error: Option<&Error>,
        usage: Option<Usage>,
    ) {
        if !self.config.enabled {
            return;
        }
        if outcome == AuditOutcome::Success && !self.config.record_success {
            return;
        }
        if outcome == AuditOutcome::Failure && !self.config.record_failure {
            return;
        }
        if outcome == AuditOutcome::Cancelled && !self.config.record_cancellation {
            return;
        }
        self.sink.record(AuditEvent::RequestFinished {
            outcome,
            duration: self.started_at.elapsed(),
            timing: self.timing_snapshot(),
            bytes: ByteCounts {
                request_body: self.request_body_bytes,
                response_body: self.state.response_body_bytes.load(Ordering::Relaxed),
            },
            status,
            error_kind: error.map(Error::kind),
            provider_request_id: self
                .config
                .include_provider_request_id
                .then_some(provider_request_id)
                .flatten(),
            usage: self
                .config
                .include_usage
                .then_some(usage)
                .flatten()
                .map(Usage::without_raw),
        });
    }

    fn timing_snapshot(&self) -> RequestTiming {
        if !self.config.record_timing {
            return RequestTiming::default();
        }
        let time_to_headers = load_duration(&self.state.time_to_headers_nanos);
        let time_to_first_token = load_duration(&self.state.time_to_first_token_nanos);
        let first_token_kind = match self.state.first_token_kind.load(Ordering::Relaxed) {
            1 => Some(FirstTokenKind::Text),
            2 => Some(FirstTokenKind::Reasoning),
            3 => Some(FirstTokenKind::ToolCall),
            _ => None,
        };
        RequestTiming {
            time_to_headers,
            time_to_first_token,
            first_token_kind,
        }
    }
}

fn duration_nanos(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128 - 1) as u64
}

fn load_duration(value: &AtomicU64) -> Option<Duration> {
    match value.load(Ordering::Relaxed) {
        AuditState::UNSET => None,
        nanos => Some(Duration::from_nanos(nanos)),
    }
}

const fn first_token_kind_code(kind: FirstTokenKind) -> u64 {
    match kind {
        FirstTokenKind::Text => 1,
        FirstTokenKind::Reasoning => 2,
        FirstTokenKind::ToolCall => 3,
    }
}
