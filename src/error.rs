use std::fmt;
use std::time::Duration;

use thiserror::Error;

use crate::retry::RetrySource;

/// Convenience alias for results returned by this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Stable high-level classification for SDK errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// No API key was configured, or the provider rejected it (HTTP 401).
    Authentication,
    /// The credentials are valid but lack access to the resource (HTTP 403).
    PermissionDenied,
    /// The requested resource does not exist (HTTP 404).
    NotFound,
    /// The provider rejected the request as malformed (HTTP 400-class).
    InvalidRequest,
    /// The prompt or output exceeds the model's context window.
    ContextLengthExceeded,
    /// The request or output was blocked by a safety or content policy.
    ContentFiltered,
    /// The provider is rate limiting the account (HTTP 429).
    RateLimited,
    /// The provider is temporarily overloaded (e.g. HTTP 425 or 529).
    Overloaded,
    /// The request timed out.
    Timeout,
    /// A network-level failure prevented reaching the provider.
    Connection,
    /// The request was cancelled by the caller.
    Cancelled,
    /// A response body could not be decoded.
    Decode,
    /// The provider emitted a malformed or unexpected stream event.
    StreamProtocol,
    /// The provider itself reported an internal error (HTTP 5xx).
    ProviderInternal,
    /// The selected provider does not support the requested operation.
    Unsupported,
    /// The SDK or provider client was misconfigured.
    Configuration,
    /// Request normalization failed.
    Normalization,
    /// A tool invocation failed or referenced an unknown tool.
    Tool,
    /// The agent loop exhausted its step budget.
    MaxSteps,
    /// The error could not be classified.
    Unknown,
}

impl ErrorKind {
    /// Stable snake_case identifier for this kind, suitable for logs and
    /// metrics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Authentication => "authentication",
            Self::PermissionDenied => "permission_denied",
            Self::NotFound => "not_found",
            Self::InvalidRequest => "invalid_request",
            Self::ContextLengthExceeded => "context_length_exceeded",
            Self::ContentFiltered => "content_filtered",
            Self::RateLimited => "rate_limited",
            Self::Overloaded => "overloaded",
            Self::Timeout => "timeout",
            Self::Connection => "connection",
            Self::Cancelled => "cancelled",
            Self::Decode => "decode",
            Self::StreamProtocol => "stream_protocol",
            Self::ProviderInternal => "provider_internal",
            Self::Unsupported => "unsupported",
            Self::Configuration => "configuration",
            Self::Normalization => "normalization",
            Self::Tool => "tool",
            Self::MaxSteps => "max_steps",
            Self::Unknown => "unknown",
        }
    }

    /// Whether errors of this kind are normally safe to retry.
    pub const fn is_retryable(self) -> bool {
        matches!(
            self,
            Self::RateLimited
                | Self::Overloaded
                | Self::Timeout
                | Self::Connection
                | Self::ProviderInternal
        )
    }
}

/// Safe, provider-neutral summary of a structured error response.
///
/// The provider message itself is never retained. Only its byte length is
/// exposed, while code/type/status tokens are sanitized and length-limited.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProviderErrorInfo {
    /// Machine-readable provider code, such as `context_length_exceeded`.
    pub code: Option<String>,
    /// Provider error type/category, such as `invalid_request_error`.
    pub error_type: Option<String>,
    /// Provider status token, such as `INVALID_ARGUMENT`.
    pub status: Option<String>,
    /// Byte length of the provider's human-readable message, if present.
    pub message_bytes: Option<usize>,
    /// Number of structured `details` entries, when the provider returns them.
    pub details_count: usize,
}

/// The error type returned by this crate's public API.
///
/// Its [`Display`](std::fmt::Display) output is safe to log: the `Api`
/// variant never embeds the raw provider response body, and the `Debug`
/// output redacts it. Use [`Error::kind`] for stable classification and
/// [`Error::raw_body`] when you explicitly need the provider body.
#[derive(Error)]
pub enum Error {
    /// No API key was configured for the provider.
    #[error("missing API key: {0}")]
    MissingApiKey(String),

    /// The HTTP request to the provider failed at the transport level.
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    /// The request was cancelled before it completed.
    #[error("request cancelled")]
    Cancelled,

    /// The request exceeded its configured timeout.
    #[error("request timed out: {0}")]
    Timeout(String),

    /// A response body was not valid JSON.
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),

    /// The provider's stream emitted a malformed or unexpected event.
    #[error("stream protocol error: {0}")]
    StreamProtocol(String),

    /// The provider reported an error inside a stream.
    #[error("provider stream error: {0}")]
    ProviderStream(String),

    #[error("{provider} returned HTTP {status}")]
    Api {
        /// The provider identifier, such as `openai`.
        provider: &'static str,
        /// HTTP status code returned by the provider.
        status: u16,
        /// Raw response body. Excluded from log-safe output; see
        /// [`Error::raw_body`].
        body: String,
        /// Parsed server-directed retry delay, when present.
        retry_after: Option<Duration>,
        /// Which header produced `retry_after`.
        retry_source: Option<RetrySource>,
        /// Provider request id copied from response metadata, when present.
        request_id: Option<String>,
    },

    /// A provider was configured without a model.
    #[error("provider `{0}` is not configured with a model")]
    MissingModel(&'static str),

    /// The request could not be built as specified.
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// The selected provider does not support the requested operation.
    #[error("unsupported operation: {0}")]
    Unsupported(String),

    /// The model requested a tool that was never registered.
    #[error("tool `{0}` was requested by the model but is not registered")]
    ToolNotFound(String),

    /// A registered tool returned an error while executing.
    #[error("tool `{tool}` failed: {message}")]
    ToolFailed {
        /// Name of the tool that failed.
        tool: String,
        /// Error message reported by the tool.
        message: String,
    },

    /// The agent loop exceeded its configured step budget.
    #[error("agent exceeded max_steps={0}")]
    MaxSteps(usize),

    /// A request could not be normalized for the target protocol.
    #[error("normalization error: {0}")]
    Normalize(String),

    /// A retry policy configuration was invalid.
    #[error("invalid retry policy: {0}")]
    InvalidRetryPolicy(String),

    /// An HTTP header value was invalid.
    #[error("invalid HTTP header value: {0}")]
    HeaderValue(#[from] reqwest::header::InvalidHeaderValue),

    /// An HTTP header name was invalid.
    #[error("invalid HTTP header name: {0}")]
    HeaderName(#[from] reqwest::header::InvalidHeaderName),

    /// An attempt was made to override a header the SDK manages itself.
    #[error("protected HTTP header cannot be overridden: {0}")]
    ProtectedHeader(String),

    /// A header value contained non-ASCII characters.
    #[error("HTTP header `{name}` must be ASCII for interoperability")]
    NonAsciiHeaderValue {
        /// Name of the offending header.
        name: String,
    },

    /// An extra body field collided with a canonical SDK field.
    #[error("{protocol} extra body field `{field}` cannot override a canonical SDK field")]
    ReservedExtraBodyField {
        /// The provider protocol that rejected the field.
        protocol: &'static str,
        /// The reserved field name.
        field: String,
    },

    /// An extra query parameter collided with an endpoint-reserved name.
    #[error("{protocol} extra query parameter `{parameter}` is reserved by the endpoint")]
    ReservedExtraQuery {
        /// The provider protocol that rejected the parameter.
        protocol: &'static str,
        /// The reserved parameter name.
        parameter: String,
    },

    /// A base URL or endpoint could not be parsed.
    #[error("invalid endpoint: {0}")]
    InvalidEndpoint(String),

    /// A provider profile configuration was invalid.
    #[error("invalid provider profile: {0}")]
    InvalidProfile(String),
}

struct RedactedBody(usize);

impl fmt::Debug for RedactedBody {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "<redacted {} bytes>", self.0)
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingApiKey(value) => {
                formatter.debug_tuple("MissingApiKey").field(value).finish()
            }
            Self::Http(error) => formatter.debug_tuple("Http").field(error).finish(),
            Self::Cancelled => formatter.write_str("Cancelled"),
            Self::Timeout(message) => formatter.debug_tuple("Timeout").field(message).finish(),
            Self::Json(error) => formatter.debug_tuple("Json").field(error).finish(),
            Self::StreamProtocol(value) => formatter
                .debug_tuple("StreamProtocol")
                .field(value)
                .finish(),
            Self::ProviderStream(value) => formatter
                .debug_tuple("ProviderStream")
                .field(value)
                .finish(),
            Self::Api {
                provider,
                status,
                body,
                retry_after,
                retry_source,
                request_id,
            } => formatter
                .debug_struct("Api")
                .field("provider", provider)
                .field("status", status)
                .field("body", &RedactedBody(body.len()))
                .field("retry_after", retry_after)
                .field("retry_source", retry_source)
                .field("request_id", request_id)
                .finish(),
            Self::MissingModel(value) => {
                formatter.debug_tuple("MissingModel").field(value).finish()
            }
            Self::InvalidRequest(value) => formatter
                .debug_tuple("InvalidRequest")
                .field(value)
                .finish(),
            Self::Unsupported(value) => formatter.debug_tuple("Unsupported").field(value).finish(),
            Self::ToolNotFound(value) => {
                formatter.debug_tuple("ToolNotFound").field(value).finish()
            }
            Self::ToolFailed { tool, message } => formatter
                .debug_struct("ToolFailed")
                .field("tool", tool)
                .field("message", message)
                .finish(),
            Self::MaxSteps(value) => formatter.debug_tuple("MaxSteps").field(value).finish(),
            Self::Normalize(value) => formatter.debug_tuple("Normalize").field(value).finish(),
            Self::InvalidRetryPolicy(value) => formatter
                .debug_tuple("InvalidRetryPolicy")
                .field(value)
                .finish(),
            Self::HeaderValue(error) => formatter.debug_tuple("HeaderValue").field(error).finish(),
            Self::HeaderName(error) => formatter.debug_tuple("HeaderName").field(error).finish(),
            Self::ProtectedHeader(value) => formatter
                .debug_tuple("ProtectedHeader")
                .field(value)
                .finish(),
            Self::NonAsciiHeaderValue { name } => formatter
                .debug_struct("NonAsciiHeaderValue")
                .field("name", name)
                .finish(),
            Self::ReservedExtraBodyField { protocol, field } => formatter
                .debug_struct("ReservedExtraBodyField")
                .field("protocol", protocol)
                .field("field", field)
                .finish(),
            Self::ReservedExtraQuery {
                protocol,
                parameter,
            } => formatter
                .debug_struct("ReservedExtraQuery")
                .field("protocol", protocol)
                .field("parameter", parameter)
                .finish(),
            Self::InvalidEndpoint(value) => formatter
                .debug_tuple("InvalidEndpoint")
                .field(value)
                .finish(),
            Self::InvalidProfile(value) => formatter
                .debug_tuple("InvalidProfile")
                .field(value)
                .finish(),
        }
    }
}

impl Error {
    /// Classify this error into a stable, provider-neutral [`ErrorKind`].
    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::MissingApiKey(_) => ErrorKind::Authentication,
            Self::Http(error) => classify_reqwest_error(error),
            Self::Cancelled => ErrorKind::Cancelled,
            Self::Timeout(_) => ErrorKind::Timeout,
            Self::Json(_) => ErrorKind::Decode,
            Self::StreamProtocol(_) => ErrorKind::StreamProtocol,
            Self::ProviderStream(_) => ErrorKind::ProviderInternal,
            Self::Api { status, body, .. } => classify_api_error(*status, body),
            Self::MissingModel(_) => ErrorKind::Configuration,
            Self::InvalidRequest(_) => ErrorKind::InvalidRequest,
            Self::Unsupported(_) => ErrorKind::Unsupported,
            Self::ToolNotFound(_) | Self::ToolFailed { .. } => ErrorKind::Tool,
            Self::MaxSteps(_) => ErrorKind::MaxSteps,
            Self::Normalize(_) => ErrorKind::Normalization,
            Self::InvalidRetryPolicy(_)
            | Self::HeaderValue(_)
            | Self::HeaderName(_)
            | Self::ProtectedHeader(_)
            | Self::NonAsciiHeaderValue { .. }
            | Self::ReservedExtraBodyField { .. }
            | Self::ReservedExtraQuery { .. }
            | Self::InvalidEndpoint(_)
            | Self::InvalidProfile(_) => ErrorKind::Configuration,
        }
    }

    /// Provider HTTP status, when this error came from an API response.
    pub fn status(&self) -> Option<u16> {
        match self {
            Self::Api { status, .. } => Some(*status),
            _ => None,
        }
    }

    /// Provider name, when this error came from an API response.
    pub fn provider(&self) -> Option<&str> {
        match self {
            Self::Api { provider, .. } => Some(provider),
            _ => None,
        }
    }

    /// Provider request id, when the response exposed one.
    pub fn request_id(&self) -> Option<&str> {
        match self {
            Self::Api { request_id, .. } => request_id.as_deref(),
            _ => None,
        }
    }

    /// Return the raw provider error body.
    ///
    /// This is intentionally opt-in because provider errors can echo request
    /// content, account data, or other sensitive values. Prefer
    /// [`Self::body_bytes`], [`Self::status`], and [`Self::request_id`] for
    /// logs and metrics.
    pub fn raw_body(&self) -> Option<&str> {
        match self {
            Self::Api { body, .. } => Some(body),
            _ => None,
        }
    }

    /// Decoded provider error-body size in bytes, without exposing the body.
    pub fn body_bytes(&self) -> Option<usize> {
        match self {
            Self::Api { body, .. } => Some(body.len()),
            _ => None,
        }
    }

    /// Parse a safe, provider-neutral summary of a structured API error.
    ///
    /// Provider messages are never included. Codes, types, and status tokens
    /// are only returned when they are short, ASCII machine tokens.
    pub fn provider_error(&self) -> Option<ProviderErrorInfo> {
        match self {
            Self::Api { body, .. } => Some(parse_provider_error(body)),
            _ => None,
        }
    }

    /// Return the parsed `Retry-After` duration, if this error came from an
    /// HTTP response that supplied one.
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::Api { retry_after, .. } => *retry_after,
            _ => None,
        }
    }

    /// Return which response header produced the retry delay.
    pub fn retry_source(&self) -> Option<RetrySource> {
        match self {
            Self::Api { retry_source, .. } => *retry_source,
            _ => None,
        }
    }

    /// Whether this error is normally safe to retry.
    ///
    /// This is a conservative classification. Callers should still avoid
    /// retrying non-idempotent operations unless they have an idempotency key.
    pub fn is_retryable(&self) -> bool {
        self.kind().is_retryable()
    }
}

fn parse_provider_error(body: &str) -> ProviderErrorInfo {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(body) else {
        return ProviderErrorInfo {
            message_bytes: Some(body.len()),
            ..ProviderErrorInfo::default()
        };
    };

    let top = value.as_object();
    let nested = top
        .and_then(|object| object.get("error"))
        .and_then(serde_json::Value::as_object);
    let error_value = top.and_then(|object| object.get("error"));

    let code = nested
        .and_then(|object| object.get("code"))
        .or_else(|| top.and_then(|object| object.get("error_code")))
        .or_else(|| top.and_then(|object| object.get("code")))
        .and_then(provider_token);
    let error_type = nested
        .and_then(|object| object.get("type"))
        .or_else(|| top.and_then(|object| object.get("type")))
        .and_then(provider_token);
    let status = nested
        .and_then(|object| object.get("status"))
        .or_else(|| top.and_then(|object| object.get("status")))
        .and_then(provider_token);
    let message = nested
        .and_then(|object| object.get("message"))
        .or_else(|| top.and_then(|object| object.get("message")))
        .or_else(|| top.and_then(|object| object.get("error_msg")))
        .or_else(|| top.and_then(|object| object.get("error_description")))
        .or(error_value);
    let message_bytes = match message {
        Some(serde_json::Value::String(message)) => Some(message.len()),
        _ => None,
    };
    let details_count = nested
        .and_then(|object| object.get("details"))
        .or_else(|| top.and_then(|object| object.get("details")))
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);

    ProviderErrorInfo {
        code,
        error_type,
        status,
        message_bytes,
        details_count,
    }
}

fn provider_token(value: &serde_json::Value) -> Option<String> {
    let token = match value {
        serde_json::Value::String(token) => token.trim(),
        serde_json::Value::Number(token) => return Some(token.to_string()),
        _ => return None,
    };
    if token.is_empty() || token.len() > 96 {
        return None;
    }
    if !token.bytes().all(|byte| {
        byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b':' | b'/')
    }) {
        return None;
    }
    Some(token.to_string())
}

fn classify_api_error(status: u16, body: &str) -> ErrorKind {
    let info = parse_provider_error(body);
    let tokens = [
        info.code.as_deref(),
        info.error_type.as_deref(),
        info.status.as_deref(),
    ];
    for token in tokens.into_iter().flatten() {
        match token.to_ascii_lowercase().as_str() {
            "context_length_exceeded"
            | "context_length_exceeded_error"
            | "context_too_long"
            | "prompt_too_long"
            | "max_tokens_exceeded" => return ErrorKind::ContextLengthExceeded,
            "content_filter"
            | "content_filtered"
            | "content_policy_violation"
            | "safety"
            | "safety_blocked" => return ErrorKind::ContentFiltered,
            _ => {}
        }
    }
    classify_status(status)
}

fn classify_status(status: u16) -> ErrorKind {
    match status {
        401 => ErrorKind::Authentication,
        403 => ErrorKind::PermissionDenied,
        404 => ErrorKind::NotFound,
        408 => ErrorKind::Timeout,
        425 => ErrorKind::Overloaded,
        429 => ErrorKind::RateLimited,
        400 | 409 | 413 | 415 | 416 | 422 => ErrorKind::InvalidRequest,
        500..=599 => ErrorKind::ProviderInternal,
        _ => ErrorKind::Unknown,
    }
}

fn classify_reqwest_error(error: &reqwest::Error) -> ErrorKind {
    if error.is_timeout() {
        ErrorKind::Timeout
    } else if error.is_connect() || error.is_request() || error.is_body() {
        ErrorKind::Connection
    } else if error.is_decode() {
        ErrorKind::Decode
    } else if error.is_builder() {
        ErrorKind::InvalidRequest
    } else {
        ErrorKind::Unknown
    }
}
