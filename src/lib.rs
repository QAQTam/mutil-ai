//! A small, provider-neutral AI SDK designed for learning.
//!
//! The important idea in this crate is the split between:
//!
//! 1. an **internal message model** that can represent messy agent histories;
//! 2. a **normalization layer** that turns that model into a clean, strict
//!    provider-facing conversation;
//! 3. provider adapters that only serialize the normalized form.
//!
//! This keeps role conversion and tool-call repair in one place.

mod adapter;
mod agent;
mod audit;
mod error;
mod headers;
mod normalize;
mod profile;
mod retry;
mod sse;
mod stream;
mod tool;
mod types;

pub use adapter::{
    Anthropic, AnthropicMessages, EndpointAdapter, Gemini, GeminiGenerateContent, ModelAdapter,
    OpenAI, OpenAIChat, OpenAICompatible, OpenAIResponses,
};
pub use agent::{Agent, AgentBuilder, AgentReply, complete_once};
pub use audit::{
    AuditConfig, AuditEvent, AuditOutcome, AuditSink, AuditStats, AuditStatsSnapshot, ByteCounts,
    FirstTokenKind, ProfileAuditSnapshot, RequestTiming, bounded_audit_channel,
};
pub use error::{Error, ErrorKind, ProviderErrorInfo, Result};
pub use headers::{
    ClientInfo, HeaderInjector, HeaderPolicy, RequestContext, RequestOptions, SDK_USER_AGENT,
    TransportConfig, apply_headers,
};
pub use normalize::{
    ExternalRole, MissingToolResultPolicy, NormalizeAction, NormalizeOptions, NormalizeReport,
    NormalizeStats, NormalizedChat, NormalizedMessage, OrphanToolResultPolicy, Protocol, normalize,
    normalize_with_options,
};
pub use profile::{
    AuthStyle, Capabilities, EffortMapping, EndpointSpec, MaxTokensSemantics, ModelMatcher,
    ModelProfile, ProfileId, ProfileRegistry, ProfileSelector, ProtocolSurface, ProviderProfile,
    ReasoningAliases, ReasoningCapabilities, ReasoningProfile, ReasoningReplayPolicy,
    RequestProfile, StreamCapabilities, StreamProfile, StreamTerminal, ThinkingRequestProfile,
    ToolCallIdPolicy, ToolCapabilities, ToolProfile, UsagePlacement, UsageProfile,
};
pub use retry::{
    RetryDirective, RetryPolicy, RetryProvider, RetrySource, parse_retry_after,
    parse_retry_after_at, parse_retry_headers, parse_retry_headers_at, retry_async,
};
pub use sse::{SseError, SseEvent, SseMessage, SseParser, SseUtf8Policy};
pub use stream::{ModelStream, StreamEvent, StreamReconnectPolicy, collect_stream, next_event};
pub use tool::{FunctionTool, Tool, ToolRegistry, tool_fn};
pub use types::{
    ChatRequest, ChatResponse, ImageDetail, ImageUrl, Message, Part, ProviderState,
    ProviderStateFormat, Reasoning, ReasoningConfig, ReasoningEffort, ReasoningKind, ReasoningMode,
    ReasoningSummary, ResponseFormat, ResponseMetadata, Role, ToolCall, ToolChoice, ToolResult,
    ToolSpec, Usage,
};
