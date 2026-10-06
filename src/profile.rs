use std::fmt;
use std::sync::Arc;

use reqwest::header::{HeaderMap, HeaderName};
use serde_json::{Map, Value};

/// Protocol surface of an endpoint, re-exported from the normalize module.
///
/// See [`crate::normalize::Protocol`] for the available values.
pub use crate::normalize::Protocol as ProtocolSurface;
use crate::normalize::SystemPlacement;

/// Stable identifier for a provider profile.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProfileId(String);

impl ProfileId {
    /// Creates a profile identifier from any string-like value.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Returns the identifier as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for ProfileId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for ProfileId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl fmt::Display for ProfileId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// How the generic adapter authenticates a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthStyle {
    /// `Authorization: Bearer <key>`.
    Bearer,
    /// A provider-specific API-key header such as `x-api-key`.
    XApiKey { header: HeaderName },
    /// A query parameter such as Gemini's `key=...`.
    QueryKey { parameter: String },
    /// A custom header, optionally with a value prefix such as `Token `.
    Custom {
        header: HeaderName,
        value_prefix: Option<String>,
    },
    /// No authentication. Intended for local gateways and test servers.
    None,
}

/// Which profile implementation should be used for an endpoint.
#[derive(Debug, Clone, Default)]
pub enum ProfileSelector {
    /// Protocol defaults only.
    #[default]
    Generic,
    /// Provider-specific behavior supplied by the application.
    Custom(Arc<ProviderProfile>),
    /// A future registry lookup. The registry is intentionally not guessed
    /// from a base URL.
    Builtin(ProfileId),
}

/// Explicit description of one HTTP endpoint.
///
/// A protocol surface is always chosen by the caller. The SDK never infers it
/// from the base URL.
#[derive(Debug, Clone)]
pub struct EndpointSpec {
    /// Protocol surface used to encode requests and decode responses.
    pub protocol: ProtocolSurface,
    /// Provider base URL, for example `https://api.openai.com/v1`.
    pub base_url: String,
    /// Path appended to [`Self::base_url`]; may contain a `{model}` placeholder.
    pub path: String,
    /// Optional path used for streaming requests; falls back to [`Self::path`].
    pub stream_path: Option<String>,
    /// Authentication scheme applied to every request.
    pub auth: AuthStyle,
    /// Which profile implementation supplies provider-specific behavior.
    pub profile: ProfileSelector,
}

impl EndpointSpec {
    /// Creates a spec with no streaming path and generic profile behavior.
    pub fn new(
        protocol: ProtocolSurface,
        base_url: impl Into<String>,
        path: impl Into<String>,
        auth: AuthStyle,
    ) -> Self {
        Self {
            protocol,
            base_url: base_url.into(),
            path: path.into(),
            stream_path: None,
            auth,
            profile: ProfileSelector::Generic,
        }
    }

    /// Creates a spec for OpenAI Chat Completions at `/chat/completions`.
    pub fn openai_chat(base_url: impl Into<String>) -> Self {
        Self::new(
            ProtocolSurface::OpenAiChat,
            base_url,
            "/chat/completions",
            AuthStyle::Bearer,
        )
    }

    /// Creates a spec for OpenAI Responses at `/responses`.
    pub fn openai_responses(base_url: impl Into<String>) -> Self {
        Self::new(
            ProtocolSurface::OpenAiResponses,
            base_url,
            "/responses",
            AuthStyle::Bearer,
        )
    }

    /// Creates a spec for Anthropic Messages at `/messages` using `x-api-key`.
    pub fn anthropic_messages(base_url: impl Into<String>) -> Self {
        Self::new(
            ProtocolSurface::AnthropicMessages,
            base_url,
            "/messages",
            AuthStyle::XApiKey {
                header: HeaderName::from_static("x-api-key"),
            },
        )
    }

    /// Creates a spec for Gemini `generateContent` with a dedicated streaming
    /// path and `key` query-parameter authentication.
    pub fn gemini_generate_content(base_url: impl Into<String>) -> Self {
        let mut endpoint = Self::new(
            ProtocolSurface::GeminiGenerateContent,
            base_url,
            "/models/{model}:generateContent",
            AuthStyle::QueryKey {
                parameter: "key".to_string(),
            },
        );
        endpoint.stream_path = Some("/models/{model}:streamGenerateContent".to_string());
        endpoint
    }

    /// Sets the path used for streaming requests.
    pub fn stream_path(mut self, path: impl Into<String>) -> Self {
        self.stream_path = Some(path.into());
        self
    }

    /// Attaches a custom [`ProviderProfile`] to this endpoint.
    pub fn profile(mut self, profile: ProviderProfile) -> Self {
        self.profile = ProfileSelector::Custom(Arc::new(profile));
        self
    }

    /// Sets the profile selector, including registry-backed
    /// [`ProfileSelector::Builtin`].
    pub fn profile_selector(mut self, profile: ProfileSelector) -> Self {
        self.profile = profile;
        self
    }

    /// Build the non-streaming endpoint URL after substituting `{model}`.
    pub fn url(&self, model: &str) -> crate::Result<String> {
        self.url_for(model, false)
    }

    /// Build the endpoint URL for either the normal or streaming operation.
    pub fn url_for(&self, model: &str, streaming: bool) -> crate::Result<String> {
        let path = if streaming {
            self.stream_path.as_deref().unwrap_or(&self.path)
        } else {
            &self.path
        };
        if self.base_url.trim().is_empty() {
            return Err(crate::Error::InvalidEndpoint(
                "base_url cannot be empty".to_string(),
            ));
        }
        if path.trim().is_empty() {
            return Err(crate::Error::InvalidEndpoint(
                "path cannot be empty".to_string(),
            ));
        }

        let base = self.base_url.trim_end_matches('/');
        let path = path.trim_start_matches('/').replace("{model}", model);
        Ok(format!("{base}/{path}"))
    }

    /// Returns the attached custom profile, if any.
    pub fn provider_profile(&self) -> Option<&ProviderProfile> {
        match &self.profile {
            ProfileSelector::Custom(profile) => Some(profile),
            ProfileSelector::Generic | ProfileSelector::Builtin(_) => None,
        }
    }

    /// Resolve custom or built-in behavior without changing protocol/path.
    pub fn resolved_profile(&self) -> Option<ProviderProfile> {
        match &self.profile {
            ProfileSelector::Generic => None,
            ProfileSelector::Custom(profile) => Some(profile.as_ref().clone()),
            ProfileSelector::Builtin(id) => ProfileRegistry::new().resolve(id),
        }
    }
}

/// Normalization behavior attached to a provider profile.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NormalizeProfile {
    /// How system/developer messages are placed.
    ///
    /// `FirstToTopRestInPlace` is honored only by the OpenAI Chat Completions
    /// and OpenAI Responses adapters; Anthropic and Gemini always merge
    /// system text into the top-level entry.
    pub system_placement: SystemPlacement,
}

/// Provider-level defaults and behavior switches.
#[derive(Debug, Clone)]
pub struct ProviderProfile {
    /// Stable identifier, also used for built-in registry lookup.
    pub id: ProfileId,
    /// Extra body, header, and query defaults applied to every request.
    pub request: RequestProfile,
    /// Typed request switches shared across OpenAI-compatible APIs.
    pub request_options: ProviderRequestOptions,
    /// Normalization behavior for requests sent to this provider.
    pub normalize: NormalizeProfile,
    /// Reasoning field names, replay policy, and thinking request controls.
    pub reasoning: ReasoningProfile,
    /// Tool-call encoding and ID handling behavior.
    pub tools: ToolProfile,
    /// Streaming terminal event and usage delivery behavior.
    pub stream: StreamProfile,
    /// Where usage information appears in streamed responses.
    pub usage: UsageProfile,
    /// Feature switches used to reject unsupported requests early.
    pub capabilities: Capabilities,
    /// Which request field carries the output token limit.
    pub max_tokens_semantics: MaxTokensSemantics,
}

impl ProviderProfile {
    /// Returns the built-in profile for a provider id such as `"deepseek"`.
    ///
    /// Common aliases are accepted, for example `"moonshot"` for Kimi or
    /// `"bailian"` for Qwen.
    pub fn builtin(id: &str) -> Option<Self> {
        match id.trim().to_ascii_lowercase().as_str() {
            "deepseek" => Some(Self::deepseek_compatible()),
            "qwen" | "bailian" | "dashscope" => Some(Self::qwen_compatible()),
            "kimi" | "moonshot" => Some(Self::kimi_compatible()),
            "glm" | "zhipu" | "bigmodel" => Some(Self::glm_compatible()),
            "doubao" | "ark" | "volcengine" => Some(Self::doubao_compatible()),
            "openai" => Some(Self::new("openai")),
            "anthropic" => Some(Self::new("anthropic")),
            "gemini" | "google" => Some(Self::new("gemini")),
            "openai-compatible" => Some(Self::new("openai-compatible")),
            _ => None,
        }
    }

    /// Creates a profile with default behavior for the given identifier.
    pub fn new(id: impl Into<ProfileId>) -> Self {
        Self {
            id: id.into(),
            request: RequestProfile::default(),
            request_options: ProviderRequestOptions::default(),
            normalize: NormalizeProfile::default(),
            reasoning: ReasoningProfile::default(),
            tools: ToolProfile::default(),
            stream: StreamProfile::default(),
            usage: UsageProfile::default(),
            capabilities: Capabilities::default(),
            max_tokens_semantics: MaxTokensSemantics::MaxCompletionTokens,
        }
    }

    /// Common OpenAI-compatible behavior documented by DeepSeek.
    ///
    /// This is a behavior profile, not an endpoint. The caller still provides
    /// the protocol, base URL, path, and auth through [`EndpointSpec`].
    pub fn deepseek_compatible() -> Self {
        let mut profile = Self::new("deepseek");
        profile.reasoning.aliases = ReasoningAliases::empty()
            .text("reasoning_content")
            .replay("reasoning_content");
        profile.reasoning.replay = ReasoningReplayPolicy::SameProvider;
        profile.reasoning.thinking = ThinkingRequestProfile::Composite(vec![
            ThinkingRequestProfile::ThinkingObject,
            ThinkingRequestProfile::MappedEffort {
                field: "reasoning_effort".to_string(),
                mapping: EffortMapping::deepseek(),
            },
        ]);
        profile
    }

    /// Common OpenAI-compatible behavior documented by Qwen / Bailian.
    pub fn qwen_compatible() -> Self {
        let mut profile = Self::new("qwen");
        profile.reasoning.aliases = ReasoningAliases::empty()
            .text("reasoning_content")
            .replay("reasoning_content");
        profile.reasoning.replay = ReasoningReplayPolicy::SameProvider;
        profile.reasoning.thinking = ThinkingRequestProfile::Composite(vec![
            ThinkingRequestProfile::EnabledFlag("enable_thinking".to_string()),
            ThinkingRequestProfile::BudgetTokens("thinking_budget".to_string()),
        ]);
        profile
    }

    /// Common OpenAI-compatible behavior documented by Moonshot Kimi.
    ///
    /// Kimi K3 and K2.x use different thinking controls. Add a matching
    /// [`ModelProfile`] for those model-specific fields.
    pub fn kimi_compatible() -> Self {
        let mut profile = Self::new("kimi");
        profile.reasoning.aliases = ReasoningAliases::empty()
            .text("reasoning_content")
            .replay("reasoning_content");
        profile.reasoning.replay = ReasoningReplayPolicy::SameProvider;
        profile
    }

    /// Common OpenAI-compatible behavior documented by Zhipu GLM.
    pub fn glm_compatible() -> Self {
        let mut profile = Self::new("glm");
        profile.reasoning.aliases = ReasoningAliases::empty()
            .text("reasoning_content")
            .replay("reasoning_content");
        profile.reasoning.replay = ReasoningReplayPolicy::SameProvider;
        profile.reasoning.thinking = ThinkingRequestProfile::ThinkingObject;
        profile
    }

    /// Conservative OpenAI-compatible behavior for Volcengine Ark / Doubao.
    ///
    /// Reasoning state is replayed only for tool-call turns because Ark uses
    /// `encrypted_content` to preserve tool-call continuity.
    pub fn doubao_compatible() -> Self {
        let mut profile = Self::new("doubao");
        profile.reasoning.aliases = ReasoningAliases::empty()
            .text("reasoning_content")
            .replay("reasoning_content")
            .encrypted("encrypted_content");
        profile.reasoning.replay = ReasoningReplayPolicy::RequiredForToolCalls;
        profile.reasoning.thinking = ThinkingRequestProfile::ThinkingObject;
        profile
    }
}

/// Explicit registry for built-in behavior profiles.
///
/// Registry lookup never changes the protocol or endpoint. Those remain owned
/// by [`EndpointSpec`].
#[derive(Debug, Clone, Copy, Default)]
pub struct ProfileRegistry;

impl ProfileRegistry {
    /// Creates an empty registry handle. Lookup uses built-in tables only.
    pub const fn new() -> Self {
        Self
    }

    /// Resolves a built-in [`ProviderProfile`] by id, or `None` if unknown.
    pub fn resolve(&self, id: &ProfileId) -> Option<ProviderProfile> {
        ProviderProfile::builtin(id.as_str())
    }

    /// Lists every id accepted by [`Self::resolve`], including aliases.
    pub const fn builtin_ids(&self) -> &'static [&'static str] {
        &[
            "deepseek",
            "qwen",
            "bailian",
            "dashscope",
            "kimi",
            "moonshot",
            "glm",
            "zhipu",
            "bigmodel",
            "doubao",
            "ark",
            "volcengine",
            "openai",
            "anthropic",
            "gemini",
            "google",
            "openai-compatible",
        ]
    }
}

/// How an assistant tool-call message should encode empty content.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ToolCallContentMode {
    /// Preserve the adapter's documented default.
    #[default]
    Auto,
    /// Encode empty content as an explicit `null`.
    Null,
    /// Omit the `content` field entirely.
    Omit,
    /// Encode empty content as an empty string.
    Empty,
}

/// Typed provider request switches that are common across compatible APIs.
///
/// `None` means "inherit the next lower-priority layer". Request-level values
/// override provider-profile defaults.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProviderRequestOptions {
    /// How assistant tool-call messages encode empty content.
    pub tool_call_content: Option<ToolCallContentMode>,
    /// Whether provider-specific parameters must be present in requests.
    pub require_provider_parameters: Option<bool>,
    /// Enables sampling for providers that default to greedy decoding (Qwen).
    pub do_sample: Option<bool>,
    /// Requests a trailing usage chunk via `stream_options.include_usage`.
    pub include_stream_usage: Option<bool>,
    /// Cache key routed to provider-side prompt caching.
    pub prompt_cache_key: Option<String>,
    /// End-user identifier forwarded to the provider.
    pub user: Option<String>,
}

/// Request fields that are applied before per-call `extra_*` values.
#[derive(Debug, Clone, Default)]
pub struct RequestProfile {
    /// JSON body fields merged into every request.
    pub extra_body: Map<String, Value>,
    /// HTTP headers added to every request.
    pub extra_headers: HeaderMap,
    /// Query parameters appended to every request URL.
    pub extra_query: Vec<(String, String)>,
}

impl RequestProfile {
    /// Adds a JSON body field, returning the profile for chaining.
    pub fn body(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.extra_body.insert(key.into(), value.into());
        self
    }

    /// Adds an HTTP header, returning the profile for chaining.
    pub fn header(mut self, name: HeaderName, value: reqwest::header::HeaderValue) -> Self {
        self.extra_headers.insert(name, value);
        self
    }

    /// Appends a query parameter, returning the profile for chaining.
    pub fn query(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_query.push((name.into(), value.into()));
        self
    }
}

/// Names observed in provider responses for each neutral reasoning category.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningAliases {
    /// Response field names holding free-form reasoning text.
    pub response_text: Vec<String>,
    /// Response field names holding reasoning summaries.
    pub response_summary: Vec<String>,
    /// Request field names used to replay reasoning into follow-up turns.
    pub request_replay: Vec<String>,
    /// Request field names holding opaque encrypted reasoning blobs.
    pub encrypted: Vec<String>,
    /// Request field names holding reasoning signatures.
    pub signature: Vec<String>,
}

impl Default for ReasoningAliases {
    fn default() -> Self {
        Self {
            response_text: vec!["reasoning_content".to_string(), "reasoning".to_string()],
            response_summary: Vec::new(),
            request_replay: vec![
                "reasoning_content".to_string(),
                "reasoning".to_string(),
                "reasoning_details".to_string(),
            ],
            encrypted: vec!["reasoning_details".to_string()],
            signature: Vec::new(),
        }
    }
}

impl ReasoningAliases {
    /// Creates an alias set with no recognized field names.
    pub fn empty() -> Self {
        Self {
            response_text: Vec::new(),
            response_summary: Vec::new(),
            request_replay: Vec::new(),
            encrypted: Vec::new(),
            signature: Vec::new(),
        }
    }

    /// Returns `true` when no field names are registered in any category.
    pub fn is_empty(&self) -> bool {
        self.response_text.is_empty()
            && self.response_summary.is_empty()
            && self.request_replay.is_empty()
            && self.encrypted.is_empty()
            && self.signature.is_empty()
    }

    /// Adds a response field name for free-form reasoning text.
    pub fn text(mut self, field: impl Into<String>) -> Self {
        self.response_text.push(field.into());
        self
    }

    /// Adds a response field name for reasoning summaries.
    pub fn summary(mut self, field: impl Into<String>) -> Self {
        self.response_summary.push(field.into());
        self
    }

    /// Adds a request field name used to replay reasoning.
    pub fn replay(mut self, field: impl Into<String>) -> Self {
        self.request_replay.push(field.into());
        self
    }

    /// Adds a request field name for encrypted reasoning blobs.
    pub fn encrypted(mut self, field: impl Into<String>) -> Self {
        self.encrypted.push(field.into());
        self
    }

    /// Adds a request field name for reasoning signatures.
    pub fn signature(mut self, field: impl Into<String>) -> Self {
        self.signature.push(field.into());
        self
    }
}

/// Whether and how reasoning state captured from a response is replayed on
/// follow-up requests.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ReasoningReplayPolicy {
    /// Reasoning state is never sent back to the provider.
    Never,
    /// Replays reasoning fields only when the follow-up request targets the
    /// same provider.
    #[default]
    SameProvider,
    /// Keeps reasoning fields inside the conversation history as-is.
    PreserveInHistory,
    /// Replays reasoning only on tool-call turns, where continuity is
    /// required (for example Volcengine Ark's `encrypted_content`).
    RequiredForToolCalls,
    /// Replays only opaque encrypted reasoning blobs, never plaintext.
    EncryptedOnly,
    /// Replay behavior is decided by model-specific profiles.
    ModelDefined,
}

/// Reasoning-related defaults for a provider.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReasoningProfile {
    /// Field names recognized in responses and replayed in requests.
    pub aliases: ReasoningAliases,
    /// Whether and how reasoning state is replayed across turns.
    pub replay: ReasoningReplayPolicy,
    /// How thinking controls are encoded into requests.
    pub thinking: ThinkingRequestProfile,
}

/// How tool-call identifiers are handled when translating between protocols.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ToolCallIdPolicy {
    /// Keeps the provider's tool-call ID unchanged.
    #[default]
    Preserve,
    /// Generates a new ID when the upstream format is not representable.
    Synthesize,
    /// Encodes IDs as `name`/`id` pairs for providers without ID support.
    NamePair,
}

/// Tool-call encoding behavior for a provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolProfile {
    /// How tool-call IDs are translated between protocols.
    pub call_id_policy: ToolCallIdPolicy,
    /// Whether the provider accepts parallel tool calls.
    pub parallel_tool_calls: bool,
    /// Whether tool-call arguments are streamed incrementally.
    pub stream_arguments: bool,
}

impl Default for ToolProfile {
    fn default() -> Self {
        Self {
            call_id_policy: ToolCallIdPolicy::Preserve,
            parallel_tool_calls: true,
            stream_arguments: true,
        }
    }
}

/// The event that marks the end of a provider's response stream.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StreamTerminal {
    /// An SSE `data: [DONE]` sentinel (OpenAI-style).
    #[default]
    DataDone,
    /// A `response.completed` event (OpenAI Responses-style).
    ResponseCompleted,
    /// A `message_stop` event (Anthropic-style).
    MessageStop,
    /// End of the stream without an explicit terminal event (Gemini-style).
    Eof,
}

/// Where usage (token counts) appears in a streamed response.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum UsagePlacement {
    /// Usage arrives only on the final chunk.
    #[default]
    LastChunk,
    /// Usage is repeated on every chunk.
    EveryChunk,
    /// Usage arrives in a dedicated event or chunk.
    UsageEvent,
    /// Streaming never reports usage.
    None,
}

/// Streaming behavior for a provider.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StreamProfile {
    /// Event that marks the end of the stream.
    pub terminal: StreamTerminal,
    /// Where usage appears inside the stream.
    pub usage: UsagePlacement,
}

/// Usage reporting behavior for a provider.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UsageProfile {
    /// Where usage appears in streamed responses.
    pub placement: UsagePlacement,
}

/// Capability switches used to reject requests before they reach a provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capabilities {
    /// Reasoning-related feature switches.
    pub reasoning: ReasoningCapabilities,
    /// Tool-call feature switches.
    pub tools: ToolCapabilities,
    /// Streaming feature switches.
    pub streaming: StreamCapabilities,
    /// Whether the provider supports a tool-choice control.
    pub tool_choice: bool,
    /// Whether the provider supports structured (JSON schema) output.
    pub structured_output: bool,
    /// Whether the provider supports a deterministic `seed` parameter.
    pub seed: bool,
    /// Whether the provider supports stop sequences.
    pub stop: bool,
    /// Whether the provider supports multimodal (image/audio) input.
    pub multimodal: bool,
    /// Whether conversation state is stored server-side between requests.
    pub server_side_state: bool,
}

impl Default for Capabilities {
    fn default() -> Self {
        Self {
            reasoning: ReasoningCapabilities::default(),
            tools: ToolCapabilities::default(),
            streaming: StreamCapabilities::default(),
            tool_choice: true,
            structured_output: true,
            seed: true,
            stop: true,
            multimodal: true,
            server_side_state: false,
        }
    }
}

/// Reasoning feature switches for a provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReasoningCapabilities {
    /// Whether reasoning is supported at all.
    pub supported: bool,
    /// Whether free-form reasoning text is exposed.
    pub text: bool,
    /// Whether reasoning summaries are exposed.
    pub summary: bool,
    /// Whether encrypted reasoning blobs are supported.
    pub encrypted: bool,
    /// Whether reasoning can be replayed on follow-up requests.
    pub replay: bool,
}

impl Default for ReasoningCapabilities {
    fn default() -> Self {
        Self {
            supported: true,
            text: true,
            summary: true,
            encrypted: true,
            replay: true,
        }
    }
}

/// Tool-call feature switches for a provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolCapabilities {
    /// Whether tool calling is supported at all.
    pub supported: bool,
    /// Whether parallel tool calls are supported.
    pub parallel: bool,
    /// Whether tool-call arguments can be streamed.
    pub streaming: bool,
}

impl Default for ToolCapabilities {
    fn default() -> Self {
        Self {
            supported: true,
            parallel: true,
            streaming: true,
        }
    }
}

/// Streaming feature switches for a provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamCapabilities {
    /// Whether streaming is supported at all.
    pub supported: bool,
    /// Whether usage can be reported inside a stream.
    pub usage: bool,
}

impl Default for StreamCapabilities {
    fn default() -> Self {
        Self {
            supported: true,
            usage: true,
        }
    }
}

/// Model-level policy selected after the endpoint protocol is known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelProfile {
    /// Selects which model names this profile applies to.
    pub matcher: ModelMatcher,
    /// How thinking controls are encoded in requests.
    pub thinking: ThinkingRequestProfile,
    /// Model-specific reasoning response and replay field names.
    pub reasoning_aliases: ReasoningAliases,
    /// Overrides the provider-level replay policy when set.
    pub replay: Option<ReasoningReplayPolicy>,
    /// Overrides which request field carries the output token limit.
    pub max_tokens_semantics: MaxTokensSemantics,
    /// Whether `function.arguments` is streamed incrementally.
    pub stream_function_arguments: bool,
    /// Overrides provider capabilities for this model when set.
    pub capabilities: Option<Capabilities>,
}

impl Default for ModelProfile {
    fn default() -> Self {
        Self {
            matcher: ModelMatcher::Any,
            thinking: ThinkingRequestProfile::None,
            reasoning_aliases: ReasoningAliases::empty(),
            replay: None,
            max_tokens_semantics: MaxTokensSemantics::MaxOutputTokens,
            stream_function_arguments: true,
            capabilities: None,
        }
    }
}

impl ModelProfile {
    /// Kimi K3 reasoning controls.
    pub fn kimi_k3() -> Self {
        Self {
            matcher: ModelMatcher::Prefix("kimi-k3".to_string()),
            thinking: ThinkingRequestProfile::Effort("reasoning_effort".to_string()),
            reasoning_aliases: ReasoningAliases::empty()
                .text("reasoning_content")
                .replay("reasoning_content"),
            replay: Some(ReasoningReplayPolicy::SameProvider),
            ..Self::default()
        }
    }

    /// Kimi K2.6 reasoning controls, including preserved thinking history.
    pub fn kimi_k2_6() -> Self {
        Self {
            matcher: ModelMatcher::Prefix("kimi-k2.6".to_string()),
            thinking: ThinkingRequestProfile::Composite(vec![
                ThinkingRequestProfile::ThinkingObject,
                ThinkingRequestProfile::Field {
                    path: "thinking.keep".to_string(),
                    value: Value::String("all".to_string()),
                },
            ]),
            reasoning_aliases: ReasoningAliases::empty()
                .text("reasoning_content")
                .replay("reasoning_content"),
            replay: Some(ReasoningReplayPolicy::SameProvider),
            ..Self::default()
        }
    }

    /// Returns `true` when this profile applies to the given model name.
    pub fn matches(&self, model: &str) -> bool {
        self.matcher.matches(model)
    }
}

/// Selects which model names a [`ModelProfile`] applies to.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ModelMatcher {
    /// Matches every model.
    #[default]
    Any,
    /// Matches exactly one model name.
    Exact(String),
    /// Matches any model name starting with the given prefix.
    Prefix(String),
}

impl ModelMatcher {
    /// Returns `true` when this matcher selects the given model name.
    pub fn matches(&self, model: &str) -> bool {
        match self {
            Self::Any => true,
            Self::Exact(expected) => model == expected,
            Self::Prefix(prefix) => model.starts_with(prefix),
        }
    }
}

/// How a neutral thinking configuration is encoded into a provider request.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ThinkingRequestProfile {
    /// No thinking controls are sent.
    #[default]
    None,
    /// Sends a boolean flag such as `enable_thinking`.
    EnabledFlag(String),
    /// Sends a `thinking` object such as `{"type": "enabled"}`.
    ThinkingObject,
    /// Sends neutral effort values under the given field name.
    Effort(String),
    /// Sends effort values translated through an [`EffortMapping`].
    MappedEffort {
        /// Request field that receives the mapped effort value.
        field: String,
        /// Translation table from neutral effort to provider values.
        mapping: EffortMapping,
    },
    /// Sends a numeric thinking token budget under the given field name.
    BudgetTokens(String),
    /// Set a static or nested JSON field, for example `thinking.keep`.
    ///
    /// Static fields are applied even when the request has no
    /// `ReasoningConfig`; dynamic thinking controls are still ignored unless a
    /// config is present.
    Field {
        path: String,
        value: Value,
    },
    /// Applies several encodings together, in order.
    Composite(Vec<ThinkingRequestProfile>),
}

/// Provider-specific translation for neutral reasoning effort values.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EffortMapping {
    /// Provider value for the neutral `none` effort; `None` means unsupported.
    pub none: Option<String>,
    /// Provider value for the neutral `minimal` effort; `None` means unsupported.
    pub minimal: Option<String>,
    /// Provider value for the neutral `low` effort; `None` means unsupported.
    pub low: Option<String>,
    /// Provider value for the neutral `medium` effort; `None` means unsupported.
    pub medium: Option<String>,
    /// Provider value for the neutral `high` effort; `None` means unsupported.
    pub high: Option<String>,
    /// Provider value for the neutral `xhigh` effort; `None` means unsupported.
    pub xhigh: Option<String>,
    /// Provider value for the neutral `max` effort; `None` means unsupported.
    pub max: Option<String>,
}

impl EffortMapping {
    /// Maps every neutral effort value to its lowercase name unchanged.
    pub fn identity() -> Self {
        Self {
            none: Some("none".to_string()),
            minimal: Some("minimal".to_string()),
            low: Some("low".to_string()),
            medium: Some("medium".to_string()),
            high: Some("high".to_string()),
            xhigh: Some("xhigh".to_string()),
            max: Some("max".to_string()),
        }
    }

    /// DeepSeek's mapping, which collapses most levels into `low`/`high`.
    pub fn deepseek() -> Self {
        Self {
            none: Some("none".to_string()),
            minimal: Some("low".to_string()),
            low: Some("low".to_string()),
            medium: Some("high".to_string()),
            high: Some("high".to_string()),
            xhigh: Some("high".to_string()),
            max: Some("max".to_string()),
        }
    }

    /// Kimi K3's mapping, which supports only `low`, `high`, and `max`.
    pub fn kimi_k3() -> Self {
        Self {
            low: Some("low".to_string()),
            high: Some("high".to_string()),
            max: Some("max".to_string()),
            ..Self::default()
        }
    }

    /// Translates a neutral effort value into the provider's value, or `None`
    /// when that level is unsupported.
    pub fn map(&self, effort: crate::types::ReasoningEffort) -> Option<&str> {
        match effort {
            crate::types::ReasoningEffort::None => self.none.as_deref(),
            crate::types::ReasoningEffort::Minimal => self.minimal.as_deref(),
            crate::types::ReasoningEffort::Low => self.low.as_deref(),
            crate::types::ReasoningEffort::Medium => self.medium.as_deref(),
            crate::types::ReasoningEffort::High => self.high.as_deref(),
            crate::types::ReasoningEffort::XHigh => self.xhigh.as_deref(),
            crate::types::ReasoningEffort::Max => self.max.as_deref(),
        }
    }
}

/// Which request field carries the maximum output token limit.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MaxTokensSemantics {
    /// `max_output_tokens` (OpenAI Responses-style).
    #[default]
    MaxOutputTokens,
    /// `max_tokens` (legacy OpenAI Chat and most compatible APIs).
    MaxTokens,
    /// `max_completion_tokens` (newer OpenAI Chat Completions).
    MaxCompletionTokens,
}
