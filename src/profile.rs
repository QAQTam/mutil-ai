use std::fmt;
use std::sync::Arc;

use reqwest::header::{HeaderMap, HeaderName};
use serde_json::{Map, Value};

pub use crate::normalize::Protocol as ProtocolSurface;

/// Stable identifier for a provider profile.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProfileId(String);

impl ProfileId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

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
    pub protocol: ProtocolSurface,
    pub base_url: String,
    pub path: String,
    pub stream_path: Option<String>,
    pub auth: AuthStyle,
    pub profile: ProfileSelector,
}

impl EndpointSpec {
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

    pub fn openai_chat(base_url: impl Into<String>) -> Self {
        Self::new(
            ProtocolSurface::OpenAiChat,
            base_url,
            "/chat/completions",
            AuthStyle::Bearer,
        )
    }

    pub fn openai_responses(base_url: impl Into<String>) -> Self {
        Self::new(
            ProtocolSurface::OpenAiResponses,
            base_url,
            "/responses",
            AuthStyle::Bearer,
        )
    }

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

    pub fn stream_path(mut self, path: impl Into<String>) -> Self {
        self.stream_path = Some(path.into());
        self
    }

    pub fn profile(mut self, profile: ProviderProfile) -> Self {
        self.profile = ProfileSelector::Custom(Arc::new(profile));
        self
    }

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

/// Provider-level defaults and behavior switches.
#[derive(Debug, Clone)]
pub struct ProviderProfile {
    pub id: ProfileId,
    pub request: RequestProfile,
    pub request_options: ProviderRequestOptions,
    pub reasoning: ReasoningProfile,
    pub tools: ToolProfile,
    pub stream: StreamProfile,
    pub usage: UsageProfile,
    pub capabilities: Capabilities,
    pub max_tokens_semantics: MaxTokensSemantics,
}

impl ProviderProfile {
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

    pub fn new(id: impl Into<ProfileId>) -> Self {
        Self {
            id: id.into(),
            request: RequestProfile::default(),
            request_options: ProviderRequestOptions::default(),
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
    pub const fn new() -> Self {
        Self
    }

    pub fn resolve(&self, id: &ProfileId) -> Option<ProviderProfile> {
        ProviderProfile::builtin(id.as_str())
    }

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
    Null,
    Omit,
    Empty,
}

/// Typed provider request switches that are common across compatible APIs.
///
/// `None` means "inherit the next lower-priority layer". Request-level values
/// override provider-profile defaults.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProviderRequestOptions {
    pub tool_call_content: Option<ToolCallContentMode>,
    pub require_provider_parameters: Option<bool>,
    pub do_sample: Option<bool>,
    pub include_stream_usage: Option<bool>,
    pub prompt_cache_key: Option<String>,
    pub user: Option<String>,
}

/// Request fields that are applied before per-call `extra_*` values.
#[derive(Debug, Clone, Default)]
pub struct RequestProfile {
    pub extra_body: Map<String, Value>,
    pub extra_headers: HeaderMap,
    pub extra_query: Vec<(String, String)>,
}

impl RequestProfile {
    pub fn body(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.extra_body.insert(key.into(), value.into());
        self
    }

    pub fn header(mut self, name: HeaderName, value: reqwest::header::HeaderValue) -> Self {
        self.extra_headers.insert(name, value);
        self
    }

    pub fn query(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_query.push((name.into(), value.into()));
        self
    }
}

/// Names observed in provider responses for each neutral reasoning category.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningAliases {
    pub response_text: Vec<String>,
    pub response_summary: Vec<String>,
    pub request_replay: Vec<String>,
    pub encrypted: Vec<String>,
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
    pub fn empty() -> Self {
        Self {
            response_text: Vec::new(),
            response_summary: Vec::new(),
            request_replay: Vec::new(),
            encrypted: Vec::new(),
            signature: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.response_text.is_empty()
            && self.response_summary.is_empty()
            && self.request_replay.is_empty()
            && self.encrypted.is_empty()
            && self.signature.is_empty()
    }

    pub fn text(mut self, field: impl Into<String>) -> Self {
        self.response_text.push(field.into());
        self
    }

    pub fn summary(mut self, field: impl Into<String>) -> Self {
        self.response_summary.push(field.into());
        self
    }

    pub fn replay(mut self, field: impl Into<String>) -> Self {
        self.request_replay.push(field.into());
        self
    }

    pub fn encrypted(mut self, field: impl Into<String>) -> Self {
        self.encrypted.push(field.into());
        self
    }

    pub fn signature(mut self, field: impl Into<String>) -> Self {
        self.signature.push(field.into());
        self
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ReasoningReplayPolicy {
    Never,
    #[default]
    SameProvider,
    PreserveInHistory,
    RequiredForToolCalls,
    EncryptedOnly,
    ModelDefined,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReasoningProfile {
    pub aliases: ReasoningAliases,
    pub replay: ReasoningReplayPolicy,
    pub thinking: ThinkingRequestProfile,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ToolCallIdPolicy {
    #[default]
    Preserve,
    Synthesize,
    NamePair,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolProfile {
    pub call_id_policy: ToolCallIdPolicy,
    pub parallel_tool_calls: bool,
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StreamTerminal {
    #[default]
    DataDone,
    ResponseCompleted,
    MessageStop,
    Eof,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum UsagePlacement {
    #[default]
    LastChunk,
    EveryChunk,
    UsageEvent,
    None,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StreamProfile {
    pub terminal: StreamTerminal,
    pub usage: UsagePlacement,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UsageProfile {
    pub placement: UsagePlacement,
}

/// Capability switches used to reject requests before they reach a provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capabilities {
    pub reasoning: ReasoningCapabilities,
    pub tools: ToolCapabilities,
    pub streaming: StreamCapabilities,
    pub tool_choice: bool,
    pub structured_output: bool,
    pub seed: bool,
    pub stop: bool,
    pub multimodal: bool,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReasoningCapabilities {
    pub supported: bool,
    pub text: bool,
    pub summary: bool,
    pub encrypted: bool,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolCapabilities {
    pub supported: bool,
    pub parallel: bool,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamCapabilities {
    pub supported: bool,
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
    pub matcher: ModelMatcher,
    pub thinking: ThinkingRequestProfile,
    pub reasoning_aliases: ReasoningAliases,
    pub replay: Option<ReasoningReplayPolicy>,
    pub max_tokens_semantics: MaxTokensSemantics,
    pub stream_function_arguments: bool,
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

    pub fn matches(&self, model: &str) -> bool {
        self.matcher.matches(model)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ModelMatcher {
    #[default]
    Any,
    Exact(String),
    Prefix(String),
}

impl ModelMatcher {
    pub fn matches(&self, model: &str) -> bool {
        match self {
            Self::Any => true,
            Self::Exact(expected) => model == expected,
            Self::Prefix(prefix) => model.starts_with(prefix),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ThinkingRequestProfile {
    #[default]
    None,
    EnabledFlag(String),
    ThinkingObject,
    Effort(String),
    MappedEffort {
        field: String,
        mapping: EffortMapping,
    },
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
    Composite(Vec<ThinkingRequestProfile>),
}

/// Provider-specific translation for neutral reasoning effort values.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EffortMapping {
    pub none: Option<String>,
    pub minimal: Option<String>,
    pub low: Option<String>,
    pub medium: Option<String>,
    pub high: Option<String>,
    pub xhigh: Option<String>,
    pub max: Option<String>,
}

impl EffortMapping {
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

    pub fn kimi_k3() -> Self {
        Self {
            low: Some("low".to_string()),
            high: Some("high".to_string()),
            max: Some("max".to_string()),
            ..Self::default()
        }
    }

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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MaxTokensSemantics {
    #[default]
    MaxOutputTokens,
    MaxTokens,
    MaxCompletionTokens,
}
