use std::fmt;

use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};

use crate::report::CompletionReport;

/// The role used inside your application.
///
/// Provider APIs only understand a small set of roles. You may keep richer
/// roles such as [`Role::Developer`] or [`Role::Custom`] in your agent history;
/// the normalization layer decides how to downgrade them before sending.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// System-level instructions.
    System,
    /// Developer-level instructions. Downgraded by providers that lack this
    /// role (see [`crate::normalize`]).
    Developer,
    /// End-user input.
    User,
    /// Model output.
    Assistant,
    /// Tool output returned to the model.
    Tool,
    /// Application-defined role. Normalization decides how to downgrade it.
    Custom(String),
}

/// A piece of a message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Part {
    /// Plain text content.
    Text {
        /// The text content.
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_state: Option<ProviderState>,
    },
    /// Model reasoning content.
    Reasoning(Reasoning),
    /// Image referenced by URL.
    ImageUrl {
        image_url: ImageUrl,
    },
    /// Image passed inline as base64 data or a file reference.
    Image {
        image: ImagePart,
    },
    /// A tool call requested by the model.
    ToolCall(ToolCall),
    /// A tool result sent back to the model.
    ToolResult(ToolResult),
    /// A provider-hosted tool item such as web-search output.
    ProviderItem(ServerToolItem),
}

impl Part {
    /// Creates a plain [`Part::Text`] part.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text {
            text: text.into(),
            provider_state: None,
        }
    }

    /// Creates a [`Part::Text`] part carrying provider-specific replay state.
    pub fn text_with_provider_state(
        text: impl Into<String>,
        format: ProviderStateFormat,
        data: serde_json::Value,
    ) -> Self {
        Self::Text {
            text: text.into(),
            provider_state: Some(ProviderState::new(format, data)),
        }
    }

    /// Creates a [`Part::Reasoning`] part.
    pub fn reasoning(reasoning: Reasoning) -> Self {
        Self::Reasoning(reasoning)
    }

    /// Creates a [`Part::ImageUrl`] part from a URL.
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::ImageUrl {
            image_url: ImageUrl {
                url: url.into(),
                detail: None,
            },
        }
    }

    /// Creates a [`Part::Image`] part from an inline image source.
    pub fn image(image: ImagePart) -> Self {
        Self::Image { image }
    }
}

/// The main display category of a reasoning item.
///
/// Provider APIs disagree on names, but consumers usually only need to know
/// whether they received a summary, readable reasoning text, encrypted state,
/// or a safety-redacted block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningKind {
    /// A short summary of the reasoning, without full thought text.
    Summary,
    /// Readable reasoning text.
    Text,
    /// Encrypted reasoning kept for exact replay on the same provider.
    Encrypted,
    /// Reasoning redacted for safety; not displayable.
    Redacted,
}

/// The wire format that owns an opaque provider state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderStateFormat {
    /// OpenAI Chat Completions wire format.
    OpenAiChat,
    /// OpenAI Responses wire format.
    OpenAiResponses,
    /// Anthropic Messages wire format.
    AnthropicMessages,
    /// Gemini GenerateContent wire format.
    GeminiGenerateContent,
    /// A format not covered by the SDK.
    Custom(String),
}

/// Opaque state needed to replay provider-specific content.
///
/// The `data` value must be passed back to the same provider unchanged. It can
/// contain a signature, encrypted content, an item id, or other fields whose
/// meaning is deliberately unknown to the SDK.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderState {
    /// The wire format this state belongs to.
    pub format: ProviderStateFormat,
    /// The opaque payload. Its meaning is deliberately unknown to the SDK.
    pub data: serde_json::Value,
}

impl fmt::Debug for ProviderState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bytes = serde_json::to_vec(&self.data)
            .map(|data| data.len())
            .unwrap_or_default();
        formatter
            .debug_struct("ProviderState")
            .field("format", &self.format)
            .field("data_bytes", &bytes)
            .finish_non_exhaustive()
    }
}

impl ProviderState {
    /// Creates provider state from a format and an opaque payload.
    pub fn new(format: ProviderStateFormat, data: serde_json::Value) -> Self {
        Self { format, data }
    }
}

/// Provider-neutral reasoning content.
///
/// `summary` and `text` are display fields. `state` is for exact replay and is
/// not user-visible content. Some providers return both a summary and readable
/// reasoning text; in that case `kind` is [`ReasoningKind::Text`] and both
/// fields are populated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Reasoning {
    /// The display category of this reasoning item.
    pub kind: ReasoningKind,
    /// Short, user-visible summary of the reasoning.
    pub summary: Option<String>,
    /// Full readable reasoning text, when the provider returns it.
    pub text: Option<String>,
    /// Opaque state for exact replay on the originating provider.
    pub state: Option<ProviderState>,
}

impl Reasoning {
    /// Creates a [`ReasoningKind::Summary`] reasoning item.
    pub fn summary(text: impl Into<String>) -> Self {
        Self {
            kind: ReasoningKind::Summary,
            summary: Some(text.into()),
            text: None,
            state: None,
        }
    }

    /// Creates a [`ReasoningKind::Text`] reasoning item.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            kind: ReasoningKind::Text,
            summary: None,
            text: Some(text.into()),
            state: None,
        }
    }

    /// Creates a placeholder for encrypted reasoning with no display text.
    pub fn encrypted() -> Self {
        Self {
            kind: ReasoningKind::Encrypted,
            summary: None,
            text: None,
            state: None,
        }
    }

    /// Creates a placeholder for safety-redacted reasoning.
    pub fn redacted() -> Self {
        Self {
            kind: ReasoningKind::Redacted,
            summary: None,
            text: None,
            state: None,
        }
    }

    /// Attaches opaque provider state for exact replay.
    pub fn with_state(mut self, format: ProviderStateFormat, data: serde_json::Value) -> Self {
        self.state = Some(ProviderState::new(format, data));
        self
    }

    /// Returns the best display text: full reasoning text if present,
    /// otherwise the summary.
    pub fn display_text(&self) -> Option<&str> {
        self.text.as_deref().or(self.summary.as_deref())
    }
}

/// Image input used by multimodal messages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageUrl {
    /// The image URL.
    pub url: String,
    /// Optional processing-detail hint (OpenAI vision style).
    pub detail: Option<ImageDetail>,
}

/// Processing-detail hint for image inputs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageDetail {
    /// Let the provider decide.
    Auto,
    /// Prefer cheap, low-resolution processing.
    Low,
    /// Prefer high-resolution processing.
    High,
}

/// A provider-hosted tool declaration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerTool {
    /// Web search executed by the provider.
    WebSearch,
    /// Fetch and ground content from URLs supplied by the model.
    UrlContext,
    /// Search over a provider-hosted file store.
    FileSearch,
    /// A provider-specific hosted tool.
    Custom {
        /// Tool name understood by the provider.
        name: String,
        /// Provider-specific configuration for the tool.
        config: serde_json::Value,
    },
}

/// Generic state of a provider-hosted tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServerToolState {
    /// The tool is still running.
    InProgress,
    /// The tool finished successfully.
    Completed,
    /// The tool reported a failure.
    Failed,
    /// The provider did not report a state.
    Unknown,
}

/// Opaque provider-hosted tool item returned by a provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServerToolItem {
    /// Best-effort tool name, when the provider reports one.
    pub tool: Option<String>,
    /// Call id linking the item to a preceding tool call, when available.
    pub call_id: Option<String>,
    /// Best-effort execution state.
    pub state: Option<ServerToolState>,
    /// Provider-owned item. It must be replayed unchanged to the same provider.
    pub provider_state: ProviderState,
}

impl ServerToolItem {
    /// Creates an item wrapping opaque provider state.
    pub fn new(provider_state: ProviderState) -> Self {
        Self {
            tool: None,
            call_id: None,
            state: None,
            provider_state,
        }
    }
}

/// A provider-neutral image source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    /// Image fetched by URL.
    Url {
        /// The image URL.
        url: String,
    },
    /// Image embedded as base64 data.
    Base64 {
        /// MIME type such as `image/png`.
        media_type: String,
        /// Base64-encoded image bytes.
        data: String,
    },
    /// Image referenced by a provider- or user-held file URI.
    FileRef {
        /// The file URI.
        uri: String,
        /// Optional MIME type.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        media_type: Option<String>,
    },
}

/// A multimodal image input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImagePart {
    /// Where the image bytes come from.
    pub source: ImageSource,
    /// Optional processing-detail hint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageDetail>,
}

impl ImagePart {
    /// Creates an image part referencing a URL.
    pub fn url(url: impl Into<String>) -> Self {
        Self {
            source: ImageSource::Url { url: url.into() },
            detail: None,
        }
    }

    /// Creates an image part from base64 data.
    pub fn base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self {
            source: ImageSource::Base64 {
                media_type: media_type.into(),
                data: data.into(),
            },
            detail: None,
        }
    }

    /// Creates an image part referencing a file URI.
    pub fn file_ref(uri: impl Into<String>, media_type: Option<String>) -> Self {
        Self {
            source: ImageSource::FileRef {
                uri: uri.into(),
                media_type,
            },
            detail: None,
        }
    }

    /// Sets the processing-detail hint.
    pub fn detail(mut self, detail: ImageDetail) -> Self {
        self.detail = Some(detail);
        self
    }
}

/// A tool call requested by the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Providers such as Gemini do not always return an id. The normalizer
    /// fills a stable synthetic id when one is needed for pairing.
    pub id: Option<String>,
    /// The name of the tool to call.
    pub name: String,
    /// Tool arguments as a JSON object (usually parsed by the application).
    pub arguments: serde_json::Value,
    /// Provider-specific data attached to the tool-call block.
    ///
    /// Gemini currently uses this to carry `thoughtSignature`. Other adapters
    /// ignore state that belongs to a different provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_state: Option<ProviderState>,
}

impl ToolCall {
    /// Creates a tool call with no id and no provider state.
    pub fn new(name: impl Into<String>, arguments: serde_json::Value) -> Self {
        Self {
            id: None,
            name: name.into(),
            arguments,
            provider_state: None,
        }
    }

    /// Sets the tool-call id.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Attaches provider-specific state such as a `thoughtSignature`.
    pub fn with_provider_state(
        mut self,
        format: ProviderStateFormat,
        data: serde_json::Value,
    ) -> Self {
        self.provider_state = Some(ProviderState::new(format, data));
        self
    }
}

/// A structured tool-result payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultPart {
    /// Plain text output.
    Text { text: String },
    /// Structured JSON output.
    Json { value: serde_json::Value },
    /// Image output produced by the tool.
    Image { image: ImagePart },
}

/// A tool result that will be sent back to the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    /// Id of the originating tool call, when the provider issued one.
    pub call_id: Option<String>,
    /// Name of the tool that was executed.
    pub name: String,
    /// Plain-text fallback retained for simple tools and provider downgrade.
    pub content: String,
    /// Structured output parts; the richer representation of `content`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parts: Vec<ToolResultPart>,
    /// Whether the tool reported a failure.
    pub is_error: bool,
    /// Optional provider-style status label (e.g. `"completed"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    /// Optional provider-specific metadata attached to the result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

impl ToolResult {
    /// Creates a successful text result. A non-empty `content` is mirrored
    /// into [`ToolResult::parts`] as a single [`ToolResultPart::Text`].
    pub fn new(
        call_id: Option<String>,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        let content = content.into();
        Self {
            call_id,
            name: name.into(),
            parts: if content.is_empty() {
                Vec::new()
            } else {
                vec![ToolResultPart::Text {
                    text: content.clone(),
                }]
            },
            content,
            is_error: false,
            status: None,
            metadata: None,
        }
    }

    /// Creates an error result with `is_error` set.
    pub fn error(
        call_id: Option<String>,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        let mut result = Self::new(call_id, name, content);
        result.is_error = true;
        result
    }

    /// Creates a result from structured parts. `content` is derived from the
    /// parts as a plain-text fallback.
    pub fn from_parts(
        call_id: Option<String>,
        name: impl Into<String>,
        parts: Vec<ToolResultPart>,
    ) -> Self {
        Self {
            call_id,
            name: name.into(),
            content: tool_result_parts_text(&parts),
            parts,
            is_error: false,
            status: None,
            metadata: None,
        }
    }

    /// Creates a result carrying a single JSON part.
    pub fn json(
        call_id: Option<String>,
        name: impl Into<String>,
        value: serde_json::Value,
    ) -> Self {
        Self::from_parts(call_id, name, vec![ToolResultPart::Json { value }])
    }

    /// Creates a result carrying a single image part.
    pub fn image(call_id: Option<String>, name: impl Into<String>, image: ImagePart) -> Self {
        Self::from_parts(call_id, name, vec![ToolResultPart::Image { image }])
    }

    /// Sets a provider-style status label.
    pub fn status(mut self, status: impl Into<String>) -> Self {
        self.status = Some(status.into());
        self
    }

    /// Attaches provider-specific metadata.
    pub fn metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

fn tool_result_parts_text(parts: &[ToolResultPart]) -> String {
    let mut output = String::new();
    for part in parts {
        if !output.is_empty() {
            output.push('\n');
        }
        match part {
            ToolResultPart::Text { text } => output.push_str(text),
            ToolResultPart::Json { value } => {
                output
                    .push_str(&serde_json::to_string(value).unwrap_or_else(|_| "null".to_string()));
            }
            ToolResultPart::Image { image } => {
                output.push_str(&format!("[image: {}]", image_source_label(&image.source)));
            }
        }
    }
    output
}

fn image_source_label(source: &ImageSource) -> String {
    match source {
        ImageSource::Url { url } => url.clone(),
        ImageSource::Base64 { media_type, .. } => format!("base64:{media_type}"),
        ImageSource::FileRef { uri, .. } => uri.clone(),
    }
}

/// A message in the internal conversation model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Who produced this message.
    pub role: Role,
    /// The message content as an ordered list of parts.
    pub parts: Vec<Part>,
}

impl Message {
    /// Creates a message from a role and an ordered list of parts.
    pub fn new(role: Role, parts: Vec<Part>) -> Self {
        Self { role, parts }
    }

    /// Creates a single-text [`Role::System`] message.
    pub fn system(text: impl Into<String>) -> Self {
        Self::text(Role::System, text)
    }

    /// Creates a single-text [`Role::Developer`] message.
    pub fn developer(text: impl Into<String>) -> Self {
        Self::text(Role::Developer, text)
    }

    /// Creates a single-text [`Role::User`] message.
    pub fn user(text: impl Into<String>) -> Self {
        Self::text(Role::User, text)
    }

    /// Creates a single-text [`Role::Assistant`] message.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self::text(Role::Assistant, text)
    }

    /// Creates a single-text [`Role::Custom`] message.
    pub fn custom(role: impl Into<String>, text: impl Into<String>) -> Self {
        Self::text(Role::Custom(role.into()), text)
    }

    /// Creates a message with a single text part of the given role.
    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Self {
            role,
            parts: vec![Part::text(text)],
        }
    }

    /// Creates an assistant message with optional text followed by tool-call
    /// parts. Empty text is skipped.
    pub fn assistant_with_tools(
        text: impl Into<String>,
        tool_calls: impl IntoIterator<Item = ToolCall>,
    ) -> Self {
        let text = text.into();
        let mut parts = Vec::new();
        if !text.is_empty() {
            parts.push(Part::text(text));
        }
        parts.extend(tool_calls.into_iter().map(Part::ToolCall));
        Self {
            role: Role::Assistant,
            parts,
        }
    }

    /// Creates an assistant message holding a single reasoning part.
    pub fn assistant_reasoning(reasoning: Reasoning) -> Self {
        Self {
            role: Role::Assistant,
            parts: vec![Part::Reasoning(reasoning)],
        }
    }

    /// Creates a [`Role::Tool`] message holding one tool result.
    pub fn tool_result(result: ToolResult) -> Self {
        Self {
            role: Role::Tool,
            parts: vec![Part::ToolResult(result)],
        }
    }

    /// Creates a [`Role::Tool`] message holding multiple tool results.
    pub fn tool_results(results: impl IntoIterator<Item = ToolResult>) -> Self {
        Self {
            role: Role::Tool,
            parts: results.into_iter().map(Part::ToolResult).collect(),
        }
    }

    /// Returns the text parts joined by newlines. Tool-result text content is
    /// included; other part kinds are skipped.
    pub fn text_content(&self) -> String {
        let mut output = String::new();
        for part in &self.parts {
            match part {
                Part::Text { text, .. } => {
                    if !output.is_empty() {
                        output.push('\n');
                    }
                    output.push_str(text);
                }
                Part::ToolResult(result) => {
                    if !output.is_empty() {
                        output.push('\n');
                    }
                    output.push_str(&result.content);
                }
                _ => {}
            }
        }
        output
    }

    /// Iterates over the tool calls requested in this message.
    pub fn tool_calls(&self) -> impl Iterator<Item = &ToolCall> {
        self.parts.iter().filter_map(|part| match part {
            Part::ToolCall(call) => Some(call),
            _ => None,
        })
    }

    /// Iterates over the tool results carried by this message.
    /// Iterates over the tool results carried by this message.
    pub fn tool_result_parts(&self) -> impl Iterator<Item = &ToolResult> {
        self.parts.iter().filter_map(|part| match part {
            Part::ToolResult(result) => Some(result),
            _ => None,
        })
    }

    /// Iterates over the reasoning parts of this message.
    pub fn reasoning(&self) -> impl Iterator<Item = &Reasoning> {
        self.parts.iter().filter_map(|part| match part {
            Part::Reasoning(reasoning) => Some(reasoning),
            _ => None,
        })
    }

    /// Returns all reasoning display text joined by newlines.
    pub fn reasoning_text(&self) -> String {
        self.reasoning()
            .filter_map(Reasoning::display_text)
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// A provider-neutral tool definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Function-style tool name the model may call.
    pub name: String,
    /// Human-readable description shown to the model.
    pub description: String,
    /// JSON Schema describing the tool's arguments.
    pub parameters: serde_json::Value,
}

impl ToolSpec {
    /// Creates a tool definition.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

/// A provider-neutral reasoning-effort level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    /// Disable reasoning entirely where the provider supports it.
    None,
    /// The cheapest reasoning tier.
    Minimal,
    Low,
    Medium,
    High,
    /// Extra-high reasoning (OpenAI `xhigh`).
    #[serde(rename = "xhigh")]
    XHigh,
    /// The provider's maximum reasoning tier.
    Max,
}

/// How much reasoning detail the provider should expose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningSummary {
    /// Let the provider decide whether to include a summary.
    Auto,
    /// A short summary.
    Concise,
    /// A detailed summary.
    Detailed,
}

/// High-level mode for providers that expose enable/disable controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningMode {
    /// Always reason before answering.
    Enabled,
    /// Let the model decide when to reason.
    Adaptive,
    /// Do not reason.
    Disabled,
}

/// Provider-neutral reasoning request configuration.
///
/// Adapters translate only the fields that their protocol supports. Fields
/// that have no equivalent are ignored rather than guessed.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ReasoningConfig {
    /// Enable/disable mode for providers with a simple toggle.
    pub mode: Option<ReasoningMode>,
    /// Coarse effort level.
    pub effort: Option<ReasoningEffort>,
    /// Token budget for reasoning.
    pub budget_tokens: Option<u32>,
    /// How much reasoning summary to expose.
    pub summary: Option<ReasoningSummary>,
    /// Ask providers such as Gemini to include readable thought parts.
    pub include_text: Option<bool>,
    /// Ask OpenAI Responses to include encrypted reasoning for stateless replay.
    pub include_encrypted: Option<bool>,
}

impl ReasoningConfig {
    /// Creates an empty configuration with all options unset.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the enable/disable mode.
    pub fn mode(mut self, mode: ReasoningMode) -> Self {
        self.mode = Some(mode);
        self
    }

    /// Sets the effort level.
    pub fn effort(mut self, effort: ReasoningEffort) -> Self {
        self.effort = Some(effort);
        self
    }

    /// Sets the reasoning token budget.
    pub fn budget_tokens(mut self, budget_tokens: u32) -> Self {
        self.budget_tokens = Some(budget_tokens);
        self
    }

    /// Sets the reasoning summary verbosity.
    pub fn summary(mut self, summary: ReasoningSummary) -> Self {
        self.summary = Some(summary);
        self
    }

    /// Requests readable thought parts (e.g. Gemini).
    pub fn include_text(mut self, include_text: bool) -> Self {
        self.include_text = Some(include_text);
        self
    }

    /// Requests encrypted reasoning for stateless replay (e.g. OpenAI
    /// Responses).
    pub fn include_encrypted(mut self, include_encrypted: bool) -> Self {
        self.include_encrypted = Some(include_encrypted);
        self
    }
}

/// Provider-neutral tool selection for one request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    /// The model decides whether to call tools.
    Auto,
    /// The model must not call tools.
    None,
    /// The model must call at least one tool.
    Required,
    /// The model must call the named tool.
    Tool(String),
}

/// Provider-neutral output format for one request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Plain text output.
    Text,
    /// Output must be a valid JSON object.
    JsonObject,
    /// Output must conform to a JSON Schema.
    JsonSchema {
        /// Schema name used by providers that require one.
        name: String,
        /// The JSON Schema itself.
        schema: serde_json::Value,
        /// Whether the provider should strictly enforce the schema.
        strict: Option<bool>,
    },
}

/// A request sent to a model adapter.
///
/// `extra_body` is an escape hatch for provider-specific wire fields. The SDK
/// merges it into the final JSON object but refuses to let it overwrite
/// canonical fields such as `model`, `messages`, `input`, or `stream`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    /// The conversation so far.
    pub messages: Vec<Message>,
    /// Client-side tools the model may call.
    pub tools: Vec<ToolSpec>,
    /// Provider-hosted tools such as web search.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub server_tools: Vec<ServerTool>,
    /// How the model should choose among tools.
    pub tool_choice: Option<ToolChoice>,
    /// Requested output format.
    pub response_format: Option<ResponseFormat>,
    /// Sequences that stop generation.
    pub stop: Vec<String>,
    /// Sampling seed for reproducibility.
    pub seed: Option<u64>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Upper bound on generated tokens.
    pub max_output_tokens: Option<u32>,
    /// Reasoning behavior, for providers that support it.
    pub reasoning: Option<ReasoningConfig>,
    /// Escape hatch for provider-specific wire fields. Reserved canonical
    /// fields cannot be overwritten.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra_body: serde_json::Map<String, serde_json::Value>,
}

impl ChatRequest {
    /// Creates a request from the given messages; all other fields default.
    pub fn new(messages: impl IntoIterator<Item = Message>) -> Self {
        Self {
            messages: messages.into_iter().collect(),
            ..Self::default()
        }
    }

    /// Creates a request with a single user text message.
    pub fn user(text: impl Into<String>) -> Self {
        Self::new([Message::user(text)])
    }

    /// Appends a message.
    pub fn push(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    /// Replaces the client-side tool list.
    pub fn tools(mut self, tools: impl IntoIterator<Item = ToolSpec>) -> Self {
        self.tools = tools.into_iter().collect();
        self
    }

    /// Replaces the provider-hosted tool list.
    pub fn server_tools(mut self, tools: impl IntoIterator<Item = ServerTool>) -> Self {
        self.server_tools = tools.into_iter().collect();
        self
    }

    /// Sets how the model should choose among tools.
    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Sets the requested output format.
    pub fn response_format(mut self, response_format: ResponseFormat) -> Self {
        self.response_format = Some(response_format);
        self
    }

    /// Replaces the stop sequences.
    pub fn stop<I, S>(mut self, stop: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.stop = stop.into_iter().map(Into::into).collect();
        self
    }

    /// Sets the sampling seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the sampling temperature.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the output token limit.
    pub fn max_output_tokens(mut self, max_output_tokens: u32) -> Self {
        self.max_output_tokens = Some(max_output_tokens);
        self
    }

    /// Sets the reasoning configuration.
    pub fn reasoning(mut self, reasoning: ReasoningConfig) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    /// Add one provider-specific request body field.
    ///
    /// Reserved canonical fields are rejected by the adapter at request time,
    /// not silently overwritten.
    pub fn extra_body(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.extra_body.insert(key.into(), value.into());
        self
    }
}

/// Token usage reported by a provider.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Prompt tokens consumed.
    pub input_tokens: Option<u64>,
    /// Tokens generated.
    pub output_tokens: Option<u64>,
    /// Total tokens, when reported by the provider.
    pub total_tokens: Option<u64>,
    /// Tokens served from a prompt cache.
    pub cache_read_tokens: Option<u64>,
    /// Tokens written to a prompt cache.
    pub cache_write_tokens: Option<u64>,
    /// Tokens spent on reasoning.
    pub reasoning_tokens: Option<u64>,
    /// Provider-native usage object for fields not yet given a neutral name.
    pub raw: Option<serde_json::Value>,
}

impl Usage {
    /// Remove provider-native data before exposing usage to audit sinks.
    pub(crate) fn without_raw(mut self) -> Self {
        self.raw = None;
        self
    }
}

/// HTTP metadata associated with a model response.
#[derive(Debug, Clone, PartialEq)]
pub struct ResponseMetadata {
    /// HTTP status code of the response.
    pub status: u16,
    /// Request id taken from common provider headers, when present.
    pub request_id: Option<String>,
    /// All response headers.
    pub headers: HeaderMap,
}

impl ResponseMetadata {
    /// Looks up a header by name. Header names are compared
    /// case-insensitively.
    pub fn header(&self, name: impl AsRef<str>) -> Option<&reqwest::header::HeaderValue> {
        reqwest::header::HeaderName::from_bytes(name.as_ref().as_bytes())
            .ok()
            .and_then(|name| self.headers.get(name))
    }

    /// Returns the request id, when one was found.
    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    /// Captures metadata from a completed response. The request id is taken
    /// from the first of `x-request-id`, `request-id`, `x-ms-request-id`, or
    /// `cf-ray` present in the headers.
    pub fn from_response(response: &reqwest::Response) -> Self {
        let request_id = ["x-request-id", "request-id", "x-ms-request-id", "cf-ray"]
            .into_iter()
            .find_map(|name| {
                response
                    .headers()
                    .get(name)
                    .and_then(|value| value.to_str().ok())
                    .map(str::to_string)
            });

        Self {
            status: response.status().as_u16(),
            request_id,
            headers: response.headers().clone(),
        }
    }
}

/// A normalized model response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatResponse {
    /// The assistant message produced by the model.
    pub message: Message,
    /// Token usage, when reported by the provider.
    pub usage: Option<Usage>,
    /// Provider-native response or accumulated stream chunks.
    ///
    /// Shape is protocol-specific and can be large. Do not log it by default.
    pub raw: serde_json::Value,
    /// HTTP metadata, set by transports that keep it; not serialized.
    #[serde(skip)]
    pub metadata: Option<ResponseMetadata>,
    /// Transformations applied while preparing or decoding this response.
    #[serde(skip)]
    pub report: CompletionReport,
}

impl ChatResponse {
    /// Creates a response with no usage, raw payload, metadata, or report.
    pub fn new(message: Message) -> Self {
        Self {
            message,
            usage: None,
            raw: serde_json::Value::Null,
            metadata: None,
            report: CompletionReport::default(),
        }
    }

    /// Returns the text content of the response message.
    pub fn text(&self) -> String {
        self.message.text_content()
    }

    /// Iterates over the tool calls requested in the response.
    pub fn tool_calls(&self) -> impl Iterator<Item = &ToolCall> {
        self.message.tool_calls()
    }

    /// Iterates over the reasoning parts of the response.
    pub fn reasoning(&self) -> impl Iterator<Item = &Reasoning> {
        self.message.reasoning()
    }

    /// Returns all reasoning display text joined by newlines.
    pub fn reasoning_text(&self) -> String {
        self.message.reasoning_text()
    }
}
