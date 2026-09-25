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
    System,
    Developer,
    User,
    Assistant,
    Tool,
    Custom(String),
}

/// A piece of a message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Part {
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_state: Option<ProviderState>,
    },
    Reasoning(Reasoning),
    ImageUrl {
        image_url: ImageUrl,
    },
    Image {
        image: ImagePart,
    },
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    ProviderItem(ServerToolItem),
}

impl Part {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text {
            text: text.into(),
            provider_state: None,
        }
    }

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

    pub fn reasoning(reasoning: Reasoning) -> Self {
        Self::Reasoning(reasoning)
    }

    pub fn image_url(url: impl Into<String>) -> Self {
        Self::ImageUrl {
            image_url: ImageUrl {
                url: url.into(),
                detail: None,
            },
        }
    }

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
    Summary,
    Text,
    Encrypted,
    Redacted,
}

/// The wire format that owns an opaque provider state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderStateFormat {
    OpenAiChat,
    OpenAiResponses,
    AnthropicMessages,
    GeminiGenerateContent,
    Custom(String),
}

/// Opaque state needed to replay provider-specific content.
///
/// The `data` value must be passed back to the same provider unchanged. It can
/// contain a signature, encrypted content, an item id, or other fields whose
/// meaning is deliberately unknown to the SDK.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderState {
    pub format: ProviderStateFormat,
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
    pub kind: ReasoningKind,
    pub summary: Option<String>,
    pub text: Option<String>,
    pub state: Option<ProviderState>,
}

impl Reasoning {
    pub fn summary(text: impl Into<String>) -> Self {
        Self {
            kind: ReasoningKind::Summary,
            summary: Some(text.into()),
            text: None,
            state: None,
        }
    }

    pub fn text(text: impl Into<String>) -> Self {
        Self {
            kind: ReasoningKind::Text,
            summary: None,
            text: Some(text.into()),
            state: None,
        }
    }

    pub fn encrypted() -> Self {
        Self {
            kind: ReasoningKind::Encrypted,
            summary: None,
            text: None,
            state: None,
        }
    }

    pub fn redacted() -> Self {
        Self {
            kind: ReasoningKind::Redacted,
            summary: None,
            text: None,
            state: None,
        }
    }

    pub fn with_state(mut self, format: ProviderStateFormat, data: serde_json::Value) -> Self {
        self.state = Some(ProviderState::new(format, data));
        self
    }

    pub fn display_text(&self) -> Option<&str> {
        self.text.as_deref().or(self.summary.as_deref())
    }
}

/// Image input used by multimodal messages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    pub detail: Option<ImageDetail>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageDetail {
    Auto,
    Low,
    High,
}

/// A provider-hosted tool declaration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerTool {
    WebSearch,
    UrlContext,
    FileSearch,
    Custom {
        name: String,
        config: serde_json::Value,
    },
}

/// Generic state of a provider-hosted tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServerToolState {
    InProgress,
    Completed,
    Failed,
    Unknown,
}

/// Opaque provider-hosted tool item returned by a provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServerToolItem {
    pub tool: Option<String>,
    pub call_id: Option<String>,
    pub state: Option<ServerToolState>,
    /// Provider-owned item. It must be replayed unchanged to the same provider.
    pub provider_state: ProviderState,
}

impl ServerToolItem {
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
    Url {
        url: String,
    },
    Base64 {
        media_type: String,
        data: String,
    },
    FileRef {
        uri: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        media_type: Option<String>,
    },
}

/// A multimodal image input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImagePart {
    pub source: ImageSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageDetail>,
}

impl ImagePart {
    pub fn url(url: impl Into<String>) -> Self {
        Self {
            source: ImageSource::Url { url: url.into() },
            detail: None,
        }
    }

    pub fn base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self {
            source: ImageSource::Base64 {
                media_type: media_type.into(),
                data: data.into(),
            },
            detail: None,
        }
    }

    pub fn file_ref(uri: impl Into<String>, media_type: Option<String>) -> Self {
        Self {
            source: ImageSource::FileRef {
                uri: uri.into(),
                media_type,
            },
            detail: None,
        }
    }

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
    pub name: String,
    pub arguments: serde_json::Value,
    /// Provider-specific data attached to the tool-call block.
    ///
    /// Gemini currently uses this to carry `thoughtSignature`. Other adapters
    /// ignore state that belongs to a different provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_state: Option<ProviderState>,
}

impl ToolCall {
    pub fn new(name: impl Into<String>, arguments: serde_json::Value) -> Self {
        Self {
            id: None,
            name: name.into(),
            arguments,
            provider_state: None,
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

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
    Text { text: String },
    Json { value: serde_json::Value },
    Image { image: ImagePart },
}

/// A tool result that will be sent back to the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub call_id: Option<String>,
    pub name: String,
    /// Plain-text fallback retained for simple tools and provider downgrade.
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parts: Vec<ToolResultPart>,
    pub is_error: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

impl ToolResult {
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

    pub fn error(
        call_id: Option<String>,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        let mut result = Self::new(call_id, name, content);
        result.is_error = true;
        result
    }

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

    pub fn json(
        call_id: Option<String>,
        name: impl Into<String>,
        value: serde_json::Value,
    ) -> Self {
        Self::from_parts(call_id, name, vec![ToolResultPart::Json { value }])
    }

    pub fn image(call_id: Option<String>, name: impl Into<String>, image: ImagePart) -> Self {
        Self::from_parts(call_id, name, vec![ToolResultPart::Image { image }])
    }

    pub fn status(mut self, status: impl Into<String>) -> Self {
        self.status = Some(status.into());
        self
    }

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
    pub role: Role,
    pub parts: Vec<Part>,
}

impl Message {
    pub fn new(role: Role, parts: Vec<Part>) -> Self {
        Self { role, parts }
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self::text(Role::System, text)
    }

    pub fn developer(text: impl Into<String>) -> Self {
        Self::text(Role::Developer, text)
    }

    pub fn user(text: impl Into<String>) -> Self {
        Self::text(Role::User, text)
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self::text(Role::Assistant, text)
    }

    pub fn custom(role: impl Into<String>, text: impl Into<String>) -> Self {
        Self::text(Role::Custom(role.into()), text)
    }

    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Self {
            role,
            parts: vec![Part::text(text)],
        }
    }

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

    pub fn assistant_reasoning(reasoning: Reasoning) -> Self {
        Self {
            role: Role::Assistant,
            parts: vec![Part::Reasoning(reasoning)],
        }
    }

    pub fn tool_result(result: ToolResult) -> Self {
        Self {
            role: Role::Tool,
            parts: vec![Part::ToolResult(result)],
        }
    }

    pub fn tool_results(results: impl IntoIterator<Item = ToolResult>) -> Self {
        Self {
            role: Role::Tool,
            parts: results.into_iter().map(Part::ToolResult).collect(),
        }
    }

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

    pub fn tool_calls(&self) -> impl Iterator<Item = &ToolCall> {
        self.parts.iter().filter_map(|part| match part {
            Part::ToolCall(call) => Some(call),
            _ => None,
        })
    }

    pub fn tool_result_parts(&self) -> impl Iterator<Item = &ToolResult> {
        self.parts.iter().filter_map(|part| match part {
            Part::ToolResult(result) => Some(result),
            _ => None,
        })
    }

    pub fn reasoning(&self) -> impl Iterator<Item = &Reasoning> {
        self.parts.iter().filter_map(|part| match part {
            Part::Reasoning(reasoning) => Some(reasoning),
            _ => None,
        })
    }

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
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

impl ToolSpec {
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
    None,
    Minimal,
    Low,
    Medium,
    High,
    #[serde(rename = "xhigh")]
    XHigh,
    Max,
}

/// How much reasoning detail the provider should expose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningSummary {
    Auto,
    Concise,
    Detailed,
}

/// High-level mode for providers that expose enable/disable controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningMode {
    Enabled,
    Adaptive,
    Disabled,
}

/// Provider-neutral reasoning request configuration.
///
/// Adapters translate only the fields that their protocol supports. Fields
/// that have no equivalent are ignored rather than guessed.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ReasoningConfig {
    pub mode: Option<ReasoningMode>,
    pub effort: Option<ReasoningEffort>,
    pub budget_tokens: Option<u32>,
    pub summary: Option<ReasoningSummary>,
    /// Ask providers such as Gemini to include readable thought parts.
    pub include_text: Option<bool>,
    /// Ask OpenAI Responses to include encrypted reasoning for stateless replay.
    pub include_encrypted: Option<bool>,
}

impl ReasoningConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn mode(mut self, mode: ReasoningMode) -> Self {
        self.mode = Some(mode);
        self
    }

    pub fn effort(mut self, effort: ReasoningEffort) -> Self {
        self.effort = Some(effort);
        self
    }

    pub fn budget_tokens(mut self, budget_tokens: u32) -> Self {
        self.budget_tokens = Some(budget_tokens);
        self
    }

    pub fn summary(mut self, summary: ReasoningSummary) -> Self {
        self.summary = Some(summary);
        self
    }

    pub fn include_text(mut self, include_text: bool) -> Self {
        self.include_text = Some(include_text);
        self
    }

    pub fn include_encrypted(mut self, include_encrypted: bool) -> Self {
        self.include_encrypted = Some(include_encrypted);
        self
    }
}

/// Provider-neutral tool selection for one request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Tool(String),
}

/// Provider-neutral output format for one request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        schema: serde_json::Value,
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
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSpec>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub server_tools: Vec<ServerTool>,
    pub tool_choice: Option<ToolChoice>,
    pub response_format: Option<ResponseFormat>,
    pub stop: Vec<String>,
    pub seed: Option<u64>,
    pub temperature: Option<f32>,
    pub max_output_tokens: Option<u32>,
    pub reasoning: Option<ReasoningConfig>,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra_body: serde_json::Map<String, serde_json::Value>,
}

impl ChatRequest {
    pub fn new(messages: impl IntoIterator<Item = Message>) -> Self {
        Self {
            messages: messages.into_iter().collect(),
            ..Self::default()
        }
    }

    pub fn user(text: impl Into<String>) -> Self {
        Self::new([Message::user(text)])
    }

    pub fn push(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    pub fn tools(mut self, tools: impl IntoIterator<Item = ToolSpec>) -> Self {
        self.tools = tools.into_iter().collect();
        self
    }

    pub fn server_tools(mut self, tools: impl IntoIterator<Item = ServerTool>) -> Self {
        self.server_tools = tools.into_iter().collect();
        self
    }

    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    pub fn response_format(mut self, response_format: ResponseFormat) -> Self {
        self.response_format = Some(response_format);
        self
    }

    pub fn stop<I, S>(mut self, stop: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.stop = stop.into_iter().map(Into::into).collect();
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn max_output_tokens(mut self, max_output_tokens: u32) -> Self {
        self.max_output_tokens = Some(max_output_tokens);
        self
    }

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
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub cache_read_tokens: Option<u64>,
    pub cache_write_tokens: Option<u64>,
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
    pub status: u16,
    pub request_id: Option<String>,
    pub headers: HeaderMap,
}

impl ResponseMetadata {
    pub fn header(&self, name: impl AsRef<str>) -> Option<&reqwest::header::HeaderValue> {
        reqwest::header::HeaderName::from_bytes(name.as_ref().as_bytes())
            .ok()
            .and_then(|name| self.headers.get(name))
    }

    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

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
    pub message: Message,
    pub usage: Option<Usage>,
    /// Provider-native response or accumulated stream chunks.
    ///
    /// Shape is protocol-specific and can be large. Do not log it by default.
    pub raw: serde_json::Value,
    #[serde(skip)]
    pub metadata: Option<ResponseMetadata>,
    /// Transformations applied while preparing or decoding this response.
    #[serde(skip)]
    pub report: CompletionReport,
}

impl ChatResponse {
    pub fn new(message: Message) -> Self {
        Self {
            message,
            usage: None,
            raw: serde_json::Value::Null,
            metadata: None,
            report: CompletionReport::default(),
        }
    }

    pub fn text(&self) -> String {
        self.message.text_content()
    }

    pub fn tool_calls(&self) -> impl Iterator<Item = &ToolCall> {
        self.message.tool_calls()
    }

    pub fn reasoning(&self) -> impl Iterator<Item = &Reasoning> {
        self.message.reasoning()
    }

    pub fn reasoning_text(&self) -> String {
        self.message.reasoning_text()
    }
}
