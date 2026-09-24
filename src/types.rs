use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};

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
    ToolCall(ToolCall),
    ToolResult(ToolResult),
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderState {
    pub format: ProviderStateFormat,
    pub data: serde_json::Value,
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

/// A tool result that will be sent back to the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub call_id: Option<String>,
    pub name: String,
    pub content: String,
    pub is_error: bool,
}

impl ToolResult {
    pub fn new(
        call_id: Option<String>,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            call_id,
            name: name.into(),
            content: content.into(),
            is_error: false,
        }
    }

    pub fn error(
        call_id: Option<String>,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            call_id,
            name: name.into(),
            content: content.into(),
            is_error: true,
        }
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
}

/// HTTP metadata associated with a model response.
#[derive(Debug, Clone, PartialEq)]
pub struct ResponseMetadata {
    pub status: u16,
    pub request_id: Option<String>,
    pub headers: HeaderMap,
}

impl ResponseMetadata {
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
    pub raw: serde_json::Value,
    #[serde(skip)]
    pub metadata: Option<ResponseMetadata>,
}

impl ChatResponse {
    pub fn new(message: Message) -> Self {
        Self {
            message,
            usage: None,
            raw: serde_json::Value::Null,
            metadata: None,
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
