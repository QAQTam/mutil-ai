use std::collections::BTreeMap;

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};

use super::{
    ModelAdapter, ReconnectRequest, ReconnectRequestParts, apply_provider_request_options,
    apply_request_transform, audit_outcome_for_error, completion_report, effective_retry_policy,
    finish_sse_stream, json_body_bytes, merge_extra_body, prepare_headers, protocol_name,
    resolve_api_key, send_json_retry, send_stream_retry, validate_server_tools,
    validate_tool_choice,
};
use crate::error::{Error, Result};
use crate::headers::{RequestOptions, TransportConfig};
use crate::normalize::{ExternalRole, NormalizedChat, Protocol, normalize};
use crate::report::CompletionReport;
use crate::retry::{RetryPolicy, RetryProvider};
use crate::sse::SseMessage;
use crate::stream::{ModelStream, SseMapper, StreamEvent};
use crate::types::{
    ChatRequest, ChatResponse, Message, Part, ProviderState, ProviderStateFormat, Reasoning,
    ReasoningConfig, ReasoningKind, ReasoningMode, ResponseMetadata, Role, ToolCall, ToolChoice,
    Usage,
};

const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Entry point for Anthropic adapters.
///
/// See [`Anthropic::messages`] to construct an [`AnthropicMessages`] adapter.
pub struct Anthropic;

impl Anthropic {
    /// Use the Anthropic Messages API.
    pub fn messages(model: impl Into<String>) -> AnthropicMessages {
        AnthropicMessages::new(model)
    }
}

/// Adapter for `POST /v1/messages`.
pub struct AnthropicMessages {
    model: String,
    api_key: Option<String>,
    base_url: String,
    retry_policy: RetryPolicy,
    transport: TransportConfig,
    client: reqwest::Client,
}

impl AnthropicMessages {
    /// Create an adapter for the given Anthropic model.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            api_key: None,
            base_url: "https://api.anthropic.com/v1".to_string(),
            retry_policy: RetryPolicy::default(),
            transport: TransportConfig::default(),
            client: reqwest::Client::new(),
        }
    }

    /// Set an explicit API key, taking precedence over `ANTHROPIC_API_KEY`.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Override the default base URL (`https://api.anthropic.com/v1`),
    /// e.g. for gateways or self-hosted proxies speaking the same protocol.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Override the retry policy used for failed requests.
    pub fn retry_policy(mut self, retry_policy: RetryPolicy) -> Self {
        self.retry_policy = retry_policy;
        self
    }

    /// Configure transport-level settings (timeouts, proxies, audit hooks).
    pub fn transport(mut self, transport: TransportConfig) -> Self {
        self.transport = transport;
        self
    }
}

#[async_trait]
impl ModelAdapter for AnthropicMessages {
    fn provider_name(&self) -> &'static str {
        "anthropic-messages"
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    async fn complete_with(
        &self,
        request: &ChatRequest,
        options: &RequestOptions,
    ) -> Result<ChatResponse> {
        let transformed = apply_request_transform(request, options)?;
        let transform_applied = matches!(transformed, std::borrow::Cow::Owned(_));
        let request = transformed.as_ref();
        let (normalized, report) = normalize(request, Protocol::AnthropicMessages)?;
        validate_anthropic_request(request)?;
        let mut body = to_anthropic_body(&self.model, &normalized, request);
        merge_extra_body(&mut body, request, Protocol::AnthropicMessages, None)?;
        apply_provider_request_options(
            &mut body,
            Protocol::AnthropicMessages,
            &options.provider_request,
            false,
        )?;
        let request_body_bytes = json_body_bytes(&body)?;
        let key = resolve_api_key(
            self.provider_name(),
            self.api_key.as_deref(),
            "ANTHROPIC_API_KEY",
        )?;
        options.validate_query(protocol_name(Protocol::AnthropicMessages), &[])?;
        let retry_policy = effective_retry_policy(&self.retry_policy, options);
        let audit = self.transport.audit_context(
            self.provider_name(),
            protocol_name(Protocol::AnthropicMessages),
            &self.model,
            options,
            None,
            request_body_bytes,
        );
        if let Some(audit) = &audit {
            audit.request_started();
            if transform_applied {
                audit.transform_applied();
            }
            audit.normalization(report.stats());
        }

        let mut base_headers = HeaderMap::new();
        base_headers.insert(
            HeaderName::from_static("x-api-key"),
            HeaderValue::from_str(&key)?,
        );
        base_headers.insert(
            HeaderName::from_static("anthropic-version"),
            HeaderValue::from_static(ANTHROPIC_VERSION),
        );

        let result = send_json_retry(
            self.provider_name(),
            RetryProvider::Anthropic,
            &retry_policy,
            audit.as_ref(),
            options.cancellation.as_ref(),
            || async {
                let headers = prepare_headers(&self.transport, &base_headers, options).await?;
                let mut request_builder = self
                    .client
                    .post(format!("{}/messages", self.base_url.trim_end_matches('/')))
                    .headers(headers)
                    .query(&options.extra_query)
                    .json(&body);
                if let Some(timeout) = options.timeout {
                    request_builder = request_builder.timeout(timeout);
                }
                Ok(request_builder)
            },
        )
        .await;

        let (value, metadata) = match result {
            Ok(response) => response,
            Err(error) => {
                if let Some(audit) = &audit {
                    audit.request_finished_with_request_id(
                        audit_outcome_for_error(&error),
                        error.status(),
                        error.request_id().map(str::to_string),
                        Some(&error),
                        None,
                    );
                }
                return Err(error);
            }
        };

        let mut response = match from_anthropic_response(value) {
            Ok(response) => response,
            Err(error) => {
                if let Some(audit) = &audit {
                    audit.request_finished_with_request_id(
                        crate::audit::AuditOutcome::Failure,
                        Some(metadata.status),
                        metadata.request_id.clone(),
                        Some(&error),
                        None,
                    );
                }
                return Err(error);
            }
        };
        response.report = completion_report(
            Protocol::AnthropicMessages,
            request,
            report.stats(),
            transform_applied,
        );
        if let Some(audit) = &audit {
            audit.request_finished_with_request_id(
                crate::audit::AuditOutcome::Success,
                Some(metadata.status),
                metadata.request_id.clone(),
                None,
                response.usage.clone(),
            );
        }
        response.metadata = Some(metadata);
        Ok(response)
    }

    async fn stream_with(
        &self,
        request: &ChatRequest,
        options: &RequestOptions,
    ) -> Result<ModelStream> {
        let transformed = apply_request_transform(request, options)?;
        let transform_applied = matches!(transformed, std::borrow::Cow::Owned(_));
        let request = transformed.as_ref();
        let (normalized, report) = normalize(request, Protocol::AnthropicMessages)?;
        validate_anthropic_request(request)?;
        let mut body = to_anthropic_body(&self.model, &normalized, request);
        merge_extra_body(&mut body, request, Protocol::AnthropicMessages, None)?;
        body["stream"] = json!(true);
        apply_provider_request_options(
            &mut body,
            Protocol::AnthropicMessages,
            &options.provider_request,
            true,
        )?;
        let request_body_bytes = json_body_bytes(&body)?;

        let key = resolve_api_key(
            self.provider_name(),
            self.api_key.as_deref(),
            "ANTHROPIC_API_KEY",
        )?;
        options.validate_query(protocol_name(Protocol::AnthropicMessages), &[])?;
        let retry_policy = effective_retry_policy(&self.retry_policy, options);
        let audit = self.transport.audit_context(
            self.provider_name(),
            protocol_name(Protocol::AnthropicMessages),
            &self.model,
            options,
            None,
            request_body_bytes,
        );
        if let Some(audit) = &audit {
            audit.request_started();
            if transform_applied {
                audit.transform_applied();
            }
            audit.normalization(report.stats());
        }

        let mut base_headers = HeaderMap::new();
        base_headers.insert(
            HeaderName::from_static("x-api-key"),
            HeaderValue::from_str(&key)?,
        );
        base_headers.insert(
            HeaderName::from_static("anthropic-version"),
            HeaderValue::from_static(ANTHROPIC_VERSION),
        );
        let url = format!("{}/messages", self.base_url.trim_end_matches('/'));

        let response = match send_stream_retry(
            self.provider_name(),
            RetryProvider::Anthropic,
            &retry_policy,
            audit.as_ref(),
            options.cancellation.as_ref(),
            || async {
                let headers = prepare_headers(&self.transport, &base_headers, options).await?;
                let mut request_builder = self
                    .client
                    .post(&url)
                    .headers(headers)
                    .query(&options.extra_query)
                    .json(&body);
                if let Some(timeout) = options.timeout {
                    request_builder = request_builder.timeout(timeout);
                }
                Ok(request_builder)
            },
        )
        .await
        {
            Ok(response) => response,
            Err(error) => {
                if let Some(audit) = &audit {
                    audit.request_finished_with_request_id(
                        audit_outcome_for_error(&error),
                        error.status(),
                        error.request_id().map(str::to_string),
                        Some(&error),
                        None,
                    );
                }
                return Err(error);
            }
        };

        let metadata = ResponseMetadata::from_response(&response);
        let mapper = AnthropicStreamMapper::new(self.model.clone(), metadata).with_report(
            completion_report(
                Protocol::AnthropicMessages,
                request,
                report.stats(),
                transform_applied,
            ),
        );
        let reconnect = options.stream_reconnect.map(|_| {
            ReconnectRequest::new(ReconnectRequestParts {
                provider: self.provider_name(),
                client: self.client.clone(),
                retry_policy: retry_policy.clone(),
                retry_provider: RetryProvider::Anthropic,
                transport: self.transport.clone(),
                options: options.clone(),
                audit: audit.clone(),
                url,
                base_headers,
                query: options.extra_query.clone(),
                body,
            })
        });
        Ok(finish_sse_stream(
            response,
            mapper,
            options.stream_reconnect,
            reconnect,
            options.cancellation.clone(),
            options.idle_timeout,
            audit,
        ))
    }
}

pub(crate) fn to_anthropic_body(
    model: &str,
    normalized: &NormalizedChat,
    request: &ChatRequest,
) -> Value {
    let mut messages: Vec<Value> = Vec::new();

    for message in &normalized.messages {
        match message.role {
            ExternalRole::System | ExternalRole::Developer => {
                // Anthropic always merges system text (see `SystemPlacement`);
                // an in-place developer message is demoted to a user turn.
                push_anthropic_message(
                    &mut messages,
                    "user",
                    vec![json!({"type": "text", "text": text_content(message)})],
                );
            }
            ExternalRole::User => {
                push_anthropic_message(&mut messages, "user", anthropic_content(&message.parts));
            }
            ExternalRole::Assistant => {
                let mut blocks = Vec::new();
                for part in &message.parts {
                    match part {
                        Part::Reasoning(reasoning) => {
                            if let Some(state) = &reasoning.state
                                && state.format == ProviderStateFormat::AnthropicMessages
                            {
                                blocks.push(state.data.clone());
                            }
                        }
                        Part::Text { text, .. } => {
                            blocks.push(json!({"type": "text", "text": text}))
                        }
                        Part::ToolCall(call) => blocks.push(json!({
                            "type": "tool_use",
                            "id": call.id.clone().unwrap_or_else(|| call.name.clone()),
                            "name": call.name,
                            "input": call.arguments,
                        })),
                        _ => {}
                    }
                }
                if !blocks.is_empty() {
                    push_anthropic_message(&mut messages, "assistant", blocks);
                }
            }
            ExternalRole::Tool => {
                let blocks = message
                    .parts
                    .iter()
                    .filter_map(|part| match part {
                        Part::ToolResult(result) => Some(json!({
                            "type": "tool_result",
                            "tool_use_id": result.call_id.clone().unwrap_or_else(|| result.name.clone()),
                            "content": anthropic_tool_result_content(result),
                            "is_error": result.is_error,
                        })),
                        _ => None,
                    })
                    .collect();
                push_anthropic_message(&mut messages, "user", blocks);
            }
        }
    }

    let mut body = json!({
        "model": model,
        "max_tokens": request.max_output_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        "messages": messages,
    });

    if let Some(system) = &normalized.system {
        body["system"] = json!(system);
    }
    if !normalized.tools.is_empty() {
        body["tools"] = Value::Array(
            normalized
                .tools
                .iter()
                .map(|tool| {
                    json!({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.parameters,
                    })
                })
                .collect(),
        );
    }
    if let Some(temperature) = request.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(thinking) = anthropic_thinking(request.reasoning.as_ref()) {
        body["thinking"] = thinking;
    }
    if let Some(tool_choice) = &request.tool_choice {
        body["tool_choice"] = anthropic_tool_choice(tool_choice);
    }
    if !request.stop.is_empty() {
        body["stop_sequences"] = json!(request.stop);
    }

    body
}

pub(crate) fn validate_anthropic_request(request: &ChatRequest) -> Result<()> {
    validate_server_tools(Protocol::AnthropicMessages, request)?;
    validate_tool_choice(&request.tools, request.tool_choice.as_ref())?;
    if request.response_format.is_some() {
        return Err(Error::Unsupported(
            "Anthropic Messages does not expose a neutral response_format mapping".to_string(),
        ));
    }
    if request.seed.is_some() {
        return Err(Error::Unsupported(
            "Anthropic Messages does not expose a seed mapping".to_string(),
        ));
    }
    Ok(())
}

fn anthropic_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!({"type": "auto"}),
        ToolChoice::None => json!({"type": "none"}),
        ToolChoice::Required => json!({"type": "any"}),
        ToolChoice::Tool(name) => json!({"type": "tool", "name": name}),
    }
}

fn push_anthropic_message(messages: &mut Vec<Value>, role: &str, blocks: Vec<Value>) {
    if blocks.is_empty() {
        return;
    }

    if let Some(last) = messages.last_mut()
        && last.get("role").and_then(Value::as_str) == Some(role)
    {
        if let Some(content) = last.get_mut("content").and_then(Value::as_array_mut) {
            content.extend(blocks);
        }
        return;
    }

    messages.push(json!({"role": role, "content": blocks}));
}

fn anthropic_content(parts: &[Part]) -> Vec<Value> {
    parts
        .iter()
        .filter_map(|part| match part {
            Part::Text { text, .. } => Some(json!({"type": "text", "text": text})),
            Part::ImageUrl { image_url } => Some(json!({
                "type": "image",
                "source": {
                    "type": "url",
                    "url": image_url.url,
                }
            })),
            Part::Image { image } => Some(anthropic_image_block(&image.source)),
            _ => None,
        })
        .collect()
}

fn anthropic_tool_result_content(result: &crate::types::ToolResult) -> Value {
    if result
        .parts
        .iter()
        .all(|part| matches!(part, crate::types::ToolResultPart::Text { .. }))
    {
        return Value::String(result.content.clone());
    }

    Value::Array(
        result
            .parts
            .iter()
            .map(|part| match part {
                crate::types::ToolResultPart::Text { text } => {
                    json!({"type": "text", "text": text})
                }
                crate::types::ToolResultPart::Json { value } => {
                    json!({"type": "text", "text": value.to_string()})
                }
                crate::types::ToolResultPart::Image { image } => {
                    anthropic_image_block(&image.source)
                }
            })
            .collect(),
    )
}

fn anthropic_image_block(source: &crate::types::ImageSource) -> Value {
    match source {
        crate::types::ImageSource::Url { url } => json!({
            "type": "image",
            "source": {"type": "url", "url": url},
        }),
        crate::types::ImageSource::Base64 { media_type, data } => json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            },
        }),
        crate::types::ImageSource::FileRef { uri, .. } => json!({
            "type": "image",
            "source": {"type": "url", "url": uri},
        }),
    }
}

fn anthropic_thinking(config: Option<&ReasoningConfig>) -> Option<Value> {
    let config = config?;
    let display = match config.include_text {
        Some(false) => Some("omitted"),
        Some(true) => Some("summarized"),
        None => config.summary.map(|_| "summarized"),
    };

    match config.mode {
        Some(ReasoningMode::Disabled) => Some(json!({"type": "disabled"})),
        Some(ReasoningMode::Adaptive) => {
            let mut value = json!({"type": "adaptive"});
            if let Some(display) = display {
                value["display"] = json!(display);
            }
            Some(value)
        }
        Some(ReasoningMode::Enabled) | None if config.budget_tokens.is_some() => {
            let mut value = json!({
                "type": "enabled",
                "budget_tokens": config.budget_tokens,
            });
            if let Some(display) = display {
                value["display"] = json!(display);
            }
            Some(value)
        }
        _ => None,
    }
}

enum AnthropicBlockAccumulator {
    Text(String),
    Thinking {
        text: String,
        signature: String,
    },
    Redacted {
        data: String,
    },
    ToolUse {
        id: String,
        name: String,
        input_json: String,
    },
}

pub(crate) struct AnthropicStreamMapper {
    model: String,
    metadata: ResponseMetadata,
    started: bool,
    done: bool,
    blocks: BTreeMap<usize, AnthropicBlockAccumulator>,
    usage: Option<Usage>,
    finish_reason: Option<String>,
    chunks: Vec<Value>,
    report: CompletionReport,
}

impl AnthropicStreamMapper {
    pub(crate) fn new(model: String, metadata: ResponseMetadata) -> Self {
        Self {
            model,
            metadata,
            started: false,
            done: false,
            blocks: BTreeMap::new(),
            usage: None,
            finish_reason: None,
            chunks: Vec::new(),
            report: CompletionReport::default(),
        }
    }

    pub(crate) fn with_report(mut self, report: CompletionReport) -> Self {
        self.report = report;
        self
    }

    fn push_start(&mut self, events: &mut Vec<StreamEvent>) {
        if self.started {
            return;
        }
        self.started = true;
        events.push(StreamEvent::Start {
            provider: "anthropic-messages",
            model: self.model.clone(),
            metadata: self.metadata.clone(),
        });
    }

    fn block_mut(&mut self, index: usize) -> Option<&mut AnthropicBlockAccumulator> {
        self.blocks.get_mut(&index)
    }

    fn done_events(&mut self) -> Vec<StreamEvent> {
        if self.done {
            return Vec::new();
        }
        self.done = true;

        let mut events = Vec::new();
        self.push_start(&mut events);
        let mut parts = Vec::new();

        for block in std::mem::take(&mut self.blocks).into_values() {
            match block {
                AnthropicBlockAccumulator::Text(text) => {
                    if !text.is_empty() {
                        parts.push(Part::text(text));
                    }
                }
                AnthropicBlockAccumulator::Thinking { text, signature } => {
                    let visible = !text.is_empty();
                    parts.push(Part::reasoning(Reasoning {
                        kind: if visible {
                            ReasoningKind::Text
                        } else {
                            ReasoningKind::Encrypted
                        },
                        summary: None,
                        text: visible.then_some(text.clone()),
                        state: Some(ProviderState::new(
                            ProviderStateFormat::AnthropicMessages,
                            json!({
                                "type": "thinking",
                                "thinking": text,
                                "signature": signature,
                            }),
                        )),
                    }));
                }
                AnthropicBlockAccumulator::Redacted { data } => {
                    parts.push(Part::reasoning(Reasoning {
                        kind: ReasoningKind::Redacted,
                        summary: None,
                        text: None,
                        state: Some(ProviderState::new(
                            ProviderStateFormat::AnthropicMessages,
                            json!({"type": "redacted_thinking", "data": data}),
                        )),
                    }));
                }
                AnthropicBlockAccumulator::ToolUse {
                    id,
                    name,
                    input_json,
                } => {
                    let arguments = if input_json.trim().is_empty() {
                        json!({})
                    } else {
                        parse_json_or_string(&input_json)
                    };
                    parts.push(Part::ToolCall(ToolCall {
                        id: (!id.is_empty()).then_some(id),
                        name,
                        arguments,
                        provider_state: None,
                    }));
                }
            }
        }

        let response = ChatResponse {
            message: Message::new(Role::Assistant, parts),
            usage: self.usage.clone(),
            raw: Value::Array(self.chunks.clone()),
            metadata: Some(self.metadata.clone()),
            report: self.report.clone(),
        };
        events.push(StreamEvent::Done {
            response: Box::new(response),
            finish_reason: self.finish_reason.clone(),
        });
        events
    }
}

impl SseMapper for AnthropicStreamMapper {
    fn map(&mut self, message: SseMessage) -> Result<Vec<StreamEvent>> {
        let event: Value = serde_json::from_str(&message.data)?;
        self.chunks.push(event.clone());
        let event_type = event
            .get("type")
            .and_then(Value::as_str)
            .or(message.event.as_deref())
            .unwrap_or_default();

        let mut events = Vec::new();
        self.push_start(&mut events);

        match event_type {
            "message_start" => {
                if let Some(usage) = event.pointer("/message/usage") {
                    anthropic_merge_usage(&mut self.usage, usage);
                }
            }
            "content_block_start" => {
                let index = event.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                let Some(block) = event.get("content_block") else {
                    return Ok(events);
                };
                let accumulator = match block.get("type").and_then(Value::as_str) {
                    Some("text") => AnthropicBlockAccumulator::Text(
                        block
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                    ),
                    Some("thinking") => AnthropicBlockAccumulator::Thinking {
                        text: block
                            .get("thinking")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        signature: block
                            .get("signature")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                    },
                    Some("redacted_thinking") => AnthropicBlockAccumulator::Redacted {
                        data: block
                            .get("data")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                    },
                    Some("tool_use") => AnthropicBlockAccumulator::ToolUse {
                        id: block
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        name: block
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        input_json: block
                            .get("input")
                            .filter(|input| {
                                !input.as_object().is_some_and(|object| object.is_empty())
                            })
                            .map(Value::to_string)
                            .unwrap_or_default(),
                    },
                    _ => return Ok(events),
                };
                self.blocks.insert(index, accumulator);
            }
            "content_block_delta" => {
                let index = event.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                let Some(delta) = event.get("delta") else {
                    return Ok(events);
                };
                match delta.get("type").and_then(Value::as_str) {
                    Some("text_delta") => {
                        if let Some(text) = delta.get("text").and_then(Value::as_str) {
                            if let Some(AnthropicBlockAccumulator::Text(existing)) =
                                self.block_mut(index)
                            {
                                existing.push_str(text);
                            }
                            events.push(StreamEvent::TextDelta {
                                text: text.to_string(),
                            });
                        }
                    }
                    Some("thinking_delta") => {
                        if let Some(text) = delta.get("thinking").and_then(Value::as_str) {
                            if let Some(AnthropicBlockAccumulator::Thinking {
                                text: existing,
                                ..
                            }) = self.block_mut(index)
                            {
                                existing.push_str(text);
                            }
                            events.push(StreamEvent::ReasoningDelta {
                                kind: ReasoningKind::Text,
                                text: text.to_string(),
                            });
                        }
                    }
                    Some("signature_delta") => {
                        if let Some(signature) = delta.get("signature").and_then(Value::as_str)
                            && let Some(AnthropicBlockAccumulator::Thinking {
                                signature: existing,
                                ..
                            }) = self.block_mut(index)
                        {
                            existing.push_str(signature);
                        }
                    }
                    Some("input_json_delta") => {
                        if let Some(partial_json) =
                            delta.get("partial_json").and_then(Value::as_str)
                        {
                            if let Some(AnthropicBlockAccumulator::ToolUse { input_json, .. }) =
                                self.block_mut(index)
                            {
                                input_json.push_str(partial_json);
                            }
                            let (id, name, arguments_so_far) = match self.block_mut(index) {
                                Some(AnthropicBlockAccumulator::ToolUse {
                                    id,
                                    name,
                                    input_json,
                                }) => (Some(id.clone()), Some(name.clone()), input_json.clone()),
                                _ => (None, None, String::new()),
                            };
                            events.push(StreamEvent::ToolCallDelta {
                                index,
                                id: None,
                                name: None,
                                arguments_delta: partial_json.to_string(),
                            });
                            events.push(StreamEvent::ToolCallProgress {
                                index,
                                id,
                                name,
                                arguments_so_far,
                            });
                        }
                    }
                    _ => {}
                }
            }
            "message_delta" => {
                if let Some(usage) = event.get("usage") {
                    anthropic_merge_usage(&mut self.usage, usage);
                    if let Some(usage) = self.usage.clone() {
                        events.push(StreamEvent::Usage { usage });
                    }
                }
                if let Some(stop_reason) =
                    event.pointer("/delta/stop_reason").and_then(Value::as_str)
                {
                    self.finish_reason = Some(stop_reason.to_string());
                }
            }
            "message_stop" => events.extend(self.done_events()),
            "error" => return Err(Error::ProviderStream(event.to_string())),
            "ping" => {}
            _ => {}
        }

        Ok(events)
    }

    fn eof_is_terminal(&self) -> bool {
        false
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>> {
        Ok(self.done_events())
    }
}

fn anthropic_usage(value: &Value) -> Usage {
    let mut usage = None;
    anthropic_merge_usage(&mut usage, value);
    usage.unwrap_or_default()
}

fn anthropic_merge_usage(current: &mut Option<Usage>, value: &Value) {
    let usage = current.get_or_insert_with(Usage::default);
    if let Some(input_tokens) = value.get("input_tokens").and_then(Value::as_u64) {
        usage.input_tokens = Some(input_tokens);
    }
    if let Some(output_tokens) = value.get("output_tokens").and_then(Value::as_u64) {
        usage.output_tokens = Some(output_tokens);
    }
    if let Some(total_tokens) = value.get("total_tokens").and_then(Value::as_u64) {
        usage.total_tokens = Some(total_tokens);
    }
    if let Some(cache_read_tokens) = value.get("cache_read_input_tokens").and_then(Value::as_u64) {
        usage.cache_read_tokens = Some(cache_read_tokens);
    }
    if let Some(cache_write_tokens) = value
        .get("cache_creation_input_tokens")
        .and_then(Value::as_u64)
    {
        usage.cache_write_tokens = Some(cache_write_tokens);
    }
    if let Some(reasoning_tokens) = value.get("reasoning_tokens").and_then(Value::as_u64) {
        usage.reasoning_tokens = Some(reasoning_tokens);
    }
    usage.raw = Some(merge_usage_raw(usage.raw.take(), value));
}

fn merge_usage_raw(current: Option<Value>, next: &Value) -> Value {
    match (current, next) {
        (Some(Value::Object(mut current)), Value::Object(next)) => {
            current.extend(next.clone());
            Value::Object(current)
        }
        (_, next) => next.clone(),
    }
}

fn parse_json_or_string(value: &str) -> Value {
    serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_string()))
}

pub(crate) fn from_anthropic_response(value: Value) -> Result<ChatResponse> {
    let mut parts = Vec::new();

    if let Some(content) = value.get("content").and_then(Value::as_array) {
        for block in content {
            match block.get("type").and_then(Value::as_str) {
                Some("text") => {
                    if let Some(text) = block.get("text").and_then(Value::as_str)
                        && !text.is_empty()
                    {
                        parts.push(Part::text(text));
                    }
                }
                Some("thinking") => {
                    let text = block
                        .get("thinking")
                        .and_then(Value::as_str)
                        .filter(|text| !text.is_empty())
                        .map(str::to_string);
                    parts.push(Part::reasoning(Reasoning {
                        kind: if text.is_some() {
                            ReasoningKind::Text
                        } else {
                            ReasoningKind::Encrypted
                        },
                        summary: None,
                        text,
                        state: Some(ProviderState::new(
                            ProviderStateFormat::AnthropicMessages,
                            block.clone(),
                        )),
                    }));
                }
                Some("redacted_thinking") => {
                    parts.push(Part::reasoning(Reasoning {
                        kind: ReasoningKind::Redacted,
                        summary: None,
                        text: None,
                        state: Some(ProviderState::new(
                            ProviderStateFormat::AnthropicMessages,
                            block.clone(),
                        )),
                    }));
                }
                Some("tool_use") => {
                    parts.push(Part::ToolCall(ToolCall {
                        id: block.get("id").and_then(Value::as_str).map(str::to_string),
                        name: block
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        arguments: block.get("input").cloned().unwrap_or(Value::Null),
                        provider_state: None,
                    }));
                }
                _ => {}
            }
        }
    }

    let usage = value.get("usage").map(anthropic_usage);

    Ok(ChatResponse {
        message: Message::new(Role::Assistant, parts),
        usage,
        raw: value,
        metadata: None,
        report: CompletionReport::default(),
    })
}

fn text_content(message: &crate::normalize::NormalizedMessage) -> String {
    message
        .parts
        .iter()
        .filter_map(|part| match part {
            Part::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize::{Protocol, normalize};
    use crate::types::ResponseFormat;

    #[test]
    fn neutral_request_controls_map_to_anthropic() {
        let request = ChatRequest::user("hi")
            .tools([crate::types::ToolSpec::new(
                "lookup",
                "lookup",
                json!({"type": "object"}),
            )])
            .tool_choice(ToolChoice::Tool("lookup".to_string()))
            .stop(["END"]);
        validate_anthropic_request(&request).unwrap();
        let (normalized, _) = normalize(&request, Protocol::AnthropicMessages).unwrap();
        let body = to_anthropic_body("claude", &normalized, &request);

        assert_eq!(body["tool_choice"]["type"], "tool");
        assert_eq!(body["tool_choice"]["name"], "lookup");
        assert_eq!(body["stop_sequences"], json!(["END"]));
    }

    #[test]
    fn anthropic_rejects_response_format_and_seed() {
        let format = ChatRequest::user("hi").response_format(ResponseFormat::JsonObject);
        assert!(matches!(
            validate_anthropic_request(&format),
            Err(Error::Unsupported(_))
        ));

        let seed = ChatRequest::user("hi").seed(42);
        assert!(matches!(
            validate_anthropic_request(&seed),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn thinking_blocks_round_trip_exactly() {
        let thinking = json!({
            "type": "thinking",
            "thinking": "先分析中文边界。",
            "signature": "sig-thinking"
        });
        let redacted = json!({
            "type": "redacted_thinking",
            "data": "opaque-redacted"
        });
        let response = from_anthropic_response(json!({
            "content": [
                thinking.clone(),
                redacted.clone(),
                {"type": "text", "text": "最终答案"}
            ]
        }))
        .unwrap();

        let reasoning: Vec<_> = response.reasoning().collect();
        assert_eq!(reasoning[0].kind, ReasoningKind::Text);
        assert_eq!(reasoning[0].text.as_deref(), Some("先分析中文边界。"));
        assert_eq!(reasoning[1].kind, ReasoningKind::Redacted);
        assert_eq!(response.text(), "最终答案");

        let request = ChatRequest::new([response.message]);
        let (normalized, _) = normalize(&request, Protocol::AnthropicMessages).unwrap();
        let body = to_anthropic_body("claude-sonnet", &normalized, &request);
        assert_eq!(body.pointer("/messages/0/content/0"), Some(&thinking));
        assert_eq!(body.pointer("/messages/0/content/1"), Some(&redacted));
    }

    #[test]
    fn omitted_thinking_is_encrypted_not_empty_text() {
        let response = from_anthropic_response(json!({
            "content": [{
                "type": "thinking",
                "thinking": "",
                "signature": "sig-omitted"
            }]
        }))
        .unwrap();

        let reasoning = response.reasoning().next().unwrap();
        assert_eq!(reasoning.kind, ReasoningKind::Encrypted);
        assert!(reasoning.text.is_none());
        assert!(reasoning.state.is_some());
    }

    #[test]
    fn adaptive_thinking_config_maps_to_display() {
        let request = ChatRequest::user("hi").reasoning(
            ReasoningConfig::new()
                .mode(ReasoningMode::Adaptive)
                .include_text(false),
        );
        let (normalized, _) = normalize(&request, Protocol::AnthropicMessages).unwrap();
        let body = to_anthropic_body("claude-sonnet", &normalized, &request);
        assert_eq!(
            body["thinking"],
            json!({"type": "adaptive", "display": "omitted"})
        );
    }

    #[test]
    fn stream_assembles_thinking_signature_and_tool_json() {
        let metadata = ResponseMetadata {
            status: 200,
            request_id: Some("req_anthropic".to_string()),
            headers: HeaderMap::new(),
        };
        let mut mapper = AnthropicStreamMapper::new("claude-sonnet".to_string(), metadata);
        let events = [
            json!({
                "type": "message_start",
                "message": {"usage": {"input_tokens": 9, "output_tokens": 0}}
            }),
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "thinking",
                    "thinking": "",
                    "signature": ""
                }
            }),
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "先分析："}
            }),
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "sig_"}
            }),
            json!({"type": "content_block_stop", "index": 0}),
            json!({
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "lookup",
                    "input": {}
                }
            }),
            json!({
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": "{\"q\":"}
            }),
            json!({
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": "\"Rust\"}"}
            }),
            json!({
                "type": "content_block_start",
                "index": 2,
                "content_block": {"type": "text", "text": ""}
            }),
            json!({
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "text_delta", "text": "最终答案"}
            }),
            json!({
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 6}
            }),
            json!({"type": "message_stop"}),
        ];

        let mut stream_events = Vec::new();
        for event in events {
            stream_events.extend(
                mapper
                    .map(SseMessage {
                        event: None,
                        data: event.to_string(),
                        id: None,
                    })
                    .unwrap(),
            );
        }

        assert!(stream_events.iter().any(|event| matches!(
            event,
            StreamEvent::ReasoningDelta { text, .. } if text == "先分析："
        )));
        assert!(stream_events.iter().any(|event| matches!(
            event,
            StreamEvent::TextDelta { text } if text == "最终答案"
        )));

        let StreamEvent::Done {
            response,
            finish_reason,
        } = stream_events.last().unwrap()
        else {
            panic!("expected done event");
        };
        assert_eq!(finish_reason.as_deref(), Some("tool_use"));
        assert_eq!(response.text(), "最终答案");
        assert_eq!(response.reasoning_text(), "先分析：");
        let reasoning = response.reasoning().next().unwrap();
        assert_eq!(
            reasoning
                .state
                .as_ref()
                .and_then(|state| state.data.get("signature"))
                .and_then(Value::as_str),
            Some("sig_")
        );
        assert_eq!(
            response.tool_calls().next().unwrap().arguments,
            json!({"q": "Rust"})
        );
        assert_eq!(response.usage.as_ref().unwrap().output_tokens, Some(6));
    }
}
