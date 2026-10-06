use async_trait::async_trait;
use reqwest::header::HeaderMap;
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
    ReasoningConfig, ReasoningEffort, ReasoningKind, ReasoningMode, ResponseFormat,
    ResponseMetadata, Role, ToolCall, ToolChoice, Usage,
};

/// Entry point for Gemini adapters.
pub struct Gemini;

impl Gemini {
    /// Use the Gemini `generateContent` API.
    pub fn generate_content(model: impl Into<String>) -> GeminiGenerateContent {
        GeminiGenerateContent::new(model)
    }
}

/// Adapter for `POST /v1beta/models/{model}:generateContent`.
pub struct GeminiGenerateContent {
    model: String,
    api_key: Option<String>,
    base_url: String,
    retry_policy: RetryPolicy,
    transport: TransportConfig,
    client: reqwest::Client,
}

impl GeminiGenerateContent {
    /// Create an adapter for the given Gemini model.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            api_key: None,
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            retry_policy: RetryPolicy::default(),
            transport: TransportConfig::default(),
            client: reqwest::Client::new(),
        }
    }

    /// Set an explicit API key, taking precedence over `GEMINI_API_KEY`.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Override the default base URL
    /// (`https://generativelanguage.googleapis.com/v1beta`).
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
impl ModelAdapter for GeminiGenerateContent {
    fn provider_name(&self) -> &'static str {
        "gemini-generate-content"
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
        let (normalized, report) = normalize(request, Protocol::GeminiGenerateContent)?;
        validate_gemini_request(request)?;
        let mut body = to_gemini_body(&normalized, request);
        merge_extra_body(&mut body, request, Protocol::GeminiGenerateContent, None)?;
        apply_provider_request_options(
            &mut body,
            Protocol::GeminiGenerateContent,
            &options.provider_request,
            false,
        )?;
        let request_body_bytes = json_body_bytes(&body)?;
        let key = resolve_api_key(
            self.provider_name(),
            self.api_key.as_deref(),
            "GEMINI_API_KEY",
        )?;
        options.validate_query(
            protocol_name(Protocol::GeminiGenerateContent),
            &["key", "alt"],
        )?;
        let retry_policy = effective_retry_policy(&self.retry_policy, options);
        let audit = self.transport.audit_context(
            self.provider_name(),
            protocol_name(Protocol::GeminiGenerateContent),
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

        let base_headers = HeaderMap::new();

        let result = send_json_retry(
            self.provider_name(),
            RetryProvider::Google,
            &retry_policy,
            audit.as_ref(),
            options.cancellation.as_ref(),
            || async {
                let headers = prepare_headers(&self.transport, &base_headers, options).await?;
                let mut request_builder = self
                    .client
                    .post(format!(
                        "{}/models/{}:generateContent",
                        self.base_url.trim_end_matches('/'),
                        self.model
                    ))
                    .query(&[("key", &key)])
                    .query(&options.extra_query)
                    .headers(headers)
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

        let mut response = match from_gemini_response(value) {
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
            Protocol::GeminiGenerateContent,
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
        let (normalized, report) = normalize(request, Protocol::GeminiGenerateContent)?;
        validate_gemini_request(request)?;
        let mut body = to_gemini_body(&normalized, request);
        merge_extra_body(&mut body, request, Protocol::GeminiGenerateContent, None)?;
        apply_provider_request_options(
            &mut body,
            Protocol::GeminiGenerateContent,
            &options.provider_request,
            false,
        )?;
        let request_body_bytes = json_body_bytes(&body)?;
        let key = resolve_api_key(
            self.provider_name(),
            self.api_key.as_deref(),
            "GEMINI_API_KEY",
        )?;
        options.validate_query(
            protocol_name(Protocol::GeminiGenerateContent),
            &["key", "alt"],
        )?;
        let retry_policy = effective_retry_policy(&self.retry_policy, options);
        let audit = self.transport.audit_context(
            self.provider_name(),
            protocol_name(Protocol::GeminiGenerateContent),
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

        let base_headers = HeaderMap::new();
        let url = format!(
            "{}/models/{}:streamGenerateContent",
            self.base_url.trim_end_matches('/'),
            self.model
        );
        let mut query = vec![
            ("key".to_string(), key.clone()),
            ("alt".to_string(), "sse".to_string()),
        ];
        query.extend(options.extra_query.iter().cloned());

        let response = match send_stream_retry(
            self.provider_name(),
            RetryProvider::Google,
            &retry_policy,
            audit.as_ref(),
            options.cancellation.as_ref(),
            || async {
                let headers = prepare_headers(&self.transport, &base_headers, options).await?;
                let mut request_builder = self
                    .client
                    .post(&url)
                    .query(&query)
                    .headers(headers)
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
        let mapper =
            GeminiStreamMapper::new(self.model.clone(), metadata).with_report(completion_report(
                Protocol::GeminiGenerateContent,
                request,
                report.stats(),
                transform_applied,
            ));
        let reconnect = options.stream_reconnect.map(|_| {
            ReconnectRequest::new(ReconnectRequestParts {
                provider: self.provider_name(),
                client: self.client.clone(),
                retry_policy: retry_policy.clone(),
                retry_provider: RetryProvider::Google,
                transport: self.transport.clone(),
                options: options.clone(),
                audit: audit.clone(),
                url,
                base_headers,
                query,
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

pub(crate) fn to_gemini_body(normalized: &NormalizedChat, request: &ChatRequest) -> Value {
    let mut contents: Vec<Value> = Vec::new();

    for message in &normalized.messages {
        match message.role {
            ExternalRole::System | ExternalRole::Developer => {
                // Gemini always merges system text (see `SystemPlacement`);
                // an in-place developer message is demoted to a user turn.
                push_gemini_content(
                    &mut contents,
                    "user",
                    vec![json!({"text": text_content(message)})],
                );
            }
            ExternalRole::User => {
                push_gemini_content(&mut contents, "user", gemini_content(&message.parts));
            }
            ExternalRole::Assistant => {
                let mut parts = Vec::new();
                for part in &message.parts {
                    match part {
                        Part::Reasoning(reasoning) => {
                            if let Some(state) = &reasoning.state {
                                if state.format == ProviderStateFormat::GeminiGenerateContent {
                                    parts.push(state.data.clone());
                                }
                            } else if let Some(text) = reasoning.display_text() {
                                parts.push(json!({"text": text, "thought": true}));
                            }
                        }
                        Part::Text {
                            text,
                            provider_state,
                        } => {
                            let mut text_part = provider_state
                                .as_ref()
                                .filter(|state| {
                                    state.format == ProviderStateFormat::GeminiGenerateContent
                                })
                                .map(|state| state.data.clone())
                                .unwrap_or_else(|| json!({}));
                            text_part["text"] = json!(text);
                            parts.push(text_part);
                        }
                        Part::ToolCall(call) => {
                            let mut call_part = call
                                .provider_state
                                .as_ref()
                                .filter(|state| {
                                    state.format == ProviderStateFormat::GeminiGenerateContent
                                })
                                .map(|state| state.data.clone())
                                .unwrap_or_else(|| json!({}));
                            call_part["functionCall"] = json!({
                                "name": call.name,
                                "args": call.arguments,
                            });
                            parts.push(call_part);
                        }
                        _ => {}
                    }
                }
                push_gemini_content(&mut contents, "model", parts);
            }
            ExternalRole::Tool => {
                let parts = message
                    .parts
                    .iter()
                    .filter_map(|part| match part {
                        Part::ToolResult(result) => Some(json!({
                            "functionResponse": {
                                "name": result.name,
                                "response": gemini_tool_response(result),
                            }
                        })),
                        _ => None,
                    })
                    .collect();
                push_gemini_content(&mut contents, "user", parts);
            }
        }
    }

    let mut body = json!({
        "contents": contents,
    });

    if let Some(system) = &normalized.system {
        body["systemInstruction"] = json!({"parts": [{"text": system}]});
    }
    if !normalized.tools.is_empty() {
        body["tools"] = json!([{
            "functionDeclarations": normalized.tools.iter().map(|tool| json!({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            })).collect::<Vec<_>>(),
        }]);
    }

    let mut generation_config = serde_json::Map::new();
    if let Some(temperature) = request.temperature {
        generation_config.insert("temperature".to_string(), json!(temperature));
    }
    if let Some(max_output_tokens) = request.max_output_tokens {
        generation_config.insert("maxOutputTokens".to_string(), json!(max_output_tokens));
    }
    if let Some(thinking) = gemini_thinking(request.reasoning.as_ref()) {
        generation_config.insert("thinkingConfig".to_string(), thinking);
    }
    if !request.stop.is_empty() {
        generation_config.insert("stopSequences".to_string(), json!(request.stop));
    }
    if let Some(seed) = request.seed {
        generation_config.insert("seed".to_string(), json!(seed));
    }
    if let Some(response_format) = &request.response_format {
        match response_format {
            ResponseFormat::Text => {
                generation_config.insert("responseMimeType".to_string(), json!("text/plain"));
            }
            ResponseFormat::JsonObject => {
                generation_config.insert("responseMimeType".to_string(), json!("application/json"));
            }
            ResponseFormat::JsonSchema { schema, .. } => {
                generation_config.insert("responseMimeType".to_string(), json!("application/json"));
                generation_config.insert("responseJsonSchema".to_string(), schema.clone());
            }
        }
    }
    if !generation_config.is_empty() {
        body["generationConfig"] = Value::Object(generation_config);
    }
    if let Some(tool_choice) = &request.tool_choice {
        body["toolConfig"] = gemini_tool_config(tool_choice);
    }

    body
}

pub(crate) fn validate_gemini_request(request: &ChatRequest) -> Result<()> {
    validate_server_tools(Protocol::GeminiGenerateContent, request)?;
    validate_tool_choice(&request.tools, request.tool_choice.as_ref())?;
    if let Some(ResponseFormat::JsonSchema {
        strict: Some(true), ..
    }) = &request.response_format
    {
        return Err(Error::Unsupported(
            "Gemini responseJsonSchema does not support a strict flag".to_string(),
        ));
    }
    Ok(())
}

fn gemini_tool_config(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!({
            "functionCallingConfig": {"mode": "AUTO"}
        }),
        ToolChoice::None => json!({
            "functionCallingConfig": {"mode": "NONE"}
        }),
        ToolChoice::Required => json!({
            "functionCallingConfig": {"mode": "ANY"}
        }),
        ToolChoice::Tool(name) => json!({
            "functionCallingConfig": {
                "mode": "ANY",
                "allowedFunctionNames": [name],
            }
        }),
    }
}

fn push_gemini_content(contents: &mut Vec<Value>, role: &str, parts: Vec<Value>) {
    if parts.is_empty() {
        return;
    }

    if let Some(last) = contents.last_mut()
        && last.get("role").and_then(Value::as_str) == Some(role)
    {
        if let Some(existing) = last.get_mut("parts").and_then(Value::as_array_mut) {
            existing.extend(parts);
        }
        return;
    }

    contents.push(json!({"role": role, "parts": parts}));
}

fn gemini_content(parts: &[Part]) -> Vec<Value> {
    parts
        .iter()
        .filter_map(|part| match part {
            Part::Text { text, .. } => Some(json!({"text": text})),
            Part::ImageUrl { image_url } => Some(json!({
                "fileData": {
                    "fileUri": image_url.url,
                }
            })),
            Part::Image { image } => Some(gemini_image_part(&image.source)),
            _ => None,
        })
        .collect()
}

fn gemini_tool_response(result: &crate::types::ToolResult) -> Value {
    match result.parts.as_slice() {
        [crate::types::ToolResultPart::Json { value }] => value.clone(),
        [] => tool_response_value(&result.content),
        parts => {
            let mut response = serde_json::Map::new();
            let mut text = Vec::new();
            for part in parts {
                match part {
                    crate::types::ToolResultPart::Text { text: value } => text.push(value.clone()),
                    crate::types::ToolResultPart::Json { value } => {
                        response.insert("json".to_string(), value.clone());
                    }
                    crate::types::ToolResultPart::Image { image } => {
                        response.insert("image".to_string(), gemini_image_part(&image.source));
                    }
                }
            }
            if !text.is_empty() {
                response.insert("text".to_string(), Value::String(text.join("\n")));
            }
            Value::Object(response)
        }
    }
}

fn gemini_image_part(source: &crate::types::ImageSource) -> Value {
    match source {
        crate::types::ImageSource::Url { url } => json!({
            "fileData": {"fileUri": url},
        }),
        crate::types::ImageSource::Base64 { media_type, data } => json!({
            "inlineData": {
                "mimeType": media_type,
                "data": data,
            },
        }),
        crate::types::ImageSource::FileRef { uri, media_type } => {
            let mut value = json!({
                "fileData": {"fileUri": uri},
            });
            if let Some(media_type) = media_type {
                value["fileData"]["mimeType"] = json!(media_type);
            }
            value
        }
    }
}

fn gemini_thinking(config: Option<&ReasoningConfig>) -> Option<Value> {
    let config = config?;
    let mut value = serde_json::Map::new();

    if config.mode == Some(ReasoningMode::Disabled) {
        value.insert("thinkingBudget".to_string(), json!(0));
    } else if let Some(effort) = config.effort {
        if effort == ReasoningEffort::None {
            value.insert("thinkingBudget".to_string(), json!(0));
        } else {
            value.insert(
                "thinkingLevel".to_string(),
                json!(match effort {
                    ReasoningEffort::None => unreachable!("handled above"),
                    ReasoningEffort::Minimal => "MINIMAL",
                    ReasoningEffort::Low => "LOW",
                    ReasoningEffort::Medium => "MEDIUM",
                    ReasoningEffort::High | ReasoningEffort::XHigh | ReasoningEffort::Max => "HIGH",
                }),
            );
        }
    } else if let Some(budget) = config.budget_tokens {
        value.insert("thinkingBudget".to_string(), json!(budget));
    }

    if let Some(include_thoughts) = config.include_text {
        value.insert("includeThoughts".to_string(), json!(include_thoughts));
    }

    (!value.is_empty()).then_some(Value::Object(value))
}

fn tool_response_value(content: &str) -> Value {
    serde_json::from_str(content).unwrap_or_else(|_| json!({"result": content}))
}

enum GeminiStreamPart {
    Text {
        text: String,
        state: Option<ProviderState>,
    },
    Reasoning {
        text: String,
        state: ProviderState,
    },
    ToolCall {
        name: String,
        arguments: Value,
        state: Option<ProviderState>,
    },
}

pub(crate) struct GeminiStreamMapper {
    model: String,
    metadata: ResponseMetadata,
    started: bool,
    done: bool,
    parts: Vec<GeminiStreamPart>,
    tool_call_index: usize,
    usage: Option<Usage>,
    finish_reason: Option<String>,
    chunks: Vec<Value>,
    report: CompletionReport,
}

impl GeminiStreamMapper {
    pub(crate) fn new(model: String, metadata: ResponseMetadata) -> Self {
        Self {
            model,
            metadata,
            started: false,
            done: false,
            parts: Vec::new(),
            tool_call_index: 0,
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
            provider: "gemini-generate-content",
            model: self.model.clone(),
            metadata: self.metadata.clone(),
        });
    }

    fn push_text(&mut self, text: &str, state: Option<ProviderState>) {
        if let Some(GeminiStreamPart::Text {
            text: existing,
            state: existing_state,
        }) = self.parts.last_mut()
        {
            existing.push_str(text);
            if state.is_some() {
                *existing_state = state;
            }
            return;
        }
        self.parts.push(GeminiStreamPart::Text {
            text: text.to_string(),
            state,
        });
    }

    fn push_reasoning(&mut self, text: &str, state: ProviderState) {
        if let Some(GeminiStreamPart::Reasoning {
            text: existing,
            state: existing_state,
        }) = self.parts.last_mut()
        {
            existing.push_str(text);
            *existing_state = state;
            return;
        }
        self.parts.push(GeminiStreamPart::Reasoning {
            text: text.to_string(),
            state,
        });
    }

    fn done_events(&mut self) -> Vec<StreamEvent> {
        if self.done {
            return Vec::new();
        }
        self.done = true;

        let mut events = Vec::new();
        self.push_start(&mut events);
        let mut parts = Vec::new();

        for part in std::mem::take(&mut self.parts) {
            match part {
                GeminiStreamPart::Text { text, state } => {
                    if !text.is_empty() {
                        if let Some(state) = state {
                            parts.push(Part::text_with_provider_state(
                                text,
                                state.format,
                                state.data,
                            ));
                        } else {
                            parts.push(Part::text(text));
                        }
                    }
                }
                GeminiStreamPart::Reasoning { text, state } => {
                    parts.push(Part::reasoning(Reasoning {
                        kind: if text.is_empty() {
                            ReasoningKind::Encrypted
                        } else {
                            ReasoningKind::Text
                        },
                        summary: None,
                        text: (!text.is_empty()).then_some(text),
                        state: Some(state),
                    }));
                }
                GeminiStreamPart::ToolCall {
                    name,
                    arguments,
                    state,
                } => {
                    parts.push(Part::ToolCall(ToolCall {
                        id: None,
                        name,
                        arguments,
                        provider_state: state,
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

impl SseMapper for GeminiStreamMapper {
    fn map(&mut self, message: SseMessage) -> Result<Vec<StreamEvent>> {
        let chunk: Value = serde_json::from_str(&message.data)?;
        if let Some(error) = chunk.get("error") {
            return Err(Error::ProviderStream(error.to_string()));
        }
        self.chunks.push(chunk.clone());

        let mut events = Vec::new();
        self.push_start(&mut events);

        if let Some(usage) = chunk.get("usageMetadata").and_then(gemini_usage) {
            self.usage = Some(usage.clone());
            events.push(StreamEvent::Usage { usage });
        }

        let Some(candidate) = chunk.pointer("/candidates/0") else {
            return Ok(events);
        };
        if let Some(finish_reason) = candidate.get("finishReason").and_then(Value::as_str) {
            self.finish_reason = Some(finish_reason.to_string());
        }

        let Some(parts) = candidate
            .pointer("/content/parts")
            .and_then(Value::as_array)
        else {
            return Ok(events);
        };

        for part in parts {
            if part.get("thought").and_then(Value::as_bool) == Some(true) {
                let text = part.get("text").and_then(Value::as_str).unwrap_or_default();
                let state =
                    ProviderState::new(ProviderStateFormat::GeminiGenerateContent, part.clone());
                self.push_reasoning(text, state);
                if !text.is_empty() {
                    events.push(StreamEvent::ReasoningDelta {
                        kind: ReasoningKind::Text,
                        text: text.to_string(),
                    });
                }
                continue;
            }

            if let Some(text) = part.get("text").and_then(Value::as_str) {
                let state = part.get("thoughtSignature").map(|_| {
                    ProviderState::new(ProviderStateFormat::GeminiGenerateContent, part.clone())
                });
                self.push_text(text, state);
                if !text.is_empty() {
                    events.push(StreamEvent::TextDelta {
                        text: text.to_string(),
                    });
                }
            }

            if let Some(call) = part.get("functionCall") {
                let name = call
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let arguments = call.get("args").cloned().unwrap_or_else(|| json!({}));
                let state = part.get("thoughtSignature").map(|_| {
                    ProviderState::new(ProviderStateFormat::GeminiGenerateContent, part.clone())
                });
                events.push(StreamEvent::ToolCallDelta {
                    index: self.tool_call_index,
                    id: None,
                    name: Some(name.clone()),
                    arguments_delta: arguments.to_string(),
                });
                events.push(StreamEvent::ToolCallProgress {
                    index: self.tool_call_index,
                    id: None,
                    name: Some(name.clone()),
                    arguments_so_far: arguments.to_string(),
                });
                self.tool_call_index += 1;
                self.parts.push(GeminiStreamPart::ToolCall {
                    name,
                    arguments,
                    state,
                });
            }
        }

        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>> {
        Ok(self.done_events())
    }
}

fn gemini_usage(usage: &Value) -> Option<Usage> {
    Some(Usage {
        input_tokens: usage.get("promptTokenCount").and_then(Value::as_u64),
        output_tokens: usage.get("candidatesTokenCount").and_then(Value::as_u64),
        total_tokens: usage.get("totalTokenCount").and_then(Value::as_u64),
        cache_read_tokens: usage.get("cachedContentTokenCount").and_then(Value::as_u64),
        cache_write_tokens: None,
        reasoning_tokens: usage.get("thoughtsTokenCount").and_then(Value::as_u64),
        raw: Some(usage.clone()),
    })
}

pub(crate) fn from_gemini_response(value: Value) -> Result<ChatResponse> {
    let mut parts = Vec::new();

    if let Some(parts_value) = value
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
    {
        for part in parts_value {
            if part.get("thought").and_then(Value::as_bool) == Some(true) {
                let text = part.get("text").and_then(Value::as_str);
                let kind = if text.is_some() {
                    ReasoningKind::Text
                } else if part.get("thoughtSignature").is_some() {
                    ReasoningKind::Encrypted
                } else {
                    ReasoningKind::Redacted
                };
                parts.push(Part::reasoning(Reasoning {
                    kind,
                    summary: None,
                    text: text.map(str::to_string),
                    state: Some(ProviderState::new(
                        ProviderStateFormat::GeminiGenerateContent,
                        part.clone(),
                    )),
                }));
                continue;
            }

            if let Some(text) = part.get("text").and_then(Value::as_str)
                && !text.is_empty()
            {
                if part.get("thoughtSignature").is_some() {
                    parts.push(Part::text_with_provider_state(
                        text,
                        ProviderStateFormat::GeminiGenerateContent,
                        part.clone(),
                    ));
                } else {
                    parts.push(Part::text(text));
                }
            } else if part.get("thoughtSignature").is_some() && part.get("functionCall").is_none() {
                parts.push(Part::reasoning(Reasoning {
                    kind: ReasoningKind::Encrypted,
                    summary: None,
                    text: None,
                    state: Some(ProviderState::new(
                        ProviderStateFormat::GeminiGenerateContent,
                        part.clone(),
                    )),
                }));
            }

            if let Some(call) = part.get("functionCall") {
                parts.push(Part::ToolCall(ToolCall {
                    id: None,
                    name: call
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    arguments: call.get("args").cloned().unwrap_or(Value::Null),
                    provider_state: Some(ProviderState::new(
                        ProviderStateFormat::GeminiGenerateContent,
                        part.clone(),
                    )),
                }));
            }
        }
    }

    let usage = value.get("usageMetadata").and_then(gemini_usage);

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

    #[test]
    fn neutral_request_controls_map_to_gemini() {
        let request = ChatRequest::user("hi")
            .tools([crate::types::ToolSpec::new(
                "lookup",
                "lookup",
                json!({"type": "object"}),
            )])
            .tool_choice(ToolChoice::Tool("lookup".to_string()))
            .response_format(ResponseFormat::JsonSchema {
                name: "answer".to_string(),
                schema: json!({"type": "object"}),
                strict: None,
            })
            .stop(["END"])
            .seed(42);
        validate_gemini_request(&request).unwrap();
        let (normalized, _) = normalize(&request, Protocol::GeminiGenerateContent).unwrap();
        let body = to_gemini_body(&normalized, &request);

        assert_eq!(body["toolConfig"]["functionCallingConfig"]["mode"], "ANY");
        assert_eq!(
            body["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"],
            json!(["lookup"])
        );
        assert_eq!(
            body["generationConfig"]["responseMimeType"],
            "application/json"
        );
        assert_eq!(body["generationConfig"]["stopSequences"], json!(["END"]));
        assert_eq!(body["generationConfig"]["seed"], 42);
    }

    #[test]
    fn gemini_rejects_strict_json_schema() {
        let request = ChatRequest::user("hi").response_format(ResponseFormat::JsonSchema {
            name: "answer".to_string(),
            schema: json!({"type": "object"}),
            strict: Some(true),
        });

        assert!(matches!(
            validate_gemini_request(&request),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn thought_and_function_signature_round_trip() {
        let thought = json!({
            "text": "先规划工具参数。",
            "thought": true,
            "thoughtSignature": "thought-sig"
        });
        let function_part = json!({
            "functionCall": {"name": "lookup", "args": {"q": "Rust"}},
            "thoughtSignature": "call-sig"
        });
        let response = from_gemini_response(json!({
            "candidates": [{
                "content": {
                    "parts": [
                        thought.clone(),
                        function_part.clone(),
                        {"text": "done"}
                    ]
                }
            }]
        }))
        .unwrap();

        let reasoning = response.reasoning().next().unwrap();
        assert_eq!(reasoning.kind, ReasoningKind::Text);
        assert_eq!(reasoning.text.as_deref(), Some("先规划工具参数。"));
        let call = response.tool_calls().next().unwrap();
        assert_eq!(
            call.provider_state.as_ref().map(|state| &state.data),
            Some(&function_part)
        );

        let request = ChatRequest::new([response.message]);
        let (normalized, _) = normalize(&request, Protocol::GeminiGenerateContent).unwrap();
        let body = to_gemini_body(&normalized, &request);
        assert_eq!(body.pointer("/contents/0/parts/0"), Some(&thought));
        assert_eq!(
            body.pointer("/contents/0/parts/1/thoughtSignature")
                .and_then(Value::as_str),
            Some("call-sig")
        );
        assert_eq!(
            body.pointer("/contents/0/parts/1/functionCall/name")
                .and_then(Value::as_str),
            Some("lookup")
        );
    }

    #[test]
    fn text_signature_stays_on_visible_message() {
        let text_part = json!({
            "text": "Hello",
            "thoughtSignature": "text-sig"
        });
        let response = from_gemini_response(json!({
            "candidates": [{
                "content": {"parts": [text_part.clone()]}
            }]
        }))
        .unwrap();

        assert_eq!(response.text(), "Hello");
        let Part::Text {
            provider_state: Some(state),
            ..
        } = &response.message.parts[0]
        else {
            panic!("expected text part with provider state");
        };
        assert_eq!(state.data, text_part);

        let request = ChatRequest::new([response.message]);
        let (normalized, _) = normalize(&request, Protocol::GeminiGenerateContent).unwrap();
        let body = to_gemini_body(&normalized, &request);
        assert_eq!(body.pointer("/contents/0/parts/0"), Some(&text_part));
    }

    #[test]
    fn thinking_level_and_include_thoughts_are_sent() {
        let request = ChatRequest::user("hi").reasoning(
            ReasoningConfig::new()
                .effort(ReasoningEffort::High)
                .include_text(true),
        );
        let (normalized, _) = normalize(&request, Protocol::GeminiGenerateContent).unwrap();
        let body = to_gemini_body(&normalized, &request);
        assert_eq!(
            body.pointer("/generationConfig/thinkingConfig/thinkingLevel"),
            Some(&json!("HIGH"))
        );
        assert_eq!(
            body.pointer("/generationConfig/thinkingConfig/includeThoughts"),
            Some(&json!(true))
        );
    }

    #[test]
    fn stream_merges_text_and_preserves_signatures() {
        let metadata = ResponseMetadata {
            status: 200,
            request_id: Some("req_gemini".to_string()),
            headers: HeaderMap::new(),
        };
        let mut mapper = GeminiStreamMapper::new("gemini-3-flash".to_string(), metadata);
        let chunks = [
            json!({
                "candidates": [{
                    "content": {"parts": [
                        {
                            "text": "思考",
                            "thought": true,
                            "thoughtSignature": "thought-sig"
                        },
                        {
                            "text": "你",
                            "thoughtSignature": "text-sig"
                        }
                    ]}
                }]
            }),
            json!({
                "candidates": [{
                    "content": {"parts": [
                        {"text": "好"},
                        {
                            "functionCall": {"name": "lookup", "args": {"q": "Rust"}},
                            "thoughtSignature": "call-sig"
                        }
                    ]}
                }],
                "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 3}
            }),
            json!({
                "candidates": [{
                    "content": {"parts": []},
                    "finishReason": "STOP"
                }]
            }),
        ];

        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(
                mapper
                    .map(SseMessage {
                        event: None,
                        data: chunk.to_string(),
                        id: None,
                    })
                    .unwrap(),
            );
        }
        events.extend(mapper.finish().unwrap());

        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::TextDelta { text } if text == "你"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::ReasoningDelta { text, .. } if text == "思考"
        )));

        let StreamEvent::Done {
            response,
            finish_reason,
        } = events.last().unwrap()
        else {
            panic!("expected done event");
        };
        assert_eq!(finish_reason.as_deref(), Some("STOP"));
        assert_eq!(response.text(), "你好");
        assert_eq!(response.reasoning_text(), "思考");
        assert_eq!(response.usage.as_ref().unwrap().input_tokens, Some(8));
        let call = response.tool_calls().next().unwrap();
        assert_eq!(call.name, "lookup");
        assert_eq!(call.arguments, json!({"q": "Rust"}));
        assert!(call.provider_state.is_some());
    }
}
