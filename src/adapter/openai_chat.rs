use std::collections::BTreeMap;

use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
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
use crate::normalize::{NormalizedChat, NormalizedMessage, Protocol, normalize};
use crate::profile::{
    MaxTokensSemantics, ModelProfile, ProviderProfile, ReasoningAliases, ReasoningReplayPolicy,
    ThinkingRequestProfile,
};
use crate::report::CompletionReport;
use crate::retry::{RetryPolicy, RetryProvider};
use crate::sse::SseMessage;
use crate::stream::{ModelStream, SseMapper, StreamEvent};
use crate::types::{
    ChatRequest, ChatResponse, Message, Part, ProviderState, ProviderStateFormat, Reasoning,
    ReasoningConfig, ReasoningKind, ReasoningMode, ResponseFormat, ResponseMetadata, Role,
    ToolCall, ToolChoice, Usage,
};

/// Entry point for OpenAI-compatible adapters.
pub struct OpenAI;

impl OpenAI {
    /// Use the OpenAI Chat Completions API.
    pub fn chat(model: impl Into<String>) -> OpenAIChat {
        OpenAIChat::new(model)
    }

    /// Use the OpenAI Responses API.
    pub fn responses(model: impl Into<String>) -> super::OpenAIResponses {
        super::OpenAIResponses::new(model)
    }
}

/// Adapter for `POST /v1/chat/completions`.
pub struct OpenAIChat {
    model: String,
    api_key: Option<String>,
    base_url: String,
    retry_policy: RetryPolicy,
    transport: TransportConfig,
    client: reqwest::Client,
}

impl OpenAIChat {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            api_key: None,
            base_url: "https://api.openai.com/v1".to_string(),
            retry_policy: RetryPolicy::default(),
            transport: TransportConfig::default(),
            client: reqwest::Client::new(),
        }
    }

    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn retry_policy(mut self, retry_policy: RetryPolicy) -> Self {
        self.retry_policy = retry_policy;
        self
    }

    pub fn transport(mut self, transport: TransportConfig) -> Self {
        self.transport = transport;
        self
    }
}

#[async_trait]
impl ModelAdapter for OpenAIChat {
    fn provider_name(&self) -> &'static str {
        "openai-chat"
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
        let (normalized, report) = normalize(request, Protocol::OpenAiChat)?;
        let mut body = to_openai_chat_body(&self.model, &normalized, request, None)?;
        apply_provider_request_options(
            &mut body,
            Protocol::OpenAiChat,
            &options.provider_request,
            false,
        )?;
        let request_body_bytes = json_body_bytes(&body)?;
        let key = resolve_api_key(
            self.provider_name(),
            self.api_key.as_deref(),
            "OPENAI_API_KEY",
        )?;
        options.validate_query(protocol_name(Protocol::OpenAiChat), &[])?;
        let retry_policy = effective_retry_policy(&self.retry_policy, options);
        let audit = self.transport.audit_context(
            self.provider_name(),
            protocol_name(Protocol::OpenAiChat),
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
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {key}"))?,
        );

        let result = send_json_retry(
            self.provider_name(),
            RetryProvider::OpenAi,
            &retry_policy,
            audit.as_ref(),
            options.cancellation.as_ref(),
            || async {
                let headers = prepare_headers(&self.transport, &base_headers, options).await?;
                let mut request_builder = self
                    .client
                    .post(format!(
                        "{}/chat/completions",
                        self.base_url.trim_end_matches('/')
                    ))
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

        let mut response = match from_openai_chat_response(value) {
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
            Protocol::OpenAiChat,
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
        let (normalized, report) = normalize(request, Protocol::OpenAiChat)?;
        let mut body = to_openai_chat_body(&self.model, &normalized, request, None)?;
        body["stream"] = json!(true);
        body["stream_options"] = json!({"include_usage": true});
        apply_provider_request_options(
            &mut body,
            Protocol::OpenAiChat,
            &options.provider_request,
            true,
        )?;
        let request_body_bytes = json_body_bytes(&body)?;

        let key = resolve_api_key(
            self.provider_name(),
            self.api_key.as_deref(),
            "OPENAI_API_KEY",
        )?;
        options.validate_query(protocol_name(Protocol::OpenAiChat), &[])?;
        let retry_policy = effective_retry_policy(&self.retry_policy, options);
        let audit = self.transport.audit_context(
            self.provider_name(),
            protocol_name(Protocol::OpenAiChat),
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
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {key}"))?,
        );
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));

        let response = match send_stream_retry(
            self.provider_name(),
            RetryProvider::OpenAi,
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
        let mapper = OpenAiChatStreamMapper::new(self.model.clone(), metadata).with_report(
            completion_report(
                Protocol::OpenAiChat,
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
                retry_provider: RetryProvider::OpenAi,
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

struct OpenAiChatWireProfile {
    aliases: ReasoningAliases,
    replay: ReasoningReplayPolicy,
    thinking: ThinkingRequestProfile,
    max_tokens_semantics: MaxTokensSemantics,
}

impl OpenAiChatWireProfile {
    fn resolve(provider: Option<&ProviderProfile>, model: Option<&ModelProfile>) -> Self {
        let provider_aliases = provider
            .map(|profile| profile.reasoning.aliases.clone())
            .unwrap_or_default();
        let aliases = model
            .filter(|profile| !profile.reasoning_aliases.is_empty())
            .map(|profile| profile.reasoning_aliases.clone())
            .unwrap_or(provider_aliases);
        let replay = model
            .and_then(|profile| profile.replay)
            .or_else(|| provider.map(|profile| profile.reasoning.replay))
            .unwrap_or(ReasoningReplayPolicy::SameProvider);
        let thinking = model
            .filter(|profile| profile.thinking != ThinkingRequestProfile::None)
            .map(|profile| profile.thinking.clone())
            .or_else(|| provider.map(|profile| profile.reasoning.thinking.clone()))
            .unwrap_or(ThinkingRequestProfile::None);
        let max_tokens_semantics = model
            .map(|profile| profile.max_tokens_semantics)
            .or_else(|| provider.map(|profile| profile.max_tokens_semantics))
            .unwrap_or(MaxTokensSemantics::MaxCompletionTokens);

        Self {
            aliases,
            replay,
            thinking,
            max_tokens_semantics,
        }
    }
}

pub(crate) fn to_openai_chat_body(
    model: &str,
    normalized: &NormalizedChat,
    request: &ChatRequest,
    profile: Option<&ProviderProfile>,
) -> Result<Value> {
    to_openai_chat_body_with_profile(model, normalized, request, profile, None)
}

pub(crate) fn to_openai_chat_body_with_profile(
    model: &str,
    normalized: &NormalizedChat,
    request: &ChatRequest,
    provider: Option<&ProviderProfile>,
    model_profile: Option<&ModelProfile>,
) -> Result<Value> {
    validate_server_tools(Protocol::OpenAiChat, request)?;
    let wire = OpenAiChatWireProfile::resolve(provider, model_profile);
    let mut messages = Vec::new();

    if let Some(system) = &normalized.system {
        messages.push(json!({"role": "system", "content": system}));
    }

    for message in &normalized.messages {
        match message.role {
            crate::normalize::ExternalRole::System => {
                messages.push(json!({"role": "system", "content": text_content(message)}));
            }
            crate::normalize::ExternalRole::User => {
                messages.push(json!({
                    "role": "user",
                    "content": content_value(&message.parts),
                }));
            }
            crate::normalize::ExternalRole::Assistant => {
                let text = text_content(message);
                let tool_calls: Vec<Value> = message
                    .parts
                    .iter()
                    .filter_map(|part| match part {
                        Part::ToolCall(call) => Some(json!({
                            "id": call.id.clone().unwrap_or_else(|| call.name.clone()),
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": serde_json::to_string(&call.arguments)
                                    .unwrap_or_else(|_| "{}".to_string()),
                            }
                        })),
                        _ => None,
                    })
                    .collect();

                let mut replay_text: BTreeMap<String, Vec<String>> = BTreeMap::new();
                let mut replay_opaque: BTreeMap<String, Value> = BTreeMap::new();
                let mut replay_details = Vec::new();

                for part in &message.parts {
                    let Part::Reasoning(reasoning_part) = part else {
                        continue;
                    };
                    if !should_replay_reasoning(
                        wire.replay,
                        reasoning_part,
                        !tool_calls.is_empty(),
                        &wire.aliases,
                    ) {
                        continue;
                    }

                    let state = reasoning_part
                        .state
                        .as_ref()
                        .filter(|state| state.format == ProviderStateFormat::OpenAiChat);
                    let field = state
                        .and_then(|state| state.data.get("field"))
                        .and_then(Value::as_str);
                    let state_value = state.and_then(|state| state.data.get("value"));

                    if field == Some("reasoning_details") {
                        match state_value {
                            Some(Value::Array(values)) => replay_details.extend(values.clone()),
                            Some(value) => replay_details.push(value.clone()),
                            None => {}
                        }
                        continue;
                    }

                    let field = field.unwrap_or_else(|| {
                        wire.aliases
                            .request_replay
                            .first()
                            .map(String::as_str)
                            .unwrap_or("reasoning_content")
                    });
                    ensure_safe_reasoning_field(field)?;

                    if let Some(Value::String(value)) = state_value {
                        replay_text
                            .entry(field.to_string())
                            .or_default()
                            .push(value.clone());
                    } else if let Some(value) = state_value {
                        replay_opaque
                            .entry(field.to_string())
                            .or_insert(value.clone());
                    } else if let Some(value) = reasoning_part.display_text() {
                        replay_text
                            .entry(field.to_string())
                            .or_default()
                            .push(value.to_string());
                    }
                }

                let mut value = json!({
                    "role": "assistant",
                    "content": if text.is_empty() { Value::Null } else { Value::String(text) },
                });
                for (field, values) in replay_text {
                    ensure_safe_reasoning_field(&field)?;
                    value[field] = Value::String(values.join("\n"));
                }
                for (field, opaque) in replay_opaque {
                    ensure_safe_reasoning_field(&field)?;
                    value[field] = opaque;
                }
                if !replay_details.is_empty() {
                    value["reasoning_details"] = Value::Array(replay_details);
                }
                if !tool_calls.is_empty() {
                    value["tool_calls"] = Value::Array(tool_calls);
                }
                messages.push(value);
            }
            crate::normalize::ExternalRole::Tool => {
                for result in message.parts.iter().filter_map(|part| match part {
                    Part::ToolResult(result) => Some(result),
                    _ => None,
                }) {
                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": result.call_id.clone().unwrap_or_else(|| result.name.clone()),
                        "content": result.content,
                    }));
                }
            }
        }
    }

    let mut body = json!({
        "model": model,
        "messages": messages,
    });

    if !normalized.tools.is_empty() {
        body["tools"] = Value::Array(
            normalized
                .tools
                .iter()
                .map(|tool| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        }
                    })
                })
                .collect(),
        );
    }
    if let Some(temperature) = request.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(max_output_tokens) = request.max_output_tokens {
        let field = match wire.max_tokens_semantics {
            MaxTokensSemantics::MaxTokens => "max_tokens",
            MaxTokensSemantics::MaxOutputTokens | MaxTokensSemantics::MaxCompletionTokens => {
                "max_completion_tokens"
            }
        };
        body[field] = json!(max_output_tokens);
    }
    validate_tool_choice(&normalized.tools, request.tool_choice.as_ref())?;
    if let Some(tool_choice) = &request.tool_choice {
        body["tool_choice"] = openai_tool_choice(tool_choice);
    }
    if let Some(response_format) = &request.response_format {
        body["response_format"] = openai_response_format(response_format);
    }
    if !request.stop.is_empty() {
        body["stop"] = json!(request.stop);
    }
    if let Some(seed) = request.seed {
        body["seed"] = json!(seed);
    }
    apply_openai_chat_thinking(&mut body, request.reasoning.as_ref(), &wire.thinking)?;

    merge_extra_body(&mut body, request, Protocol::OpenAiChat, provider)?;
    Ok(body)
}

fn openai_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Tool(name) => json!({
            "type": "function",
            "function": {"name": name},
        }),
    }
}

fn openai_response_format(format: &ResponseFormat) -> Value {
    match format {
        ResponseFormat::Text => json!({"type": "text"}),
        ResponseFormat::JsonObject => json!({"type": "json_object"}),
        ResponseFormat::JsonSchema {
            name,
            schema,
            strict,
        } => json!({
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "schema": schema,
                "strict": strict.unwrap_or(false),
            }
        }),
    }
}

fn should_replay_reasoning(
    policy: ReasoningReplayPolicy,
    reasoning: &Reasoning,
    has_tool_calls: bool,
    aliases: &ReasoningAliases,
) -> bool {
    match policy {
        ReasoningReplayPolicy::Never => false,
        ReasoningReplayPolicy::SameProvider
        | ReasoningReplayPolicy::PreserveInHistory
        | ReasoningReplayPolicy::ModelDefined => true,
        ReasoningReplayPolicy::RequiredForToolCalls => has_tool_calls,
        ReasoningReplayPolicy::EncryptedOnly => {
            if reasoning.kind == ReasoningKind::Encrypted {
                return true;
            }
            reasoning
                .state
                .as_ref()
                .filter(|state| state.format == ProviderStateFormat::OpenAiChat)
                .and_then(|state| state.data.get("field"))
                .and_then(Value::as_str)
                .is_some_and(|field| {
                    field == "reasoning_details"
                        || aliases.encrypted.iter().any(|alias| alias == field)
                })
        }
    }
}

fn ensure_safe_reasoning_field(field: &str) -> Result<()> {
    if matches!(field, "role" | "content" | "tool_calls" | "tool_call_id") {
        return Err(Error::InvalidProfile(format!(
            "reasoning alias `{field}` would overwrite an assistant message field"
        )));
    }
    Ok(())
}

fn apply_openai_chat_thinking(
    body: &mut Value,
    config: Option<&ReasoningConfig>,
    profile: &ThinkingRequestProfile,
) -> Result<()> {
    if matches!(profile, ThinkingRequestProfile::None) {
        if let Some(effort) = openai_chat_reasoning_effort(config) {
            body["reasoning_effort"] = json!(effort);
        }
        return Ok(());
    }

    apply_thinking_profile(body, config, profile)
}

fn apply_thinking_profile(
    body: &mut Value,
    config: Option<&ReasoningConfig>,
    profile: &ThinkingRequestProfile,
) -> Result<()> {
    match profile {
        ThinkingRequestProfile::None => {}
        ThinkingRequestProfile::Field { path, value } => {
            set_thinking_field(body, path, value.clone())?;
        }
        ThinkingRequestProfile::Composite(parts) => {
            for part in parts {
                apply_thinking_profile(body, config, part)?;
            }
        }
        ThinkingRequestProfile::EnabledFlag(field) => {
            let Some(config) = config else {
                return Ok(());
            };
            ensure_safe_thinking_field(field)?;
            body[field] = json!(config.mode != Some(ReasoningMode::Disabled));
        }
        ThinkingRequestProfile::ThinkingObject => {
            let Some(config) = config else {
                return Ok(());
            };
            ensure_safe_thinking_field("thinking")?;
            let enabled = config.mode != Some(ReasoningMode::Disabled);
            let mut thinking = serde_json::Map::new();
            thinking.insert(
                "type".to_string(),
                json!(if enabled { "enabled" } else { "disabled" }),
            );
            if let Some(budget) = config.budget_tokens {
                thinking.insert("budget_tokens".to_string(), json!(budget));
            }
            body["thinking"] = Value::Object(thinking);
        }
        ThinkingRequestProfile::Effort(field) => {
            let Some(config) = config else {
                return Ok(());
            };
            ensure_safe_thinking_field(field)?;
            if let Some(effort) = openai_chat_reasoning_effort(Some(config)) {
                body[field] = json!(effort);
            }
        }
        ThinkingRequestProfile::MappedEffort { field, mapping } => {
            let Some(config) = config else {
                return Ok(());
            };
            ensure_safe_thinking_field(field)?;
            if let Some(effort) = config.effort {
                if let Some(mapped) = mapping.map(effort) {
                    body[field] = json!(mapped);
                }
            } else if config.mode == Some(ReasoningMode::Disabled)
                && let Some(mapped) = mapping.map(crate::types::ReasoningEffort::None)
            {
                body[field] = json!(mapped);
            }
        }
        ThinkingRequestProfile::BudgetTokens(field) => {
            let Some(config) = config else {
                return Ok(());
            };
            ensure_safe_thinking_field(field)?;
            if config.mode != Some(ReasoningMode::Disabled)
                && let Some(budget) = config.budget_tokens
            {
                body[field] = json!(budget);
            }
        }
    }

    Ok(())
}

fn set_thinking_field(body: &mut Value, path: &str, value: Value) -> Result<()> {
    let segments: Vec<_> = path
        .split('.')
        .filter(|segment| !segment.is_empty())
        .collect();
    let Some((last, parents)) = segments.split_last() else {
        return Err(Error::InvalidProfile(
            "thinking field path cannot be empty".to_string(),
        ));
    };
    ensure_safe_thinking_field(parents.first().unwrap_or(last))?;

    let mut current = body;
    for segment in parents {
        if !current.is_object() {
            return Err(Error::InvalidProfile(format!(
                "thinking field path `{path}` crosses a non-object value"
            )));
        }
        current = current
            .as_object_mut()
            .expect("checked above")
            .entry((*segment).to_string())
            .or_insert_with(|| json!({}));
    }

    let Some(object) = current.as_object_mut() else {
        return Err(Error::InvalidProfile(format!(
            "thinking field path `{path}` does not target an object"
        )));
    };
    object.insert((*last).to_string(), value);
    Ok(())
}

fn ensure_safe_thinking_field(field: &str) -> Result<()> {
    if matches!(
        field,
        "model"
            | "messages"
            | "stream"
            | "stream_options"
            | "tools"
            | "temperature"
            | "max_tokens"
            | "max_completion_tokens"
    ) {
        return Err(Error::InvalidProfile(format!(
            "thinking field `{field}` would overwrite a canonical request field"
        )));
    }
    Ok(())
}

#[derive(Default)]
struct OpenAiToolCallAccumulator {
    id: Option<String>,
    name: String,
    arguments: String,
}

pub(crate) struct OpenAiChatStreamMapper {
    model: String,
    metadata: ResponseMetadata,
    aliases: ReasoningAliases,
    started: bool,
    done: bool,
    text: String,
    reasoning_text: BTreeMap<String, String>,
    reasoning_summary: BTreeMap<String, String>,
    reasoning_opaque: BTreeMap<String, Value>,
    reasoning_details: Vec<Value>,
    tool_calls: BTreeMap<usize, OpenAiToolCallAccumulator>,
    usage: Option<Usage>,
    finish_reason: Option<String>,
    chunks: Vec<Value>,
    report: CompletionReport,
}

impl OpenAiChatStreamMapper {
    pub(crate) fn new(model: String, metadata: ResponseMetadata) -> Self {
        Self::with_aliases(model, metadata, ReasoningAliases::default())
    }

    pub(crate) fn with_aliases(
        model: String,
        metadata: ResponseMetadata,
        aliases: ReasoningAliases,
    ) -> Self {
        Self {
            model,
            metadata,
            aliases,
            started: false,
            done: false,
            text: String::new(),
            reasoning_text: BTreeMap::new(),
            reasoning_summary: BTreeMap::new(),
            reasoning_opaque: BTreeMap::new(),
            reasoning_details: Vec::new(),
            tool_calls: BTreeMap::new(),
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
            provider: "openai-chat",
            model: self.model.clone(),
            metadata: self.metadata.clone(),
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

        for (field, text) in &self.reasoning_text {
            if !text.is_empty() {
                parts.push(Part::reasoning(openai_chat_reasoning(
                    ReasoningKind::Text,
                    Some(text),
                    field,
                    Value::String(text.clone()),
                )));
            }
        }
        for (field, text) in &self.reasoning_summary {
            if !text.is_empty() {
                parts.push(Part::reasoning(openai_chat_reasoning(
                    ReasoningKind::Summary,
                    Some(text),
                    field,
                    Value::String(text.clone()),
                )));
            }
        }
        for (field, value) in &self.reasoning_opaque {
            parts.push(Part::reasoning(openai_chat_reasoning(
                ReasoningKind::Encrypted,
                None,
                field,
                value.clone(),
            )));
        }
        for detail in &self.reasoning_details {
            parts.push(Part::reasoning(openai_chat_reasoning_from_detail(detail)));
        }
        if !self.text.is_empty() {
            parts.push(Part::text(&self.text));
        }
        for (index, call) in &self.tool_calls {
            let arguments = if call.arguments.trim().is_empty() {
                json!({})
            } else {
                parse_json_or_string(&call.arguments)
            };
            parts.push(Part::ToolCall(ToolCall {
                id: call.id.clone().or_else(|| Some(format!("call_{index}"))),
                name: call.name.clone(),
                arguments,
                provider_state: None,
            }));
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

impl SseMapper for OpenAiChatStreamMapper {
    fn map(&mut self, message: SseMessage) -> Result<Vec<StreamEvent>> {
        if message.data.trim() == "[DONE]" {
            return Ok(self.done_events());
        }

        let chunk: Value = serde_json::from_str(&message.data)?;
        if let Some(error) = chunk.get("error") {
            return Err(Error::ProviderStream(error.to_string()));
        }
        self.chunks.push(chunk.clone());

        let mut events = Vec::new();
        self.push_start(&mut events);

        if let Some(usage) = chunk.get("usage").and_then(openai_chat_usage) {
            self.usage = Some(usage.clone());
            events.push(StreamEvent::Usage { usage });
        }

        let Some(choice) = chunk.pointer("/choices/0") else {
            return Ok(events);
        };

        if let Some(finish_reason) = choice.get("finish_reason").and_then(Value::as_str) {
            self.finish_reason = Some(finish_reason.to_string());
        }

        let Some(delta) = choice.get("delta") else {
            return Ok(events);
        };

        for field in &self.aliases.response_text {
            if field == "reasoning_details" {
                continue;
            }
            if let Some(text) = delta.get(field).and_then(Value::as_str) {
                self.reasoning_text
                    .entry(field.clone())
                    .or_default()
                    .push_str(text);
                events.push(StreamEvent::ReasoningDelta {
                    kind: ReasoningKind::Text,
                    text: text.to_string(),
                });
            }
        }
        for field in &self.aliases.response_summary {
            if field == "reasoning_details" {
                continue;
            }
            if let Some(text) = delta.get(field).and_then(Value::as_str) {
                self.reasoning_summary
                    .entry(field.clone())
                    .or_default()
                    .push_str(text);
                events.push(StreamEvent::ReasoningDelta {
                    kind: ReasoningKind::Summary,
                    text: text.to_string(),
                });
            }
        }
        for field in self
            .aliases
            .encrypted
            .iter()
            .chain(self.aliases.signature.iter())
        {
            if field == "reasoning_details" {
                continue;
            }
            if let Some(value) = delta.get(field) {
                if let Some(text) = value.as_str() {
                    let existing = self
                        .reasoning_opaque
                        .entry(field.clone())
                        .or_insert_with(|| Value::String(String::new()));
                    if let Value::String(existing) = existing {
                        existing.push_str(text);
                    } else {
                        *existing = Value::String(text.to_string());
                    }
                } else if !value.is_null() {
                    self.reasoning_opaque.insert(field.clone(), value.clone());
                }
            }
        }
        if let Some(details) = delta.get("reasoning_details").and_then(Value::as_array) {
            for detail in details {
                self.reasoning_details.push(detail.clone());
                if let Some(text) = detail
                    .get("text")
                    .and_then(Value::as_str)
                    .or_else(|| detail.get("summary").and_then(Value::as_str))
                    && !text.is_empty()
                {
                    events.push(StreamEvent::ReasoningDelta {
                        kind: openai_chat_reasoning_kind(detail),
                        text: text.to_string(),
                    });
                }
            }
        }
        if let Some(text) = delta.get("content").and_then(Value::as_str) {
            self.text.push_str(text);
            events.push(StreamEvent::TextDelta {
                text: text.to_string(),
            });
        }

        if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for call in tool_calls {
                let index = call.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                let accumulator = self.tool_calls.entry(index).or_default();
                let id = call.get("id").and_then(Value::as_str).map(str::to_string);
                let name = call
                    .pointer("/function/name")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                let arguments_delta = call
                    .pointer("/function/arguments")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();

                if id.is_some() {
                    accumulator.id = id.clone();
                }
                if let Some(name) = &name
                    && accumulator.name.is_empty()
                {
                    accumulator.name.clone_from(name);
                }
                accumulator.arguments.push_str(&arguments_delta);

                let progress_id = accumulator.id.clone().or_else(|| id.clone());
                let progress_name = (!accumulator.name.is_empty())
                    .then(|| accumulator.name.clone())
                    .or_else(|| name.clone());
                let arguments_so_far = accumulator.arguments.clone();
                events.push(StreamEvent::ToolCallDelta {
                    index,
                    id,
                    name,
                    arguments_delta,
                });
                events.push(StreamEvent::ToolCallProgress {
                    index,
                    id: progress_id,
                    name: progress_name,
                    arguments_so_far,
                });
            }
        }

        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>> {
        Ok(self.done_events())
    }
}

pub(crate) fn from_openai_chat_response(value: Value) -> Result<ChatResponse> {
    from_openai_chat_response_with_aliases(value, &ReasoningAliases::default())
}

pub(crate) fn from_openai_chat_response_with_aliases(
    value: Value,
    aliases: &ReasoningAliases,
) -> Result<ChatResponse> {
    let message = value
        .pointer("/choices/0/message")
        .cloned()
        .unwrap_or(Value::Null);

    let mut parts = Vec::new();
    let mut handled = Vec::new();

    for field in &aliases.response_text {
        if handled.iter().any(|seen| seen == field) {
            continue;
        }
        handled.push(field.clone());
        if field == "reasoning_details" {
            push_reasoning_details(&mut parts, message.get(field));
        } else if let Some(text) = message.get(field).and_then(Value::as_str)
            && !text.is_empty()
        {
            parts.push(Part::reasoning(openai_chat_reasoning(
                ReasoningKind::Text,
                Some(text),
                field,
                Value::String(text.to_string()),
            )));
        }
    }

    for field in &aliases.response_summary {
        if handled.iter().any(|seen| seen == field) {
            continue;
        }
        handled.push(field.clone());
        if field == "reasoning_details" {
            push_reasoning_details(&mut parts, message.get(field));
        } else if let Some(text) = message.get(field).and_then(Value::as_str)
            && !text.is_empty()
        {
            parts.push(Part::reasoning(openai_chat_reasoning(
                ReasoningKind::Summary,
                Some(text),
                field,
                Value::String(text.to_string()),
            )));
        }
    }

    for field in aliases.encrypted.iter().chain(aliases.signature.iter()) {
        if handled.iter().any(|seen| seen == field) {
            continue;
        }
        handled.push(field.clone());
        let Some(value) = message.get(field) else {
            continue;
        };
        if field == "reasoning_details" {
            push_reasoning_details(&mut parts, Some(value));
        } else if let Some(text) = value.as_str() {
            if !text.is_empty() {
                parts.push(Part::reasoning(openai_chat_reasoning(
                    ReasoningKind::Encrypted,
                    None,
                    field,
                    Value::String(text.to_string()),
                )));
            }
        } else if !value.is_null() {
            parts.push(Part::reasoning(openai_chat_reasoning(
                ReasoningKind::Encrypted,
                None,
                field,
                value.clone(),
            )));
        }
    }

    if !handled.iter().any(|field| field == "reasoning_details")
        && let Some(details) = message.get("reasoning_details")
    {
        push_reasoning_details(&mut parts, Some(details));
    }

    if let Some(text) = message.get("content").and_then(Value::as_str)
        && !text.is_empty()
    {
        parts.push(Part::text(text));
    }

    if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
        for call in tool_calls {
            let id = call.get("id").and_then(Value::as_str).map(str::to_string);
            let name = call
                .pointer("/function/name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let arguments = call
                .pointer("/function/arguments")
                .and_then(Value::as_str)
                .map(parse_json_or_string)
                .unwrap_or(Value::Null);
            parts.push(Part::ToolCall(ToolCall {
                id,
                name,
                arguments,
                provider_state: None,
            }));
        }
    }

    let usage = value.get("usage").map(openai_chat_usage_value);

    Ok(ChatResponse {
        message: Message::new(Role::Assistant, parts),
        usage,
        raw: value,
        metadata: None,
        report: CompletionReport::default(),
    })
}

fn push_reasoning_details(parts: &mut Vec<Part>, value: Option<&Value>) {
    let Some(Value::Array(details)) = value else {
        return;
    };
    for detail in details {
        parts.push(Part::reasoning(openai_chat_reasoning_from_detail(detail)));
    }
}

fn openai_chat_reasoning(
    kind: ReasoningKind,
    text: Option<&str>,
    field: &str,
    value: Value,
) -> Reasoning {
    let (summary, text) = match kind {
        ReasoningKind::Summary => (text.map(str::to_string), None),
        _ => (None, text.map(str::to_string)),
    };
    Reasoning {
        kind,
        summary,
        text,
        state: Some(ProviderState::new(
            ProviderStateFormat::OpenAiChat,
            json!({"field": field, "value": value}),
        )),
    }
}

fn openai_chat_reasoning_kind(detail: &Value) -> ReasoningKind {
    match detail.get("type").and_then(Value::as_str) {
        Some(kind) if kind.contains("summary") => ReasoningKind::Summary,
        Some(kind) if kind.contains("encrypted") => ReasoningKind::Encrypted,
        Some(kind) if kind.contains("redacted") => ReasoningKind::Redacted,
        _ => ReasoningKind::Text,
    }
}

fn openai_chat_reasoning_from_detail(detail: &Value) -> Reasoning {
    let kind = openai_chat_reasoning_kind(detail);
    let text = detail
        .get("text")
        .and_then(Value::as_str)
        .or_else(|| detail.get("summary").and_then(Value::as_str));
    openai_chat_reasoning(kind, text, "reasoning_details", detail.clone())
}

fn openai_chat_usage(usage: &Value) -> Option<Usage> {
    Some(openai_chat_usage_value(usage))
}

fn openai_chat_usage_value(usage: &Value) -> Usage {
    Usage {
        input_tokens: usage.get("prompt_tokens").and_then(Value::as_u64),
        output_tokens: usage.get("completion_tokens").and_then(Value::as_u64),
        total_tokens: usage.get("total_tokens").and_then(Value::as_u64),
        cache_read_tokens: usage
            .pointer("/prompt_tokens_details/cached_tokens")
            .and_then(Value::as_u64)
            .or_else(|| usage.get("cache_read_input_tokens").and_then(Value::as_u64)),
        cache_write_tokens: usage
            .get("cache_creation_input_tokens")
            .and_then(Value::as_u64),
        reasoning_tokens: usage
            .pointer("/completion_tokens_details/reasoning_tokens")
            .and_then(Value::as_u64),
        raw: Some(usage.clone()),
    }
}

fn parse_json_or_string(value: &str) -> Value {
    serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_string()))
}

fn openai_chat_reasoning_effort(config: Option<&ReasoningConfig>) -> Option<&'static str> {
    let config = config?;
    if let Some(effort) = config.effort {
        return Some(match effort {
            crate::types::ReasoningEffort::None => "none",
            crate::types::ReasoningEffort::Minimal => "minimal",
            crate::types::ReasoningEffort::Low => "low",
            crate::types::ReasoningEffort::Medium => "medium",
            crate::types::ReasoningEffort::High => "high",
            crate::types::ReasoningEffort::XHigh => "xhigh",
            crate::types::ReasoningEffort::Max => "max",
        });
    }

    match config.mode {
        Some(ReasoningMode::Disabled) => Some("none"),
        _ => None,
    }
}

fn text_content(message: &NormalizedMessage) -> String {
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

fn image_source_url(source: &crate::types::ImageSource) -> String {
    match source {
        crate::types::ImageSource::Url { url } => url.clone(),
        crate::types::ImageSource::Base64 { media_type, data } => {
            format!("data:{media_type};base64,{data}")
        }
        crate::types::ImageSource::FileRef { uri, .. } => uri.clone(),
    }
}

fn content_value(parts: &[Part]) -> Value {
    let has_image = parts
        .iter()
        .any(|part| matches!(part, Part::ImageUrl { .. } | Part::Image { .. }));
    if !has_image {
        return Value::String(
            parts
                .iter()
                .filter_map(|part| match part {
                    Part::Text { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }

    Value::Array(
        parts
            .iter()
            .filter_map(|part| match part {
                Part::Text { text, .. } => Some(json!({"type": "text", "text": text})),
                Part::ImageUrl { image_url } => Some(json!({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url.url,
                        "detail": image_url.detail,
                    }
                })),
                Part::Image { image } => Some(json!({
                    "type": "image_url",
                    "image_url": {
                        "url": image_source_url(&image.source),
                        "detail": image.detail,
                    }
                })),
                _ => None,
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize::{Protocol, normalize};

    #[test]
    fn reasoning_content_round_trips_cjk() {
        let response = from_openai_chat_response(json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "reasoning_content": "我先检查边界，再回答：你好，世界。",
                    "content": "最终答案"
                }
            }]
        }))
        .unwrap();

        let reasoning = response.reasoning().next().unwrap();
        assert_eq!(reasoning.kind, ReasoningKind::Text);
        assert_eq!(
            reasoning.text.as_deref(),
            Some("我先检查边界，再回答：你好，世界。")
        );
        assert_eq!(response.text(), "最终答案");

        let request = ChatRequest::new([response.message]);
        let (normalized, _) = normalize(&request, Protocol::OpenAiChat).unwrap();
        let body = to_openai_chat_body("deepseek-reasoner", &normalized, &request, None).unwrap();
        assert_eq!(
            body.pointer("/messages/0/reasoning_content")
                .and_then(Value::as_str),
            Some("我先检查边界，再回答：你好，世界。")
        );
    }

    #[test]
    fn reasoning_details_round_trip_in_order() {
        let response = from_openai_chat_response(json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "reasoning_details": [
                        {"type": "reasoning.summary", "text": "摘要"},
                        {"type": "reasoning.encrypted", "data": "opaque"}
                    ],
                    "content": "answer"
                }
            }]
        }))
        .unwrap();

        let reasoning: Vec<_> = response.reasoning().collect();
        assert_eq!(reasoning[0].kind, ReasoningKind::Summary);
        assert_eq!(reasoning[0].summary.as_deref(), Some("摘要"));
        assert_eq!(reasoning[1].kind, ReasoningKind::Encrypted);

        let request = ChatRequest::new([response.message]);
        let (normalized, _) = normalize(&request, Protocol::OpenAiChat).unwrap();
        let body = to_openai_chat_body("compatible", &normalized, &request, None).unwrap();
        assert_eq!(
            body.pointer("/messages/0/reasoning_details"),
            Some(&json!([
                {"type": "reasoning.summary", "text": "摘要"},
                {"type": "reasoning.encrypted", "data": "opaque"}
            ]))
        );
    }

    #[test]
    fn neutral_request_controls_map_to_openai_chat() {
        let request = ChatRequest::user("hi")
            .tools([crate::types::ToolSpec::new(
                "lookup",
                "lookup",
                json!({"type": "object"}),
            )])
            .tool_choice(ToolChoice::Required)
            .response_format(ResponseFormat::JsonSchema {
                name: "answer".to_string(),
                schema: json!({"type": "object"}),
                strict: Some(true),
            })
            .stop(["END"])
            .seed(42);
        let (normalized, _) = normalize(&request, Protocol::OpenAiChat).unwrap();
        let body = to_openai_chat_body("gpt-5", &normalized, &request, None).unwrap();

        assert_eq!(body["tool_choice"], "required");
        assert_eq!(body["response_format"]["type"], "json_schema");
        assert_eq!(body["response_format"]["json_schema"]["strict"], true);
        assert_eq!(body["stop"], json!(["END"]));
        assert_eq!(body["seed"], 42);
    }

    #[test]
    fn unknown_tool_choice_is_rejected() {
        let request = ChatRequest::user("hi")
            .tools([crate::types::ToolSpec::new(
                "lookup",
                "lookup",
                json!({"type": "object"}),
            )])
            .tool_choice(ToolChoice::Tool("missing".to_string()));
        let (normalized, _) = normalize(&request, Protocol::OpenAiChat).unwrap();
        let error = to_openai_chat_body("gpt-5", &normalized, &request, None).unwrap_err();

        assert!(matches!(error, Error::InvalidRequest(_)));
    }

    #[test]
    fn reasoning_effort_is_sent() {
        let request = ChatRequest::user("hi")
            .reasoning(ReasoningConfig::new().effort(crate::types::ReasoningEffort::XHigh));
        let (normalized, _) = normalize(&request, Protocol::OpenAiChat).unwrap();
        let body = to_openai_chat_body("gpt-5", &normalized, &request, None).unwrap();
        assert_eq!(body["reasoning_effort"], "xhigh");
    }

    #[test]
    fn extra_body_is_merged_but_cannot_override_canonical_fields() {
        let request = ChatRequest::user("hi")
            .extra_body("enable_thinking", true)
            .extra_body("reasoning_format", "parsed");
        let (normalized, _) = normalize(&request, Protocol::OpenAiChat).unwrap();
        let body = to_openai_chat_body("qwen3", &normalized, &request, None).unwrap();

        assert_eq!(body["enable_thinking"], true);
        assert_eq!(body["reasoning_format"], "parsed");

        let request = ChatRequest::user("hi").extra_body("model", "attacker-model");
        let (normalized, _) = normalize(&request, Protocol::OpenAiChat).unwrap();
        let error = to_openai_chat_body("qwen3", &normalized, &request, None).unwrap_err();
        assert!(matches!(
            error,
            Error::ReservedExtraBodyField { field, .. } if field == "model"
        ));
    }

    #[test]
    fn profile_extra_body_is_a_default_and_request_extra_wins() {
        let mut profile = ProviderProfile::new("qwen");
        profile
            .request
            .extra_body
            .insert("enable_thinking".to_string(), json!(false));
        profile
            .request
            .extra_body
            .insert("reasoning_format".to_string(), json!("hidden"));

        let request = ChatRequest::user("hi").extra_body("enable_thinking", true);
        let (normalized, _) = normalize(&request, Protocol::OpenAiChat).unwrap();
        let body = to_openai_chat_body("qwen3", &normalized, &request, Some(&profile)).unwrap();

        assert_eq!(body["enable_thinking"], true);
        assert_eq!(body["reasoning_format"], "hidden");
    }

    #[test]
    fn profile_aliases_control_response_parsing_and_history_replay() {
        let aliases = ReasoningAliases::empty()
            .text("thinking_text")
            .replay("thinking_text")
            .encrypted("encrypted_thinking");
        let response = from_openai_chat_response_with_aliases(
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "thinking_text": "先检查输入。",
                        "encrypted_thinking": "opaque-state",
                        "content": "答案"
                    }
                }]
            }),
            &aliases,
        )
        .unwrap();

        assert_eq!(response.reasoning_text(), "先检查输入。");
        assert_eq!(response.text(), "答案");
        assert_eq!(response.reasoning().count(), 2);

        let mut profile = ProviderProfile::new("custom-compatible");
        profile.reasoning.aliases = aliases;
        let request = ChatRequest::new([response.message]);
        let (normalized, _) = normalize(&request, Protocol::OpenAiChat).unwrap();
        let body =
            to_openai_chat_body("custom-model", &normalized, &request, Some(&profile)).unwrap();

        assert_eq!(body["messages"][0]["thinking_text"], "先检查输入。");
        assert_eq!(body["messages"][0]["encrypted_thinking"], "opaque-state");
    }

    #[test]
    fn profile_thinking_request_maps_neutral_config() {
        let mut profile = ProviderProfile::new("qwen");
        profile.reasoning.thinking = ThinkingRequestProfile::Composite(vec![
            ThinkingRequestProfile::EnabledFlag("enable_thinking".to_string()),
            ThinkingRequestProfile::BudgetTokens("thinking_budget".to_string()),
        ]);

        let request = ChatRequest::user("hi").reasoning(
            ReasoningConfig::new()
                .mode(ReasoningMode::Enabled)
                .budget_tokens(2048),
        );
        let (normalized, _) = normalize(&request, Protocol::OpenAiChat).unwrap();
        let body = to_openai_chat_body("qwen3", &normalized, &request, Some(&profile)).unwrap();

        assert_eq!(body["enable_thinking"], true);
        assert_eq!(body["thinking_budget"], 2048);

        let request =
            ChatRequest::user("hi").reasoning(ReasoningConfig::new().mode(ReasoningMode::Disabled));
        let (normalized, _) = normalize(&request, Protocol::OpenAiChat).unwrap();
        let body = to_openai_chat_body("qwen3", &normalized, &request, Some(&profile)).unwrap();

        assert_eq!(body["enable_thinking"], false);
        assert!(body.get("thinking_budget").is_none());
    }

    #[test]
    fn replay_policy_never_drops_reasoning_history() {
        let response = from_openai_chat_response(json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "reasoning_content": "private",
                    "content": "answer"
                }
            }]
        }))
        .unwrap();

        let mut profile = ProviderProfile::new("private-history");
        profile.reasoning.replay = ReasoningReplayPolicy::Never;
        let request = ChatRequest::new([response.message]);
        let (normalized, _) = normalize(&request, Protocol::OpenAiChat).unwrap();
        let body =
            to_openai_chat_body("private-model", &normalized, &request, Some(&profile)).unwrap();

        assert!(body["messages"][0].get("reasoning_content").is_none());
    }

    #[test]
    fn model_profile_overrides_provider_thinking_and_aliases() {
        let mut provider = ProviderProfile::new("moonshot");
        provider.reasoning.thinking =
            ThinkingRequestProfile::EnabledFlag("enable_thinking".to_string());
        provider.reasoning.aliases = ReasoningAliases::default();

        let model_profile = ModelProfile {
            matcher: crate::profile::ModelMatcher::Prefix("kimi-k3".to_string()),
            thinking: ThinkingRequestProfile::Effort("reasoning_effort".to_string()),
            reasoning_aliases: ReasoningAliases::empty()
                .text("reasoning_content")
                .replay("reasoning_content"),
            replay: Some(ReasoningReplayPolicy::SameProvider),
            ..ModelProfile::default()
        };

        let request = ChatRequest::user("hi")
            .reasoning(ReasoningConfig::new().effort(crate::types::ReasoningEffort::High));
        let (normalized, _) = normalize(&request, Protocol::OpenAiChat).unwrap();
        let body = to_openai_chat_body_with_profile(
            "kimi-k3",
            &normalized,
            &request,
            Some(&provider),
            Some(&model_profile),
        )
        .unwrap();

        assert_eq!(body["reasoning_effort"], "high");
        assert!(body.get("enable_thinking").is_none());
    }

    #[test]
    fn domestic_profile_presets_emit_documented_request_fields() {
        let deepseek = ChatRequest::user("hi").reasoning(
            ReasoningConfig::new()
                .effort(crate::types::ReasoningEffort::Minimal)
                .budget_tokens(1024),
        );
        let (normalized, _) = normalize(&deepseek, Protocol::OpenAiChat).unwrap();
        let body = to_openai_chat_body(
            "deepseek-reasoner",
            &normalized,
            &deepseek,
            Some(&ProviderProfile::deepseek_compatible()),
        )
        .unwrap();
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["budget_tokens"], 1024);
        assert_eq!(body["reasoning_effort"], "low");

        let qwen = ChatRequest::user("hi").reasoning(
            ReasoningConfig::new()
                .mode(ReasoningMode::Enabled)
                .budget_tokens(2048),
        );
        let (normalized, _) = normalize(&qwen, Protocol::OpenAiChat).unwrap();
        let body = to_openai_chat_body(
            "qwen3",
            &normalized,
            &qwen,
            Some(&ProviderProfile::qwen_compatible()),
        )
        .unwrap();
        assert_eq!(body["enable_thinking"], true);
        assert_eq!(body["thinking_budget"], 2048);

        let glm =
            ChatRequest::user("hi").reasoning(ReasoningConfig::new().mode(ReasoningMode::Disabled));
        let (normalized, _) = normalize(&glm, Protocol::OpenAiChat).unwrap();
        let body = to_openai_chat_body(
            "glm-4.7",
            &normalized,
            &glm,
            Some(&ProviderProfile::glm_compatible()),
        )
        .unwrap();
        assert_eq!(body["thinking"]["type"], "disabled");

        let kimi = ChatRequest::user("hi")
            .reasoning(ReasoningConfig::new().effort(crate::types::ReasoningEffort::High));
        let (normalized, _) = normalize(&kimi, Protocol::OpenAiChat).unwrap();
        let body = to_openai_chat_body(
            "kimi-k3",
            &normalized,
            &kimi,
            Some(&ProviderProfile::kimi_compatible()),
        )
        .unwrap();
        assert_eq!(body["reasoning_effort"], "high");
    }

    #[test]
    fn doubao_preset_replays_encrypted_content_for_tool_turns() {
        let aliases = ReasoningAliases::empty()
            .text("reasoning_content")
            .replay("reasoning_content")
            .encrypted("encrypted_content");
        let response = from_openai_chat_response_with_aliases(
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "tool reasoning",
                        "encrypted_content": "opaque",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"}
                        }]
                    }
                }]
            }),
            &aliases,
        )
        .unwrap();

        let request = ChatRequest::new([response.message]);
        let (normalized, _) = normalize(&request, Protocol::OpenAiChat).unwrap();
        let body = to_openai_chat_body(
            "doubao-seed",
            &normalized,
            &request,
            Some(&ProviderProfile::doubao_compatible()),
        )
        .unwrap();

        assert_eq!(body["messages"][0]["reasoning_content"], "tool reasoning");
        assert_eq!(body["messages"][0]["encrypted_content"], "opaque");
    }

    #[test]
    fn kimi_model_presets_map_k3_and_k2_6_controls() {
        let k3 = ChatRequest::user("hi")
            .reasoning(ReasoningConfig::new().effort(crate::types::ReasoningEffort::High));
        let (normalized, _) = normalize(&k3, Protocol::OpenAiChat).unwrap();
        let body = to_openai_chat_body_with_profile(
            "kimi-k3",
            &normalized,
            &k3,
            Some(&ProviderProfile::kimi_compatible()),
            Some(&ModelProfile::kimi_k3()),
        )
        .unwrap();
        assert_eq!(body["reasoning_effort"], "high");

        let k2_6 =
            ChatRequest::user("hi").reasoning(ReasoningConfig::new().mode(ReasoningMode::Enabled));
        let (normalized, _) = normalize(&k2_6, Protocol::OpenAiChat).unwrap();
        let body = to_openai_chat_body_with_profile(
            "kimi-k2.6",
            &normalized,
            &k2_6,
            Some(&ProviderProfile::kimi_compatible()),
            Some(&ModelProfile::kimi_k2_6()),
        )
        .unwrap();
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["keep"], "all");
    }

    #[test]
    fn static_thinking_fields_apply_without_reasoning_config() {
        let mut profile = ProviderProfile::new("custom");
        profile.reasoning.thinking = ThinkingRequestProfile::Field {
            path: "clear_thinking".to_string(),
            value: Value::Bool(false),
        };
        let request = ChatRequest::user("hi");
        let (normalized, _) = normalize(&request, Protocol::OpenAiChat).unwrap();
        let body =
            to_openai_chat_body("custom-model", &normalized, &request, Some(&profile)).unwrap();

        assert_eq!(body["clear_thinking"], false);
    }

    #[test]
    fn model_profile_can_replace_budget_with_effort_controls() {
        let model_profile = ModelProfile {
            matcher: crate::profile::ModelMatcher::Prefix("qwen-effort".to_string()),
            thinking: ThinkingRequestProfile::MappedEffort {
                field: "reasoning_effort".to_string(),
                mapping: crate::profile::EffortMapping::identity(),
            },
            reasoning_aliases: ReasoningAliases::empty()
                .text("reasoning_content")
                .replay("reasoning_content"),
            replay: Some(ReasoningReplayPolicy::SameProvider),
            ..ModelProfile::default()
        };
        let request = ChatRequest::user("hi").reasoning(
            ReasoningConfig::new()
                .effort(crate::types::ReasoningEffort::High)
                .budget_tokens(2048),
        );
        let (normalized, _) = normalize(&request, Protocol::OpenAiChat).unwrap();
        let body = to_openai_chat_body_with_profile(
            "qwen-effort-235b",
            &normalized,
            &request,
            Some(&ProviderProfile::qwen_compatible()),
            Some(&model_profile),
        )
        .unwrap();

        assert_eq!(body["reasoning_effort"], "high");
        assert!(body.get("enable_thinking").is_none());
        assert!(body.get("thinking_budget").is_none());
    }

    #[test]
    fn doubao_preset_drops_reasoning_when_turn_has_no_tool_calls() {
        let response = from_openai_chat_response_with_aliases(
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "private",
                        "encrypted_content": "opaque",
                        "content": "answer"
                    }
                }]
            }),
            &ReasoningAliases::empty()
                .text("reasoning_content")
                .replay("reasoning_content")
                .encrypted("encrypted_content"),
        )
        .unwrap();

        let request = ChatRequest::new([response.message]);
        let (normalized, _) = normalize(&request, Protocol::OpenAiChat).unwrap();
        let body = to_openai_chat_body(
            "doubao-seed",
            &normalized,
            &request,
            Some(&ProviderProfile::doubao_compatible()),
        )
        .unwrap();

        assert!(body["messages"][0].get("reasoning_content").is_none());
        assert!(body["messages"][0].get("encrypted_content").is_none());
    }

    #[test]
    fn model_profile_without_replay_inherits_provider_policy() {
        let aliases = ReasoningAliases::empty()
            .text("reasoning_content")
            .replay("reasoning_content")
            .encrypted("encrypted_content");
        let response = from_openai_chat_response_with_aliases(
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "encrypted_content": "opaque",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"}
                        }]
                    }
                }]
            }),
            &aliases,
        )
        .unwrap();

        let request = ChatRequest::new([response.message]);
        let (normalized, _) = normalize(&request, Protocol::OpenAiChat).unwrap();
        let body = to_openai_chat_body_with_profile(
            "doubao-seed",
            &normalized,
            &request,
            Some(&ProviderProfile::doubao_compatible()),
            Some(&ModelProfile::default()),
        )
        .unwrap();

        assert_eq!(body["messages"][0]["encrypted_content"], "opaque");
    }

    #[test]
    fn stream_mapper_uses_profile_aliases() {
        let aliases = ReasoningAliases::empty()
            .text("thinking_text")
            .replay("thinking_text");
        let metadata = ResponseMetadata {
            status: 200,
            request_id: None,
            headers: HeaderMap::new(),
        };
        let mut mapper =
            OpenAiChatStreamMapper::with_aliases("custom-model".to_string(), metadata, aliases);

        let events = mapper
            .map(SseMessage {
                event: None,
                data: json!({
                    "choices": [{
                        "delta": {"thinking_text": "先分析", "content": "答案"},
                        "finish_reason": null
                    }]
                })
                .to_string(),
                id: None,
            })
            .unwrap();
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::ReasoningDelta { text, .. } if text == "先分析"
        )));

        let events = mapper
            .map(SseMessage {
                event: None,
                data: "[DONE]".to_string(),
                id: None,
            })
            .unwrap();
        let response = events
            .into_iter()
            .find_map(|event| match event {
                StreamEvent::Done { response, .. } => Some(response),
                _ => None,
            })
            .unwrap();
        assert_eq!(response.reasoning_text(), "先分析");
        assert_eq!(response.text(), "答案");
    }

    #[test]
    fn stream_assembles_reasoning_text_and_tool_arguments() {
        let metadata = ResponseMetadata {
            status: 200,
            request_id: Some("req-stream".to_string()),
            headers: HeaderMap::new(),
        };
        let mut mapper = OpenAiChatStreamMapper::new("deepseek-reasoner".to_string(), metadata);
        let chunks = [
            json!({
                "choices": [{
                    "delta": {
                        "role": "assistant",
                        "reasoning_content": "先检查边界，",
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "lookup", "arguments": "{\"q\":"}
                        }]
                    },
                    "finish_reason": null
                }]
            }),
            json!({
                "choices": [{
                    "delta": {
                        "reasoning_content": "再调用工具。",
                        "content": "结果如下",
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": "\"Rust\"}"}
                        }]
                    },
                    "finish_reason": "tool_calls"
                }]
            }),
            json!({
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}
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
        events.extend(
            mapper
                .map(SseMessage {
                    event: None,
                    data: "[DONE]".to_string(),
                    id: None,
                })
                .unwrap(),
        );

        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::ReasoningDelta { text, .. } if text == "先检查边界，"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::TextDelta { text } if text == "结果如下"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::ToolCallDelta {
                index: 0,
                arguments_delta,
                ..
            } if arguments_delta == "\"Rust\"}"
        )));

        let StreamEvent::Done {
            response,
            finish_reason,
        } = events.last().unwrap()
        else {
            panic!("expected done event");
        };
        assert_eq!(finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(response.text(), "结果如下");
        assert_eq!(response.reasoning_text(), "先检查边界，再调用工具。");
        let call = response.tool_calls().next().unwrap();
        assert_eq!(call.name, "lookup");
        assert_eq!(call.arguments, json!({"q": "Rust"}));
        assert_eq!(response.usage.as_ref().unwrap().input_tokens, Some(10));
    }
}
