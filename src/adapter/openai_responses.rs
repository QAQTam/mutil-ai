use crate::security::{RequestBudget, ResponseLimits};
use std::collections::BTreeMap;

use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use serde_json::{Value, json};

use super::{
    ModelAdapter, ReconnectRequest, ReconnectRequestParts, apply_provider_request_options,
    apply_request_transform, audit_outcome_for_error, completion_report, effective_retry_policy,
    finish_sse_stream, json_body_bytes, merge_extra_body, prepare_headers, protocol_name,
    resolve_api_key, send_json_retry, send_stream_retry, validate_tool_choice,
};
use crate::error::{Error, Result};
use crate::headers::{RequestOptions, TransportConfig};
use crate::native_tool_stream::{CustomToolAccumulator, NativeToolIdentity};
use crate::normalize::{ExternalRole, NormalizedChat, Protocol, normalize};
use crate::report::CompletionReport;
use crate::retry::{RetryPolicy, RetryProvider};
use crate::sse::SseMessage;
use crate::stream::{ModelStream, SseMapper, StreamEvent};
use crate::types::{
    ChatRequest, ChatResponse, CustomToolFormat, Message, NativeToolCall, NativeToolCompletion,
    NativeToolKind, NativeToolPayload, Part, ProviderState, ProviderStateFormat, Reasoning,
    ReasoningConfig, ReasoningKind, ReasoningMode, ResponseFormat, ResponseMetadata, Role,
    ScopedToolDeclaration, ToolCall, ToolChoice, ToolDeclaration, Usage,
};

/// Adapter for `POST /v1/responses`.
///
/// [`OpenAIResponses`] speaks OpenAI's Responses protocol: neutral
/// [`ChatRequest`]s are mapped onto the `/v1/responses` body, and outputs
/// (messages, reasoning items, function calls, and server tool items) are
/// mapped back to a neutral [`ChatResponse`]. Reasoning items round-trip
/// through their original provider state, so encrypted reasoning can be
/// replayed in follow-up turns.
///
/// Streaming consumes the Responses event stream (`response.output_text.delta`,
/// `response.completed`, and related events). Requests are authorized with
/// `Authorization: Bearer <key>`, where the key comes from
/// [`OpenAIResponses::api_key`] or the `OPENAI_API_KEY` environment variable.
pub struct OpenAIResponses {
    model: String,
    api_key: Option<String>,
    base_url: String,
    retry_policy: RetryPolicy,
    transport: TransportConfig,
    client: reqwest::Client,
}

impl OpenAIResponses {
    /// Creates a Responses API adapter for `model`, defaulting to OpenAI's
    /// public endpoint.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            api_key: None,
            base_url: "https://api.openai.com/v1".to_string(),
            retry_policy: RetryPolicy::default(),
            transport: TransportConfig::default(),
            client: super::secure_http_client(),
        }
    }

    /// Sets the API key sent as `Authorization: Bearer <key>`.
    ///
    /// When unset, the `OPENAI_API_KEY` environment variable is used.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Overrides the API base URL (default `https://api.openai.com/v1`).
    ///
    /// The `/responses` path is appended, so this can point at any
    /// Responses-compatible endpoint.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Overrides the retry policy applied to transient transport failures.
    pub fn retry_policy(mut self, retry_policy: RetryPolicy) -> Self {
        self.retry_policy = retry_policy;
        self
    }

    /// Sets the transport configuration used when preparing outgoing requests.
    pub fn transport(mut self, transport: TransportConfig) -> Self {
        self.transport = transport;
        self
    }
}

#[async_trait]
impl ModelAdapter for OpenAIResponses {
    fn provider_name(&self) -> &'static str {
        "openai-responses"
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    async fn complete_with(
        &self,
        request: &ChatRequest,
        options: &RequestOptions,
    ) -> Result<ChatResponse> {
        let budget = RequestBudget::new(options)?;
        let transformed = apply_request_transform(request, options)?;
        let transform_applied = matches!(transformed, std::borrow::Cow::Owned(_));
        let request = transformed.as_ref();
        let (normalized, report) = normalize(request, Protocol::OpenAiResponses)?;
        validate_responses_request(request)?;
        let mut body = to_responses_body(&self.model, &normalized, request);
        merge_extra_body(&mut body, request, Protocol::OpenAiResponses, None)?;
        apply_provider_request_options(
            &mut body,
            Protocol::OpenAiResponses,
            &options.provider_request,
            false,
        )?;
        let request_body_bytes = json_body_bytes(&body)?;
        let key = resolve_api_key(
            self.provider_name(),
            self.api_key.as_deref(),
            "OPENAI_API_KEY",
        )?;
        options.validate_query(protocol_name(Protocol::OpenAiResponses), &[])?;
        let retry_policy = effective_retry_policy(&self.retry_policy, options);
        let audit = self.transport.audit_context(
            self.provider_name(),
            protocol_name(Protocol::OpenAiResponses),
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
            &budget,
            || async {
                let headers = prepare_headers(&self.transport, &base_headers, options).await?;
                let mut request_builder = self
                    .client
                    .post(format!("{}/responses", self.base_url.trim_end_matches('/')))
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

        let mut response = match from_responses_response(value).and_then(|response| {
            budget.check_deadline()?;
            Ok(response)
        }) {
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
        if !budget.limits.retain_raw {
            response.raw = serde_json::Value::Null;
        }
        response.report = completion_report(
            Protocol::OpenAiResponses,
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
        let budget = RequestBudget::new(options)?;
        let transformed = apply_request_transform(request, options)?;
        let transform_applied = matches!(transformed, std::borrow::Cow::Owned(_));
        let request = transformed.as_ref();
        let (normalized, report) = normalize(request, Protocol::OpenAiResponses)?;
        validate_responses_request(request)?;
        let mut body = to_responses_body(&self.model, &normalized, request);
        merge_extra_body(&mut body, request, Protocol::OpenAiResponses, None)?;
        body["stream"] = json!(true);
        apply_provider_request_options(
            &mut body,
            Protocol::OpenAiResponses,
            &options.provider_request,
            true,
        )?;
        let request_body_bytes = json_body_bytes(&body)?;

        let key = resolve_api_key(
            self.provider_name(),
            self.api_key.as_deref(),
            "OPENAI_API_KEY",
        )?;
        options.validate_query(protocol_name(Protocol::OpenAiResponses), &[])?;
        let retry_policy = effective_retry_policy(&self.retry_policy, options);
        let audit = self.transport.audit_context(
            self.provider_name(),
            protocol_name(Protocol::OpenAiResponses),
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
        let url = format!("{}/responses", self.base_url.trim_end_matches('/'));

        let response = match send_stream_retry(
            self.provider_name(),
            RetryProvider::OpenAi,
            &retry_policy,
            audit.as_ref(),
            options.cancellation.as_ref(),
            &budget,
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
        let mapper = ResponsesStreamMapper::new(self.model.clone(), metadata).with_report(
            completion_report(
                Protocol::OpenAiResponses,
                request,
                report.stats(),
                transform_applied,
            ),
        );
        let reconnect = options.stream_reconnect.map(|_| {
            ReconnectRequest::new(ReconnectRequestParts {
                budget: budget.clone(),
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
            budget,
        ))
    }
}

/// Builds the OpenAI Responses request body (`input`, `instructions`,
/// `tools`, and sampling controls) from a normalized request.
#[cfg_attr(not(test), allow(dead_code))]
fn responses_declaration(declaration: &ToolDeclaration) -> Value {
    fn scoped(tool: &ScopedToolDeclaration) -> Value {
        match tool {
            ScopedToolDeclaration::Function { tool } => json!({
                "type": "function", "name": tool.name, "description": tool.description,
                "parameters": tool.parameters, "strict": tool.strict,
            }),
            ScopedToolDeclaration::Custom {
                name,
                description,
                format,
            } => {
                let mut result =
                    json!({"type": "custom", "name": name, "description": description});
                if let CustomToolFormat::Grammar { syntax, definition } = format {
                    result["format"] =
                        json!({"type": "grammar", "syntax": syntax, "definition": definition});
                }
                result
            }
        }
    }
    match declaration {
        ToolDeclaration::Tool { tool } => scoped(tool),
        ToolDeclaration::Namespace {
            name,
            description,
            tools,
        } => json!({
            "type": "namespace", "name": name, "description": description,
            "tools": tools.iter().map(scoped).collect::<Vec<_>>()
        }),
    }
}

pub(crate) fn to_responses_body(
    model: &str,
    normalized: &NormalizedChat,
    request: &ChatRequest,
) -> Value {
    let mut input = Vec::new();
    let mut native_kinds = std::collections::HashMap::new();

    for message in &normalized.messages {
        match message.role {
            ExternalRole::System => {
                input.push(json!({"role": "system", "content": text_content(message)}));
            }
            ExternalRole::Developer => {
                input.push(json!({"role": "developer", "content": text_content(message)}));
            }
            ExternalRole::User => {
                input.push(json!({
                    "role": "user",
                    "content": responses_content_value(&message.parts),
                }));
            }
            ExternalRole::Assistant => {
                for part in &message.parts {
                    match part {
                        Part::Reasoning(reasoning) => {
                            if let Some(state) = &reasoning.state
                                && state.format == ProviderStateFormat::OpenAiResponses
                            {
                                input.push(state.data.clone());
                            }
                        }
                        Part::Text { text, .. } => {
                            input.push(json!({
                                "role": "assistant",
                                "content": text,
                            }));
                        }
                        Part::ToolCall(call) => {
                            input.push(json!({
                                "type": "function_call",
                                "call_id": call.id.clone().unwrap_or_else(|| call.name.clone()),
                                "name": call.name,
                                "arguments": serde_json::to_string(&call.arguments)
                                    .unwrap_or_else(|_| "{}".to_string()),
                            }));
                        }
                        Part::NativeToolCall(call) => {
                            native_kinds.insert(call.call_id.clone(), call.kind);
                            let mut item = call
                                .provider_state
                                .as_ref()
                                .map(|state| state.data.clone())
                                .filter(Value::is_object)
                                .unwrap_or_else(|| json!({}));
                            item["type"] = match call.kind {
                                NativeToolKind::Function => "function_call",
                                NativeToolKind::Custom => "custom_tool_call",
                            }
                            .into();
                            item["name"] = json!(call.name);
                            item["call_id"] = json!(call.call_id);
                            if let Some(object) = item.as_object_mut() {
                                object.remove("input");
                                object.remove("arguments");
                                object.remove("namespace");
                                object.remove("id");
                                object.remove("qaqh_native_call");
                            }
                            if call.kind == NativeToolKind::Custom {
                                item["status"] = json!("completed");
                            }
                            if let Some(namespace) = &call.namespace {
                                item["namespace"] = json!(namespace);
                            }
                            if let Some(item_id) = &call.item_id {
                                item["id"] = json!(item_id);
                            }
                            match &call.payload {
                                NativeToolPayload::RawText(input) => item["input"] = json!(input),
                                NativeToolPayload::JsonArguments(arguments) => {
                                    item["arguments"] = json!(arguments);
                                }
                            }
                            input.push(item);
                        }
                        Part::ProviderItem(item)
                            if item.provider_state.format
                                == ProviderStateFormat::OpenAiResponses =>
                        {
                            input.push(item.provider_state.data.clone());
                        }
                        _ => {}
                    }
                }
            }
            ExternalRole::Tool => {
                for result in message.parts.iter().filter_map(|part| match part {
                    Part::ToolResult(result) => Some(result),
                    _ => None,
                }) {
                    let kind = result.call_id.as_ref().and_then(|id| native_kinds.get(id));
                    input.push(json!({
                        "type": if kind == Some(&NativeToolKind::Custom) { "custom_tool_call_output" } else { "function_call_output" },
                        "call_id": result.call_id.clone().unwrap_or_else(|| result.name.clone()),
                        "output": result.content,
                    }));
                }
            }
        }
    }

    let mut body = json!({
        "model": model,
        "input": input,
    });

    if let Some(system) = &normalized.system {
        body["instructions"] = json!(system);
    }
    let mut tools = normalized
        .tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "strict": tool.strict,
            })
        })
        .collect::<Vec<_>>();
    tools.extend(request.tool_declarations.iter().map(responses_declaration));
    tools.extend(request.server_tools.iter().map(responses_server_tool));
    if !tools.is_empty() {
        body["tools"] = Value::Array(tools);
    }
    if let Some(temperature) = request.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(max_output_tokens) = request.max_output_tokens {
        body["max_output_tokens"] = json!(max_output_tokens);
    }
    if let Some(reasoning) = responses_reasoning(request.reasoning.as_ref()) {
        body["reasoning"] = reasoning;
    }
    if request
        .reasoning
        .as_ref()
        .and_then(|reasoning| reasoning.include_encrypted)
        .unwrap_or(false)
    {
        body["include"] = json!(["reasoning.encrypted_content"]);
    }
    if let Some(tool_choice) = &request.tool_choice {
        body["tool_choice"] = responses_tool_choice(tool_choice);
    }
    if let Some(response_format) = &request.response_format {
        body["text"] = json!({"format": responses_response_format(response_format)});
    }

    body
}

/// Rejects neutral request controls that the Responses protocol cannot
/// express.
///
/// # Errors
///
/// Returns [`Error::Unsupported`] when `stop` sequences or `seed` are set,
/// and propagates tool-choice validation errors.
pub(crate) fn validate_responses_request(request: &ChatRequest) -> Result<()> {
    validate_tool_choice(&request.tools, request.tool_choice.as_ref())?;
    if !request.stop.is_empty() {
        return Err(Error::Unsupported(
            "OpenAI Responses does not expose a neutral stop-sequence mapping".to_string(),
        ));
    }
    if request.seed.is_some() {
        return Err(Error::Unsupported(
            "OpenAI Responses does not expose a neutral seed mapping".to_string(),
        ));
    }
    Ok(())
}

fn responses_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Tool(name) => json!({
            "type": "function",
            "name": name,
        }),
    }
}

fn responses_response_format(format: &ResponseFormat) -> Value {
    match format {
        ResponseFormat::Text => json!({"type": "text"}),
        ResponseFormat::JsonObject => json!({"type": "json_object"}),
        ResponseFormat::JsonSchema {
            name,
            schema,
            strict,
        } => json!({
            "type": "json_schema",
            "name": name,
            "schema": schema,
            "strict": strict.unwrap_or(false),
        }),
    }
}

enum ResponsesPartAccumulator {
    Text(String),
    Reasoning {
        summary: String,
        text: String,
        item: Option<Value>,
    },
    ToolCall {
        call_id: Option<String>,
        name: String,
        arguments: String,
        item: Option<Value>,
    },
    ProviderItem(Value),
}

/// Stateful SSE mapper for Responses API events, accumulating output items
/// (text, reasoning, function calls, and server tool items) into a neutral
/// [`ChatResponse`].
pub(crate) struct ResponsesStreamMapper {
    model: String,
    metadata: ResponseMetadata,
    started: bool,
    done: bool,
    parts: BTreeMap<usize, ResponsesPartAccumulator>,
    custom_calls: BTreeMap<usize, CustomToolAccumulator>,
    usage: Option<Usage>,
    finish_reason: Option<String>,
    final_response: Option<ChatResponse>,
    chunks: Vec<Value>,
    limits: ResponseLimits,
    report: CompletionReport,
}

impl ResponsesStreamMapper {
    /// Creates a mapper for the given model and response metadata.
    pub(crate) fn new(model: String, metadata: ResponseMetadata) -> Self {
        Self {
            model,
            metadata,
            started: false,
            done: false,
            parts: BTreeMap::new(),
            custom_calls: BTreeMap::new(),
            usage: None,
            finish_reason: None,
            final_response: None,
            chunks: Vec::new(),
            limits: ResponseLimits::default(),
            report: CompletionReport::default(),
        }
    }

    /// Attaches a [`CompletionReport`] to the response emitted with
    /// [`StreamEvent::Done`].
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
            provider: "openai-responses",
            model: self.model.clone(),
            metadata: self.metadata.clone(),
        });
    }

    fn text_mut(&mut self, output_index: usize) -> &mut String {
        let part = self
            .parts
            .entry(output_index)
            .or_insert_with(|| ResponsesPartAccumulator::Text(String::new()));
        if !matches!(part, ResponsesPartAccumulator::Text(_)) {
            *part = ResponsesPartAccumulator::Text(String::new());
        }
        let ResponsesPartAccumulator::Text(text) = part else {
            unreachable!("part was just set to Text")
        };
        text
    }

    fn reasoning_mut(
        &mut self,
        output_index: usize,
    ) -> (&mut String, &mut String, &mut Option<Value>) {
        let part =
            self.parts
                .entry(output_index)
                .or_insert_with(|| ResponsesPartAccumulator::Reasoning {
                    summary: String::new(),
                    text: String::new(),
                    item: None,
                });
        if !matches!(part, ResponsesPartAccumulator::Reasoning { .. }) {
            *part = ResponsesPartAccumulator::Reasoning {
                summary: String::new(),
                text: String::new(),
                item: None,
            };
        }
        let ResponsesPartAccumulator::Reasoning {
            summary,
            text,
            item,
        } = part
        else {
            unreachable!("part was just set to Reasoning")
        };
        (summary, text, item)
    }

    fn tool_call_mut(
        &mut self,
        output_index: usize,
    ) -> (
        &mut Option<String>,
        &mut String,
        &mut String,
        &mut Option<Value>,
    ) {
        let part =
            self.parts
                .entry(output_index)
                .or_insert_with(|| ResponsesPartAccumulator::ToolCall {
                    call_id: None,
                    name: String::new(),
                    arguments: String::new(),
                    item: None,
                });
        if !matches!(part, ResponsesPartAccumulator::ToolCall { .. }) {
            *part = ResponsesPartAccumulator::ToolCall {
                call_id: None,
                name: String::new(),
                arguments: String::new(),
                item: None,
            };
        }
        let ResponsesPartAccumulator::ToolCall {
            call_id,
            name,
            arguments,
            item,
        } = part
        else {
            unreachable!("part was just set to ToolCall")
        };
        (call_id, name, arguments, item)
    }

    fn complete(&mut self, response: Value) -> Result<Vec<StreamEvent>> {
        let mut events = Vec::new();
        self.push_start(&mut events);

        if let Some(output) = response.get("output").and_then(Value::as_array) {
            for (index, item) in output.iter().enumerate() {
                if item.get("type").and_then(Value::as_str) == Some("custom_tool_call") {
                    let call = parse_custom_item(item)?;
                    let accumulator = self.custom_calls.entry(index).or_default();
                    accumulator.merge_identity(&custom_identity(item))?;
                    let NativeToolPayload::RawText(input) = &call.payload else {
                        unreachable!()
                    };
                    accumulator.finish_item(input)?;
                } else if self.custom_calls.contains_key(&index) {
                    return Err(Error::StreamProtocol(
                        "tool_identity_conflict: output item changed type".into(),
                    ));
                }
            }
        }
        if self.custom_calls.values().any(|call| !call.is_completed()) {
            return Err(Error::StreamProtocol("tool_call_incomplete".into()));
        }
        let mut normalized = from_responses_response(response.clone())?;
        let mut terminal_ids = std::collections::HashSet::new();
        for part in &normalized.message.parts {
            let id = match part {
                Part::NativeToolCall(call) => Some(call.call_id.as_str()),
                Part::ToolCall(call) => call.id.as_deref(),
                _ => None,
            };
            if let Some(id) = id
                && !terminal_ids.insert(id.to_owned())
            {
                return Err(Error::StreamProtocol(
                    "tool_identity_conflict: duplicate call_id".into(),
                ));
            }
        }
        if normalized.message.parts.is_empty() && !self.parts.is_empty() {
            // Compatible endpoints that omit the authoritative `output` array
            // on `response.completed` still streamed their items through
            // `output_item.done`; backfill them from the accumulator so the
            // terminal response is never empty.
            normalized.message = Message::new(Role::Assistant, self.accumulated_parts());
        }
        // A terminal response may omit one function item while still carrying
        // text. Backfill completed `output_item.done` function items that are
        // absent from the terminal output; namespaced calls keep their native
        // identity instead of degrading to a flat legacy call.
        let streamed_function_items: Vec<Value> = self
            .parts
            .values()
            .filter_map(|part| match part {
                ResponsesPartAccumulator::ToolCall {
                    item: Some(item), ..
                } => Some(item.clone()),
                _ => None,
            })
            .collect();
        for item in streamed_function_items {
            if item
                .get("status")
                .and_then(Value::as_str)
                .is_some_and(|status| status != "completed")
            {
                continue;
            }
            let Some(call_id) = item.get("call_id").and_then(Value::as_str) else {
                continue;
            };
            if terminal_ids.contains(call_id) {
                continue;
            }
            let mut extra = from_responses_response(json!({"output": [item]}))?;
            for part in extra.message.parts.drain(..) {
                let id = match &part {
                    Part::NativeToolCall(call) => Some(call.call_id.clone()),
                    Part::ToolCall(call) => call.id.clone(),
                    _ => None,
                };
                if let Some(id) = id
                    && terminal_ids.insert(id)
                {
                    normalized.message.parts.push(part);
                }
            }
        }
        let mut streamed_ids = std::collections::HashSet::new();
        for call in self.custom_calls.values() {
            let call = call.completed_call()?;
            if !streamed_ids.insert(call.call_id.clone()) {
                return Err(Error::StreamProtocol(
                    "tool_identity_conflict: duplicate call_id".into(),
                ));
            }
            if terminal_ids.contains(&call.call_id)
                && !normalized.message.parts.iter().any(|part| matches!(part, Part::NativeToolCall(other) if other.call_id == call.call_id && other.kind == NativeToolKind::Custom)) {
                return Err(Error::StreamProtocol("tool_identity_conflict: duplicate call_id".into()));
            }
            if let Some(other) = normalized.message.parts.iter().find_map(|part| match part {
                Part::NativeToolCall(other) if other.call_id == call.call_id => Some(other),
                _ => None,
            }) && (other.namespace != call.namespace
                || other.name != call.name
                || other.item_id != call.item_id
                || other.payload != call.payload)
            {
                return Err(Error::StreamProtocol(
                    "tool_identity_conflict: terminal call differs from stream".into(),
                ));
            }
            if !terminal_ids.contains(&call.call_id) {
                terminal_ids.insert(call.call_id.clone());
                normalized.message.parts.push(Part::NativeToolCall(call));
            }
        }
        normalized.metadata = Some(self.metadata.clone());
        if !self.limits.retain_raw {
            normalized.raw = Value::Null;
        }
        normalized.report = self.report.clone();
        self.usage = normalized.usage.clone();
        self.finish_reason = response
            .get("incomplete_details")
            .and_then(|details| details.get("reason"))
            .and_then(Value::as_str)
            .map(str::to_string);
        self.final_response = Some(normalized.clone());
        self.done = true;

        if let Some(usage) = normalized.usage.clone() {
            events.push(StreamEvent::Usage { usage });
        }
        events.push(StreamEvent::Done {
            response: Box::new(normalized),
            finish_reason: self.finish_reason.clone(),
        });
        Ok(events)
    }

    fn accumulated_parts(&mut self) -> Vec<Part> {
        let mut parts = Vec::new();

        for part in std::mem::take(&mut self.parts).into_values() {
            match part {
                ResponsesPartAccumulator::Text(text) => {
                    if !text.is_empty() {
                        parts.push(Part::text(text));
                    }
                }
                ResponsesPartAccumulator::Reasoning {
                    summary,
                    text,
                    item,
                } => {
                    if let Some(item) = item {
                        parts.push(Part::reasoning(responses_reasoning_part(&item)));
                    } else {
                        parts.push(Part::reasoning(Reasoning {
                            kind: if text.is_empty() {
                                ReasoningKind::Summary
                            } else {
                                ReasoningKind::Text
                            },
                            summary: (!summary.is_empty()).then_some(summary),
                            text: (!text.is_empty()).then_some(text),
                            state: None,
                        }));
                    }
                }
                ResponsesPartAccumulator::ToolCall {
                    call_id,
                    name,
                    arguments,
                    item,
                } => {
                    let arguments = if arguments.trim().is_empty() {
                        json!({})
                    } else {
                        parse_json_or_string(&arguments)
                    };
                    parts.push(Part::ToolCall(ToolCall {
                        id: call_id.or_else(|| {
                            item.as_ref()
                                .and_then(|item| item.get("id"))
                                .and_then(Value::as_str)
                                .map(str::to_string)
                        }),
                        name,
                        arguments,
                        provider_state: None,
                    }));
                }
                ResponsesPartAccumulator::ProviderItem(item) => {
                    parts.push(Part::ProviderItem(responses_server_tool_item(&item)));
                }
            }
        }

        parts
    }

    fn finish_events(&mut self) -> Result<Vec<StreamEvent>> {
        if self.done {
            return Ok(Vec::new());
        }
        if !self.custom_calls.is_empty() {
            return Err(Error::StreamProtocol(
                "tool_call_incomplete: stream ended without response.completed".into(),
            ));
        }
        if let Some(response) = self.final_response.take() {
            self.done = true;
            return Ok(vec![StreamEvent::Done {
                response: Box::new(response),
                finish_reason: self.finish_reason.clone(),
            }]);
        }

        let mut events = Vec::new();
        self.push_start(&mut events);
        let parts = self.accumulated_parts();

        let response = ChatResponse {
            message: Message::new(Role::Assistant, parts),
            usage: self.usage.clone(),
            raw: if self.limits.retain_raw {
                Value::Array(std::mem::take(&mut self.chunks))
            } else {
                Value::Null
            },
            metadata: Some(self.metadata.clone()),
            report: self.report.clone(),
        };
        self.done = true;
        events.push(StreamEvent::Done {
            response: Box::new(response),
            finish_reason: self.finish_reason.clone(),
        });
        Ok(events)
    }
}
impl SseMapper for ResponsesStreamMapper {
    fn configure_limits(&mut self, limits: ResponseLimits) {
        self.limits = limits;
    }
    fn map(&mut self, message: SseMessage) -> Result<Vec<StreamEvent>> {
        if message.data.trim() == "[DONE]" {
            return self.finish_events();
        }

        let event: Value = serde_json::from_str(&message.data)?;
        if self.limits.retain_raw {
            self.chunks.push(event.clone());
        }
        let event_type = event
            .get("type")
            .and_then(Value::as_str)
            .or(message.event.as_deref())
            .unwrap_or_default();

        if matches!(
            event_type,
            "response.output_text.delta"
                | "response.reasoning_summary_text.delta"
                | "response.reasoning_text.delta"
                | "response.function_call_arguments.delta"
        ) && event
            .get("output_index")
            .and_then(Value::as_u64)
            .and_then(|i| usize::try_from(i).ok())
            .is_some_and(|i| self.custom_calls.contains_key(&i))
        {
            return Err(Error::StreamProtocol(
                "tool_identity_conflict: output index reused".into(),
            ));
        }

        if event_type == "response.completed" {
            let Some(response) = event.get("response").cloned() else {
                return Err(Error::StreamProtocol(
                    "response.completed did not contain response".to_string(),
                ));
            };
            return self.complete(response);
        }

        if matches!(
            event_type,
            "response.failed" | "response.incomplete" | "error"
        ) {
            return Err(Error::ProviderStream(event.to_string()));
        }

        let mut events = Vec::new();
        self.push_start(&mut events);

        match event_type {
            "response.custom_tool_call_input.delta" | "response.custom_tool_call_input.done" => {
                let index = custom_event_index(&event)?;
                if self.parts.contains_key(&index) {
                    return Err(Error::StreamProtocol(
                        "tool_identity_conflict: output index reused".into(),
                    ));
                }
                let item_id = event
                    .get("item_id")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                let call = self.custom_calls.entry(index).or_default();
                call.merge_identity(&NativeToolIdentity {
                    item_id: item_id.clone(),
                    ..Default::default()
                })?;
                if event_type.ends_with(".delta") {
                    let delta = event
                        .get("delta")
                        .and_then(Value::as_str)
                        .ok_or_else(|| Error::StreamProtocol("tool_payload_conflict".into()))?;
                    call.push_delta(delta)?;
                    events.push(StreamEvent::NativeToolProgress {
                        index,
                        item_id,
                        input_so_far: call.input().to_owned(),
                    });
                } else {
                    let input = event
                        .get("input")
                        .and_then(Value::as_str)
                        .ok_or_else(|| Error::StreamProtocol("tool_payload_conflict".into()))?;
                    call.finish_input(input)?;
                }
            }
            "response.output_text.delta" => {
                if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                    let output_index = event
                        .get("output_index")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as usize;
                    self.text_mut(output_index).push_str(delta);
                    events.push(StreamEvent::TextDelta {
                        text: delta.to_string(),
                    });
                }
            }
            "response.reasoning_summary_text.delta" => {
                if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                    let output_index = event
                        .get("output_index")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as usize;
                    self.reasoning_mut(output_index).0.push_str(delta);
                    events.push(StreamEvent::ReasoningDelta {
                        kind: ReasoningKind::Summary,
                        text: delta.to_string(),
                    });
                }
            }
            "response.reasoning_text.delta" => {
                if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                    let output_index = event
                        .get("output_index")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as usize;
                    self.reasoning_mut(output_index).1.push_str(delta);
                    events.push(StreamEvent::ReasoningDelta {
                        kind: ReasoningKind::Text,
                        text: delta.to_string(),
                    });
                }
            }
            "response.function_call_arguments.delta" => {
                if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                    let output_index = event
                        .get("output_index")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as usize;
                    self.tool_call_mut(output_index).2.push_str(delta);
                    let (call_id, name, arguments, _) = self.tool_call_mut(output_index);
                    let progress_id = call_id.clone();
                    let progress_name = (!name.is_empty()).then(|| name.clone());
                    let arguments_so_far = arguments.clone();
                    events.push(StreamEvent::ToolCallDelta {
                        index: output_index,
                        id: event
                            .get("item_id")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                        name: None,
                        arguments_delta: delta.to_string(),
                    });
                    events.push(StreamEvent::ToolCallProgress {
                        index: output_index,
                        id: progress_id,
                        name: progress_name,
                        arguments_so_far,
                    });
                }
            }
            "response.output_item.added" => {
                let Some(item) = event.get("item") else {
                    return Ok(events);
                };
                let output_index =
                    if item.get("type").and_then(Value::as_str) == Some("custom_tool_call") {
                        custom_event_index(&event)?
                    } else {
                        event
                            .get("output_index")
                            .and_then(Value::as_u64)
                            .unwrap_or(0) as usize
                    };
                if self.custom_calls.contains_key(&output_index)
                    && item.get("type").and_then(Value::as_str) != Some("custom_tool_call")
                {
                    return Err(Error::StreamProtocol(
                        "tool_identity_conflict: output index reused".into(),
                    ));
                }
                match item.get("type").and_then(Value::as_str) {
                    Some("custom_tool_call") => {
                        if self.parts.contains_key(&output_index) {
                            return Err(Error::StreamProtocol(
                                "tool_identity_conflict: output index reused".into(),
                            ));
                        }
                        self.custom_calls
                            .entry(output_index)
                            .or_default()
                            .merge_identity(&custom_identity(item))?;
                    }
                    Some("function_call") => {
                        let (call_id, name, _, stored_item) = self.tool_call_mut(output_index);
                        *call_id = item
                            .get("call_id")
                            .and_then(Value::as_str)
                            .map(str::to_string);
                        *name = item
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        *stored_item = Some(item.clone());
                        events.push(StreamEvent::ToolCallDelta {
                            index: output_index,
                            id: call_id.clone(),
                            name: Some(name.clone()),
                            arguments_delta: String::new(),
                        });
                    }
                    Some("reasoning") => {
                        *self.reasoning_mut(output_index).2 = Some(item.clone());
                    }
                    Some(kind) if is_responses_server_item(kind) => {
                        self.parts.insert(
                            output_index,
                            ResponsesPartAccumulator::ProviderItem(item.clone()),
                        );
                        events.push(server_tool_status_event(item));
                    }
                    _ => {}
                }
            }
            "response.output_item.done" => {
                let Some(item) = event.get("item") else {
                    return Ok(events);
                };
                let output_index =
                    if item.get("type").and_then(Value::as_str) == Some("custom_tool_call") {
                        custom_event_index(&event)?
                    } else {
                        event
                            .get("output_index")
                            .and_then(Value::as_u64)
                            .unwrap_or(0) as usize
                    };
                if self.custom_calls.contains_key(&output_index)
                    && item.get("type").and_then(Value::as_str) != Some("custom_tool_call")
                {
                    return Err(Error::StreamProtocol(
                        "tool_identity_conflict: output index reused".into(),
                    ));
                }
                match item.get("type").and_then(Value::as_str) {
                    Some("custom_tool_call") => {
                        let parsed = parse_custom_item(item)?;
                        if self.parts.contains_key(&output_index) {
                            return Err(Error::StreamProtocol(
                                "tool_identity_conflict: output index reused".into(),
                            ));
                        }
                        let call = self.custom_calls.entry(output_index).or_default();
                        let was_completed = call.is_completed();
                        call.merge_identity(&custom_identity(item))?;
                        let NativeToolPayload::RawText(input) = &parsed.payload else {
                            unreachable!()
                        };
                        call.finish_item(input)?;
                        if !was_completed {
                            events.push(StreamEvent::NativeToolCompleted {
                                index: output_index,
                                call: call.completed_call()?,
                            });
                        }
                    }
                    Some("function_call") => {
                        let (call_id, name, arguments, stored_item) =
                            self.tool_call_mut(output_index);
                        *call_id = item
                            .get("call_id")
                            .and_then(Value::as_str)
                            .map(str::to_string);
                        *name = item
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        if let Some(full_arguments) = item.get("arguments").and_then(Value::as_str)
                            && arguments.is_empty()
                        {
                            arguments.push_str(full_arguments);
                        }
                        *stored_item = Some(item.clone());
                    }
                    Some("reasoning") => {
                        *self.reasoning_mut(output_index).2 = Some(item.clone());
                    }
                    Some("message") => {
                        if let Some(content) = item.get("content").and_then(Value::as_array) {
                            let text = self.text_mut(output_index);
                            if text.is_empty() {
                                for block in content {
                                    if let Some(block_text) =
                                        block.get("text").and_then(Value::as_str)
                                    {
                                        text.push_str(block_text);
                                    }
                                }
                            }
                        }
                    }
                    Some(kind) if is_responses_server_item(kind) => {
                        self.parts.insert(
                            output_index,
                            ResponsesPartAccumulator::ProviderItem(item.clone()),
                        );
                        events.push(server_tool_status_event(item));
                    }
                    _ => {}
                }
            }
            _ => {}
        }

        Ok(events)
    }

    fn eof_is_terminal(&self) -> bool {
        false
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>> {
        self.finish_events()
    }
}

/// Converts an OpenAI Responses API JSON payload into a neutral
/// [`ChatResponse`].
///
/// # Errors
///
/// Returns an error if the payload cannot be parsed into the neutral
/// representation.
pub(crate) fn from_responses_response(value: Value) -> Result<ChatResponse> {
    let mut parts = Vec::new();
    let mut call_ids = std::collections::HashSet::new();

    if let Some(output) = value.get("output").and_then(Value::as_array) {
        for item in output {
            match item.get("type").and_then(Value::as_str) {
                Some("message") => {
                    if let Some(content) = item.get("content").and_then(Value::as_array) {
                        for block in content {
                            if let Some(text) = block.get("text").and_then(Value::as_str)
                                && !text.is_empty()
                            {
                                parts.push(Part::text(text));
                            }
                        }
                    }
                }
                Some("reasoning") => {
                    parts.push(Part::reasoning(responses_reasoning_part(item)));
                }
                Some("function_call") => {
                    if let Some(id) = item.get("call_id").and_then(Value::as_str)
                        && !call_ids.insert(id.to_owned())
                    {
                        return Err(Error::StreamProtocol(
                            "tool_identity_conflict: duplicate call_id".into(),
                        ));
                    }
                    if item.get("namespace").is_some() {
                        let call_id = item
                            .get("call_id")
                            .and_then(Value::as_str)
                            .filter(|s| !s.is_empty())
                            .ok_or_else(|| {
                                Error::StreamProtocol("tool_identity_conflict".into())
                            })?;
                        let name = item
                            .get("name")
                            .and_then(Value::as_str)
                            .filter(|s| !s.is_empty())
                            .ok_or_else(|| {
                                Error::StreamProtocol("tool_identity_conflict".into())
                            })?;
                        let namespace = item
                            .get("namespace")
                            .and_then(Value::as_str)
                            .filter(|s| !s.is_empty())
                            .ok_or_else(|| {
                                Error::StreamProtocol("tool_identity_conflict".into())
                            })?;
                        let arguments = item
                            .get("arguments")
                            .and_then(Value::as_str)
                            .ok_or_else(|| Error::StreamProtocol("tool_payload_conflict".into()))?;
                        if item
                            .get("status")
                            .and_then(Value::as_str)
                            .is_some_and(|s| s != "completed")
                        {
                            return Err(Error::StreamProtocol("tool_call_incomplete".into()));
                        }
                        parts.push(Part::NativeToolCall(NativeToolCall {
                            kind: NativeToolKind::Function,
                            namespace: Some(namespace.into()),
                            name: name.into(),
                            call_id: call_id.into(),
                            item_id: item.get("id").and_then(Value::as_str).map(str::to_string),
                            payload: NativeToolPayload::JsonArguments(arguments.into()),
                            completion: NativeToolCompletion::Completed,
                            provider_state: Some(ProviderState::new(
                                ProviderStateFormat::OpenAiResponses,
                                item.clone(),
                            )),
                        }));
                        continue;
                    }
                    let id = item
                        .get("call_id")
                        .and_then(Value::as_str)
                        .or_else(|| item.get("id").and_then(Value::as_str))
                        .map(str::to_string);
                    let name = item
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let arguments = item
                        .get("arguments")
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
                Some("custom_tool_call") => {
                    let call = parse_custom_item(item)?;
                    if !call_ids.insert(call.call_id.clone()) {
                        return Err(Error::StreamProtocol(
                            "tool_identity_conflict: duplicate call_id".into(),
                        ));
                    }
                    parts.push(Part::NativeToolCall(call));
                }
                Some(kind) if is_responses_server_item(kind) => {
                    parts.push(Part::ProviderItem(responses_server_tool_item(item)));
                }
                _ => {}
            }
        }
    }

    let usage = value.get("usage").map(responses_usage);

    Ok(ChatResponse {
        message: Message::new(Role::Assistant, parts),
        usage,
        raw: value,
        metadata: None,
        report: CompletionReport::default(),
    })
}

fn parse_custom_item(item: &Value) -> Result<NativeToolCall> {
    if item.get("status").and_then(Value::as_str) != Some("completed") {
        return Err(Error::StreamProtocol("tool_call_incomplete".into()));
    }
    let required = |key: &str| {
        item.get(key)
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .ok_or_else(|| Error::StreamProtocol("tool_identity_conflict".into()))
    };
    let input = item
        .get("input")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::StreamProtocol("tool_payload_conflict".into()))?;
    let call = NativeToolCall {
        kind: NativeToolKind::Custom,
        namespace: item
            .get("namespace")
            .and_then(Value::as_str)
            .map(str::to_string),
        name: required("name")?,
        call_id: required("call_id")?,
        item_id: item.get("id").and_then(Value::as_str).map(str::to_string),
        payload: NativeToolPayload::RawText(input.to_owned()),
        completion: NativeToolCompletion::Completed,
        provider_state: Some(ProviderState::new(
            ProviderStateFormat::OpenAiResponses,
            item.clone(),
        )),
    };
    call.validate_completed()?;
    Ok(call)
}

fn custom_identity(item: &Value) -> NativeToolIdentity {
    let get = |key: &str| item.get(key).and_then(Value::as_str).map(str::to_string);
    NativeToolIdentity {
        namespace: get("namespace"),
        name: get("name"),
        call_id: get("call_id"),
        item_id: get("id"),
    }
}

fn custom_event_index(event: &Value) -> Result<usize> {
    event
        .get("output_index")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| Error::StreamProtocol("tool_identity_conflict: missing output_index".into()))
}

fn is_responses_server_item(kind: &str) -> bool {
    matches!(
        kind,
        "web_search_call"
            | "url_context_call"
            | "file_search_call"
            | "computer_call"
            | "code_interpreter_call"
            | "image_generation_call"
            | "mcp_call"
    )
}

fn responses_server_tool_item(item: &Value) -> crate::types::ServerToolItem {
    let kind = item.get("type").and_then(Value::as_str);
    let call_id = item
        .get("call_id")
        .and_then(Value::as_str)
        .or_else(|| item.get("id").and_then(Value::as_str))
        .map(str::to_string);
    let state = item
        .get("status")
        .and_then(Value::as_str)
        .map(responses_server_tool_state);
    crate::types::ServerToolItem {
        tool: kind.map(|kind| kind.trim_end_matches("_call").to_string()),
        call_id,
        state,
        provider_state: ProviderState::new(ProviderStateFormat::OpenAiResponses, item.clone()),
    }
}

fn server_tool_status_event(item: &Value) -> StreamEvent {
    let item = responses_server_tool_item(item);
    StreamEvent::ServerToolStatus {
        tool: item.tool.unwrap_or_else(|| "unknown".to_string()),
        call_id: item.call_id,
        state: item.state.unwrap_or(crate::types::ServerToolState::Unknown),
    }
}

fn responses_server_tool_state(status: &str) -> crate::types::ServerToolState {
    match status {
        "in_progress" | "searching" | "running" => crate::types::ServerToolState::InProgress,
        "completed" | "succeeded" => crate::types::ServerToolState::Completed,
        "failed" | "incomplete" => crate::types::ServerToolState::Failed,
        _ => crate::types::ServerToolState::Unknown,
    }
}

fn responses_server_tool(tool: &crate::types::ServerTool) -> Value {
    match tool {
        crate::types::ServerTool::WebSearch => json!({"type": "web_search"}),
        crate::types::ServerTool::UrlContext => json!({"type": "url_context"}),
        crate::types::ServerTool::FileSearch => json!({"type": "file_search"}),
        crate::types::ServerTool::Custom { name, config } => {
            let mut value = config.clone();
            if let Some(object) = value.as_object_mut() {
                object.insert("name".to_string(), Value::String(name.clone()));
                value
            } else {
                json!({"type": name, "config": config})
            }
        }
    }
}

fn responses_reasoning_part(item: &Value) -> Reasoning {
    let summary = item
        .get("summary")
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .filter(|text| !text.is_empty());
    let text = item
        .get("content")
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .filter(|text| !text.is_empty());

    let kind = if text.is_some() {
        ReasoningKind::Text
    } else if summary.is_some() {
        ReasoningKind::Summary
    } else {
        ReasoningKind::Encrypted
    };

    Reasoning {
        kind,
        summary,
        text,
        state: Some(ProviderState::new(
            ProviderStateFormat::OpenAiResponses,
            item.clone(),
        )),
    }
}

fn responses_usage(usage: &Value) -> Usage {
    Usage {
        input_tokens: usage.get("input_tokens").and_then(Value::as_u64),
        output_tokens: usage.get("output_tokens").and_then(Value::as_u64),
        total_tokens: usage.get("total_tokens").and_then(Value::as_u64),
        cache_read_tokens: usage
            .pointer("/input_tokens_details/cached_tokens")
            .and_then(Value::as_u64),
        cache_write_tokens: usage
            .get("cache_creation_input_tokens")
            .and_then(Value::as_u64),
        reasoning_tokens: usage
            .pointer("/output_tokens_details/reasoning_tokens")
            .and_then(Value::as_u64),
        raw: Some(usage.clone()),
    }
}

fn parse_json_or_string(value: &str) -> Value {
    serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_string()))
}

fn responses_reasoning(config: Option<&ReasoningConfig>) -> Option<Value> {
    let config = config?;
    let mut value = serde_json::Map::new();

    if let Some(effort) = config.effort {
        value.insert(
            "effort".to_string(),
            json!(match effort {
                crate::types::ReasoningEffort::None => "none",
                crate::types::ReasoningEffort::Minimal => "minimal",
                crate::types::ReasoningEffort::Low => "low",
                crate::types::ReasoningEffort::Medium => "medium",
                crate::types::ReasoningEffort::High => "high",
                crate::types::ReasoningEffort::XHigh => "xhigh",
                crate::types::ReasoningEffort::Max => "max",
                crate::types::ReasoningEffort::Ultra => "ultra",
            }),
        );
    } else if config.mode == Some(ReasoningMode::Disabled) {
        value.insert("effort".to_string(), json!("none"));
    }

    if let Some(summary) = config.summary {
        value.insert(
            "summary".to_string(),
            json!(match summary {
                crate::types::ReasoningSummary::Auto => "auto",
                crate::types::ReasoningSummary::Concise => "concise",
                crate::types::ReasoningSummary::Detailed => "detailed",
            }),
        );
    }

    (!value.is_empty()).then_some(Value::Object(value))
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

fn image_source_url(source: &crate::types::ImageSource) -> String {
    match source {
        crate::types::ImageSource::Url { url } => url.clone(),
        crate::types::ImageSource::Base64 { media_type, data } => {
            format!("data:{media_type};base64,{data}")
        }
        crate::types::ImageSource::FileRef { uri, .. } => uri.clone(),
    }
}

fn responses_content_value(parts: &[Part]) -> Value {
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
                Part::Text { text, .. } => Some(json!({"type": "input_text", "text": text})),
                Part::ImageUrl { image_url } => Some(json!({
                    "type": "input_image",
                    "image_url": image_url.url,
                })),
                Part::Image { image } => Some(json!({
                    "type": "input_image",
                    "image_url": image_source_url(&image.source),
                })),
                _ => None,
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn in_place_developer_becomes_developer_input_item() {
        use crate::normalize::{Protocol, SystemPlacement};
        let mut profile = crate::profile::ProviderProfile::new("test-resp");
        profile.normalize.system_placement = SystemPlacement::FirstToTopRestInPlace;
        let request = crate::types::ChatRequest::new(vec![
            crate::types::Message::system("base"),
            crate::types::Message::user("hi"),
            crate::types::Message::new(
                crate::types::Role::Developer,
                vec![crate::types::Part::text("skill body")],
            ),
        ]);
        let (normalized, _report) = super::super::normalize_for_protocol(
            &request,
            Protocol::OpenAiResponses,
            Some(&profile),
        )
        .expect("normalize");
        let body = to_responses_body("m", &normalized, &request);
        assert_eq!(body["instructions"], "base");
        let input = body["input"].as_array().expect("input");
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[1]["role"], "developer");
        assert_eq!(input[1]["content"], "skill body");
    }

    use super::*;
    use crate::normalize::{Protocol, normalize};

    #[test]
    fn native_declaration_wire_shapes_are_not_hosted_tools() {
        let function = crate::types::ToolSpec::new("read", "read", json!({"type":"object"}));
        let custom = ScopedToolDeclaration::Custom {
            name: "apply_patch".into(),
            description: "apply".into(),
            format: CustomToolFormat::Text,
        };
        let flat = ToolDeclaration::Tool {
            tool: custom.clone(),
        };
        assert_eq!(
            responses_declaration(&flat),
            json!({
                "type":"custom", "name":"apply_patch", "description":"apply"
            })
        );
        let namespace = ToolDeclaration::Namespace {
            name: "functions".into(),
            description: "workspace".into(),
            tools: vec![function.into(), custom],
        };
        assert_eq!(
            responses_declaration(&namespace),
            json!({
                "type":"namespace", "name":"functions", "description":"workspace", "tools":[
                    {"type":"function", "name":"read", "description":"read", "parameters":{"type":"object"}, "strict":false},
                    {"type":"custom", "name":"apply_patch", "description":"apply"}
                ]
            })
        );
        let grammar = ToolDeclaration::Tool {
            tool: ScopedToolDeclaration::Custom {
                name: "apply_patch".into(),
                description: "apply".into(),
                format: CustomToolFormat::Grammar {
                    syntax: "lark".into(),
                    definition: "start: /x/".into(),
                },
            },
        };
        assert_eq!(
            responses_declaration(&grammar)["format"],
            json!({
                "type":"grammar", "syntax":"lark", "definition":"start: /x/"
            })
        );
    }

    #[test]
    fn native_declarations_are_encoded_in_the_outbound_body() {
        let request = ChatRequest::user("hi").tool_declarations([ToolDeclaration::Namespace {
            name: "functions".into(),
            description: "workspace".into(),
            tools: vec![
                crate::ToolSpec::new("read", "read", json!({"type": "object"})).into(),
                ScopedToolDeclaration::Custom {
                    name: "apply_patch".into(),
                    description: "apply".into(),
                    format: CustomToolFormat::Text,
                },
            ],
        }]);
        let (normalized, _) = normalize(&request, Protocol::OpenAiResponses).expect("normalize");
        let body = to_responses_body("m", &normalized, &request);
        assert_eq!(body["tools"][0]["type"], "namespace");
        assert_eq!(body["tools"][0]["name"], "functions");
        assert_eq!(body["tools"][0]["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["tools"][1]["type"], "custom");
        assert_eq!(body["tools"][0]["tools"][1]["name"], "apply_patch");
    }

    #[test]
    fn custom_response_and_result_replay_preserve_identity_and_raw_input() {
        let patch = "*** Begin Patch\n+你好 \\\"x\\\"\n*** End Patch\n";
        let output = json!({"type":"custom_tool_call", "id":"ctc_1", "call_id":"call_1", "name":"apply_patch", "namespace":"functions", "input":patch, "status":"completed"});
        let decoded = from_responses_response(json!({"output":[output]})).unwrap();
        let request = ChatRequest::new([
            decoded.message,
            Message::tool_results([crate::types::ToolResult::error(
                Some("call_1".into()),
                "apply_patch",
                "denied",
            )]),
        ]);
        let (normalized, _) = normalize(&request, Protocol::OpenAiResponses).unwrap();
        let body = to_responses_body("m", &normalized, &request);
        assert_eq!(body["input"][0]["type"], "custom_tool_call");
        assert_eq!(body["input"][0]["id"], "ctc_1");
        assert_eq!(body["input"][0]["call_id"], "call_1");
        assert_eq!(body["input"][0]["namespace"], "functions");
        assert_eq!(body["input"][0]["input"], patch);
        assert_eq!(
            body["input"][1],
            json!({"type":"custom_tool_call_output", "call_id":"call_1", "output":"denied"})
        );
        assert!(from_responses_response(json!({"output":[{"type":"custom_tool_call", "id":"ctc_1", "name":"apply_patch", "input":patch, "status":"completed"}]})).is_err());
        assert!(from_responses_response(json!({"output":[{"type":"custom_tool_call", "call_id":"call_1", "name":"apply_patch", "input":patch, "status":"in_progress"}]})).is_err());
    }

    #[test]
    fn custom_stream_reconciles_snapshots_and_backfills_with_text() {
        fn send(mapper: &mut ResponsesStreamMapper, event: Value) -> Result<Vec<StreamEvent>> {
            mapper.map(SseMessage {
                event: None,
                data: event.to_string(),
                id: None,
            })
        }
        let mut mapper = ResponsesStreamMapper::new(
            "m".into(),
            ResponseMetadata {
                status: 200,
                request_id: None,
                headers: HeaderMap::new(),
            },
        );
        let item = json!({"type":"custom_tool_call", "id":"ctc_1", "call_id":"call_1", "name":"apply_patch", "input":"你好\n*** End Patch", "status":"completed"});
        send(&mut mapper, json!({"type":"response.output_item.added", "output_index":1, "item":{"type":"custom_tool_call", "id":"ctc_1", "call_id":"call_1", "name":"apply_patch", "input":"", "status":"in_progress"}})).unwrap();
        send(&mut mapper, json!({"type":"response.custom_tool_call_input.delta", "output_index":1, "item_id":"ctc_1", "delta":"你好\n"})).unwrap();
        send(&mut mapper, json!({"type":"response.custom_tool_call_input.done", "output_index":1, "item_id":"ctc_1", "input":"你好\n*** End Patch"})).unwrap();
        let completed = send(
            &mut mapper,
            json!({"type":"response.output_item.done", "output_index":1, "item":item}),
        )
        .unwrap();
        assert_eq!(
            completed
                .iter()
                .filter(|e| matches!(e, StreamEvent::NativeToolCompleted { .. }))
                .count(),
            1
        );
        let repeated = send(
            &mut mapper,
            json!({"type":"response.output_item.done", "output_index":1, "item":item}),
        )
        .unwrap();
        assert!(
            !repeated
                .iter()
                .any(|e| matches!(e, StreamEvent::NativeToolCompleted { .. }))
        );
        let terminal = send(&mut mapper, json!({"type":"response.completed", "response":{"status":"completed", "output":[{"type":"message", "content":[{"text":"hello"}]}]}})).unwrap();
        let response = terminal
            .iter()
            .find_map(|e| {
                if let StreamEvent::Done { response, .. } = e {
                    Some(response)
                } else {
                    None
                }
            })
            .unwrap();
        assert!(
            response
                .message
                .parts
                .iter()
                .any(|p| matches!(p, Part::NativeToolCall(call) if call.call_id == "call_1"))
        );
        assert!(
            response
                .message
                .parts
                .iter()
                .any(|p| matches!(p, Part::Text {text, ..} if text == "hello"))
        );

        let mut conflict = ResponsesStreamMapper::new(
            "m".into(),
            ResponseMetadata {
                status: 200,
                request_id: None,
                headers: HeaderMap::new(),
            },
        );
        send(
            &mut conflict,
            json!({"type":"response.output_item.done", "output_index":0, "item":item}),
        )
        .unwrap();
        let mut changed = item.clone();
        changed["input"] = json!("different");
        assert!(
            send(
                &mut conflict,
                json!({"type":"response.completed", "response":{"output":[changed]}})
            )
            .is_err()
        );
        let mut incomplete = ResponsesStreamMapper::new(
            "m".into(),
            ResponseMetadata {
                status: 200,
                request_id: None,
                headers: HeaderMap::new(),
            },
        );
        send(&mut incomplete, json!({"type":"response.custom_tool_call_input.delta", "output_index":0, "item_id":"ctc_1", "delta":"*** End Patch"})).unwrap();
        assert!(
            send(
                &mut incomplete,
                json!({"type":"response.completed", "response":{"output":[]}})
            )
            .is_err()
        );
    }

    #[test]
    fn namespaced_function_replays_without_losing_address() {
        let response = from_responses_response(json!({"output":[{
            "type":"function_call", "namespace":"functions", "name":"read", "id":"fc_1", "call_id":"call_2", "arguments":"{\"path\":\"a\"}", "status":"completed"
        }]})).unwrap();
        let request = ChatRequest::new([
            response.message,
            Message::tool_results([crate::types::ToolResult::new(
                Some("call_2".into()),
                "read",
                "ok",
            )]),
        ]);
        let (normalized, _) = normalize(&request, Protocol::OpenAiResponses).unwrap();
        let body = to_responses_body("m", &normalized, &request);
        assert_eq!(body["input"][0]["namespace"], "functions");
        assert_eq!(body["input"][0]["arguments"], "{\"path\":\"a\"}");
        assert_eq!(body["input"][1]["type"], "function_call_output");
    }

    #[test]
    fn streamed_namespaced_function_is_backfilled_when_terminal_omits_it() {
        fn send(mapper: &mut ResponsesStreamMapper, event: Value) -> Result<Vec<StreamEvent>> {
            mapper.map(SseMessage {
                event: None,
                data: event.to_string(),
                id: None,
            })
        }
        let mut mapper = ResponsesStreamMapper::new(
            "m".into(),
            ResponseMetadata {
                status: 200,
                request_id: None,
                headers: HeaderMap::new(),
            },
        );
        let item = json!({
            "type":"function_call", "namespace":"functions", "name":"read",
            "id":"fc_1", "call_id":"call_2", "arguments":"{\"path\":\"a\"}",
            "status":"completed"
        });
        send(
            &mut mapper,
            json!({
                "type":"response.output_item.added", "output_index":0,
                "item":{"type":"function_call", "namespace":"functions", "name":"read",
                        "id":"fc_1", "call_id":"call_2", "arguments":"", "status":"in_progress"}
            }),
        )
        .unwrap();
        send(
            &mut mapper,
            json!({
                "type":"response.output_item.done", "output_index":0, "item":item
            }),
        )
        .unwrap();
        let terminal = send(
            &mut mapper,
            json!({
                "type":"response.completed",
                "response":{"status":"completed", "output":[
                    {"type":"message", "content":[{"text":"hello"}]}
                ]}
            }),
        )
        .unwrap();
        let response = terminal
            .iter()
            .find_map(|event| match event {
                StreamEvent::Done { response, .. } => Some(response),
                _ => None,
            })
            .expect("done");
        assert!(response.message.parts.iter().any(|part| matches!(
            part,
            Part::NativeToolCall(call)
                if call.call_id == "call_2"
                    && call.namespace.as_deref() == Some("functions")
                    && call.name == "read"
        )));
    }

    #[test]
    fn optional_tool_arguments_are_not_implicitly_made_required() {
        let schema = json!({"type":"object", "properties": {
            "path":{"type":"string"}, "image_index":{"type":"integer"}
        }, "additionalProperties":false});
        let request = ChatRequest::user("inspect").tools([crate::types::ToolSpec::new(
            "read_image",
            "inspect image",
            schema.clone(),
        )]);
        let (normalized, _) = normalize(&request, Protocol::OpenAiResponses).unwrap();
        let body = to_responses_body("m", &normalized, &request);
        assert_eq!(body["tools"][0]["strict"], false);
        assert_eq!(body["tools"][0]["parameters"], schema);
    }

    #[test]
    fn caller_can_enable_strict_schema_without_schema_rewriting() {
        let schema = json!({"type":"object","properties":{"path":{"type":["string","null"]}},"required":["path"],"additionalProperties":false});
        let request = ChatRequest::user("inspect").tools([crate::types::ToolSpec::new(
            "image",
            "inspect",
            schema.clone(),
        )
        .strict(true)]);
        let (normalized, _) = normalize(&request, Protocol::OpenAiResponses).unwrap();
        let body = to_responses_body("m", &normalized, &request);
        assert_eq!(body["tools"][0]["strict"], true);
        assert_eq!(body["tools"][0]["parameters"], schema);
    }

    #[test]
    fn neutral_request_controls_map_to_responses() {
        let request = ChatRequest::user("hi")
            .tools([crate::types::ToolSpec::new(
                "lookup",
                "lookup",
                json!({"type": "object"}),
            )])
            .tool_choice(ToolChoice::Tool("lookup".to_string()))
            .response_format(ResponseFormat::JsonObject);
        validate_responses_request(&request).unwrap();
        let (normalized, _) = normalize(&request, Protocol::OpenAiResponses).unwrap();
        let body = to_responses_body("gpt-5", &normalized, &request);

        assert_eq!(body["tool_choice"]["type"], "function");
        assert_eq!(body["tool_choice"]["name"], "lookup");
        assert_eq!(body["text"]["format"]["type"], "json_object");
    }

    #[test]
    fn responses_rejects_unsupported_stop_and_seed() {
        let stop = ChatRequest::user("hi").stop(["END"]);
        assert!(matches!(
            validate_responses_request(&stop),
            Err(Error::Unsupported(_))
        ));

        let seed = ChatRequest::user("hi").seed(42);
        assert!(matches!(
            validate_responses_request(&seed),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn reasoning_item_round_trips_exactly() {
        let item = json!({
            "id": "rs_123",
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "简短摘要"}],
            "content": [{"type": "reasoning_text", "text": "可读思考"}],
            "encrypted_content": "encrypted-state",
            "status": "completed"
        });
        let response = from_responses_response(json!({
            "output": [
                item.clone(),
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "最终答案"}]
                }
            ]
        }))
        .unwrap();

        let reasoning = response.reasoning().next().unwrap();
        assert_eq!(reasoning.kind, ReasoningKind::Text);
        assert_eq!(reasoning.summary.as_deref(), Some("简短摘要"));
        assert_eq!(reasoning.text.as_deref(), Some("可读思考"));
        assert_eq!(
            reasoning.state.as_ref().map(|state| &state.data),
            Some(&item)
        );
        assert_eq!(response.text(), "最终答案");

        let request = ChatRequest::new([response.message]);
        let (normalized, _) = normalize(&request, Protocol::OpenAiResponses).unwrap();
        let body = to_responses_body("gpt-5", &normalized, &request);
        assert_eq!(body.pointer("/input/0"), Some(&item));
    }

    #[test]
    fn reasoning_config_maps_to_responses_fields() {
        let request = ChatRequest::user("hi").reasoning(
            ReasoningConfig::new()
                .effort(crate::types::ReasoningEffort::High)
                .summary(crate::types::ReasoningSummary::Detailed)
                .include_encrypted(true),
        );
        let (normalized, _) = normalize(&request, Protocol::OpenAiResponses).unwrap();
        let body = to_responses_body("gpt-5", &normalized, &request);

        assert_eq!(body["reasoning"]["effort"], "high");
        assert_eq!(body["reasoning"]["summary"], "detailed");
        assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
    }

    #[test]
    fn stream_events_assemble_final_response() {
        let metadata = ResponseMetadata {
            status: 200,
            request_id: Some("req_responses".to_string()),
            headers: HeaderMap::new(),
        };
        let mut mapper = ResponsesStreamMapper::new("gpt-5".to_string(), metadata);
        let events = [
            json!({
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": ""
                }
            }),
            json!({
                "type": "response.reasoning_summary_text.delta",
                "output_index": 0,
                "item_id": "rs_1",
                "summary_index": 0,
                "delta": "先查资料。"
            }),
            json!({
                "type": "response.function_call_arguments.delta",
                "output_index": 1,
                "item_id": "fc_1",
                "delta": "{\"q\":\"Rust\"}"
            }),
            json!({
                "type": "response.output_text.delta",
                "output_index": 2,
                "item_id": "msg_1",
                "content_index": 0,
                "delta": "答案"
            }),
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "rs_1",
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "先查资料。"}],
                    "encrypted_content": "encrypted"
                }
            }),
            json!({
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "output": [
                        {
                            "id": "rs_1",
                            "type": "reasoning",
                            "summary": [{"type": "summary_text", "text": "先查资料。"}],
                            "encrypted_content": "encrypted"
                        },
                        {
                            "id": "fc_1",
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "lookup",
                            "arguments": "{\"q\":\"Rust\"}"
                        },
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "答案"}]
                        }
                    ],
                    "usage": {"input_tokens": 12, "output_tokens": 4}
                }
            }),
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
            StreamEvent::ReasoningDelta {
                kind: ReasoningKind::Summary,
                text
            } if text == "先查资料。"
        )));
        assert!(stream_events.iter().any(|event| matches!(
            event,
            StreamEvent::TextDelta { text } if text == "答案"
        )));
        assert!(stream_events.iter().any(|event| matches!(
            event,
            StreamEvent::ToolCallDelta {
                arguments_delta,
                ..
            } if arguments_delta == "{\"q\":\"Rust\"}"
        )));

        let StreamEvent::Done {
            response,
            finish_reason,
        } = stream_events.last().unwrap()
        else {
            panic!("expected done event");
        };
        assert_eq!(finish_reason.as_deref(), None);
        assert_eq!(response.text(), "答案");
        assert_eq!(response.reasoning_text(), "先查资料。");
        assert_eq!(response.tool_calls().next().unwrap().name, "lookup");
        assert_eq!(response.usage.as_ref().unwrap().input_tokens, Some(12));
    }

    #[test]
    fn completed_without_output_backfills_accumulated_items() {
        let metadata = ResponseMetadata {
            status: 200,
            request_id: None,
            headers: HeaderMap::new(),
        };
        let mut mapper = ResponsesStreamMapper::new("gpt-5".to_string(), metadata);
        for event in [
            json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "item_id": "msg_1",
                "content_index": 0,
                "delta": "answer"
            }),
            json!({
                "type": "response.function_call_arguments.delta",
                "output_index": 1,
                "item_id": "fc_1",
                "delta": "{\"q\":\"Rust\"}"
            }),
            json!({
                "type": "response.output_item.done",
                "output_index": 1,
                "item": {
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": "{\"q\":\"Rust\"}"
                }
            }),
        ] {
            mapper
                .map(SseMessage {
                    event: None,
                    data: event.to_string(),
                    id: None,
                })
                .expect("stream event");
        }
        let events = mapper
            .map(SseMessage {
                event: None,
                data: json!({
                    "type": "response.completed",
                    "response": {"status": "completed"}
                })
                .to_string(),
                id: None,
            })
            .expect("completed");
        let StreamEvent::Done { response, .. } = events.last().expect("done event") else {
            panic!("expected done event");
        };
        assert_eq!(response.text(), "answer");
        let call = response.tool_calls().next().expect("backfilled tool call");
        assert_eq!(call.name, "lookup");
        assert_eq!(call.arguments, json!({"q": "Rust"}));
        assert!(
            response.raw.is_object(),
            "real terminal keeps the response object"
        );
    }
}
