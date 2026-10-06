mod anthropic;
mod endpoint;
mod gemini;
mod openai_chat;
mod openai_responses;

use std::future::Future;
use std::pin::Pin;
use std::time::Instant;

use async_trait::async_trait;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;

use crate::audit::{AuditContext, AuditOutcome};
use crate::cancel::CancellationToken;
use crate::error::{Error, Result};
use crate::headers::{RequestOptions, TransportConfig};
use crate::normalize::Protocol;
use crate::profile::{
    ProfileSelector, ProviderProfile, ProviderRequestOptions, ToolCallContentMode,
};
use crate::report::{CompletionReport, WireAction, WireFeature};
use crate::retry::{RetryPolicy, RetryProvider, next_delay, parse_retry_headers};
use crate::stream::{ModelStream, StreamReconnectPolicy, sse_stream_with_reconnect};
use crate::types::{ChatRequest, ResponseMetadata, ToolChoice, ToolSpec};

pub use anthropic::{Anthropic, AnthropicMessages};
pub use endpoint::{EndpointAdapter, OpenAICompatible};
pub use gemini::{Gemini, GeminiGenerateContent};
pub use openai_chat::{OpenAI, OpenAIChat};
pub use openai_responses::OpenAIResponses;

/// A provider adapter.
///
/// The adapter receives a provider-neutral [`ChatRequest`] and is responsible
/// for:
///
/// 1. calling the crate's normalization layer;
/// 2. serializing the normalized conversation to the provider wire format;
/// 3. mapping the provider response back to [`crate::ChatResponse`].
#[async_trait]
pub trait ModelAdapter: Send + Sync {
    /// Stable identifier of the provider behind this adapter,
    /// e.g. `"anthropic-messages"`.
    fn provider_name(&self) -> &'static str;

    /// Model identifier used for this adapter, e.g. `"claude-sonnet-4"`.
    fn model_name(&self) -> &str;

    /// Complete a request with default [`RequestOptions`].
    ///
    /// # Errors
    ///
    /// Returns an error if the request cannot be normalized, the provider
    /// rejects it, or the response fails to parse.
    async fn complete(&self, request: &ChatRequest) -> Result<crate::ChatResponse> {
        let options = RequestOptions::default();
        self.complete_with(request, &options).await
    }

    /// Complete a request with explicit [`RequestOptions`].
    ///
    /// This is the primary entry point for non-streaming inference: the
    /// adapter normalizes the [`ChatRequest`], serializes it to the provider
    /// wire format, sends it, and parses the response back into
    /// [`crate::ChatResponse`].
    ///
    /// # Errors
    ///
    /// Returns an error if the request cannot be normalized or validated,
    /// the provider returns an error status, or the response fails to parse.
    async fn complete_with(
        &self,
        request: &ChatRequest,
        options: &RequestOptions,
    ) -> Result<crate::ChatResponse>;

    /// Start a streaming response with default request options.
    async fn stream(&self, request: &ChatRequest) -> Result<ModelStream> {
        let options = RequestOptions::default();
        self.stream_with(request, &options).await
    }

    /// Start a streaming response.
    ///
    /// The default implementation reports that streaming is unsupported.
    /// Built-in adapters override this method.
    async fn stream_with(
        &self,
        _request: &ChatRequest,
        _options: &RequestOptions,
    ) -> Result<ModelStream> {
        Err(Error::Unsupported(format!(
            "{} does not implement streaming",
            self.provider_name()
        )))
    }
}

pub(crate) fn validate_server_tools(protocol: Protocol, request: &ChatRequest) -> Result<()> {
    if !request.server_tools.is_empty() && protocol != Protocol::OpenAiResponses {
        return Err(Error::Unsupported(format!(
            "{protocol:?} does not implement generic server-side tools"
        )));
    }
    Ok(())
}

pub(crate) fn validate_tool_choice(tools: &[ToolSpec], choice: Option<&ToolChoice>) -> Result<()> {
    match choice {
        None | Some(ToolChoice::None) => Ok(()),
        Some(ToolChoice::Auto | ToolChoice::Required) if tools.is_empty() => Err(
            Error::InvalidRequest("tool_choice requires at least one tool".to_string()),
        ),
        Some(ToolChoice::Tool(name)) if tools.is_empty() => Err(Error::InvalidRequest(format!(
            "tool_choice `{name}` requires at least one tool"
        ))),
        Some(ToolChoice::Tool(name)) if !tools.iter().any(|tool| tool.name == *name) => Err(
            Error::InvalidRequest(format!("tool_choice references unknown tool `{name}`")),
        ),
        _ => Ok(()),
    }
}

pub(crate) fn json_body_bytes(body: &Value) -> Result<u64> {
    Ok(serde_json::to_vec(body)?.len() as u64)
}

pub(crate) fn merge_extra_body(
    body: &mut Value,
    request: &ChatRequest,
    protocol: Protocol,
    profile: Option<&ProviderProfile>,
) -> Result<()> {
    let reserved = reserved_body_fields(protocol);
    let body = body.as_object_mut().ok_or_else(|| {
        Error::InvalidProfile(format!("{protocol:?} request body must be a JSON object"))
    })?;

    if let Some(profile) = profile {
        merge_body_fields(body, &profile.request.extra_body, protocol, reserved)?;
    }
    merge_body_fields(body, &request.extra_body, protocol, reserved)
}

fn merge_body_fields(
    body: &mut serde_json::Map<String, Value>,
    fields: &serde_json::Map<String, Value>,
    protocol: Protocol,
    reserved: &[&str],
) -> Result<()> {
    for (key, value) in fields {
        if reserved.contains(&key.as_str()) {
            return Err(Error::ReservedExtraBodyField {
                protocol: protocol_name(protocol),
                field: key.clone(),
            });
        }
        body.insert(key.clone(), value.clone());
    }
    Ok(())
}

fn reserved_body_fields(protocol: Protocol) -> &'static [&'static str] {
    match protocol {
        Protocol::OpenAiChat => &[
            "model",
            "messages",
            "stream",
            "stream_options",
            "tools",
            "temperature",
            "max_tokens",
            "max_completion_tokens",
        ],
        Protocol::OpenAiResponses => &[
            "model",
            "input",
            "instructions",
            "stream",
            "tools",
            "temperature",
            "max_output_tokens",
            "reasoning",
            "include",
        ],
        Protocol::AnthropicMessages => &[
            "model",
            "max_tokens",
            "messages",
            "system",
            "tools",
            "temperature",
            "thinking",
            "stream",
        ],
        Protocol::GeminiGenerateContent => {
            &["contents", "systemInstruction", "tools", "generationConfig"]
        }
    }
}

pub(crate) fn protocol_name(protocol: Protocol) -> &'static str {
    match protocol {
        Protocol::OpenAiChat => "openai-chat",
        Protocol::OpenAiResponses => "openai-responses",
        Protocol::AnthropicMessages => "anthropic-messages",
        Protocol::GeminiGenerateContent => "gemini-generate-content",
    }
}

pub(crate) fn validate_profile_selector(selector: &ProfileSelector) -> Result<()> {
    match selector {
        ProfileSelector::Generic | ProfileSelector::Custom(_) => Ok(()),
        ProfileSelector::Builtin(id) => {
            if crate::profile::ProfileRegistry::new().resolve(id).is_some() {
                Ok(())
            } else {
                Err(Error::Unsupported(format!(
                    "built-in provider profile `{id}` is not registered; use ProfileSelector::Custom"
                )))
            }
        }
    }
}

pub(crate) fn resolve_api_key(
    provider: &'static str,
    explicit: Option<&str>,
    env_name: &'static str,
) -> Result<String> {
    if let Some(key) = explicit
        && !key.trim().is_empty()
    {
        return Ok(key.to_string());
    }

    let _ = provider;
    match std::env::var(env_name) {
        Ok(value) if !value.trim().is_empty() => Ok(value),
        _ => Err(Error::MissingApiKey(env_name.to_string())),
    }
}

pub(crate) async fn prepare_headers(
    transport: &TransportConfig,
    base_headers: &HeaderMap,
    options: &RequestOptions,
) -> Result<HeaderMap> {
    let mut headers = base_headers.clone();
    transport.apply_to_async(&mut headers, options).await?;
    Ok(headers)
}

pub(crate) fn effective_retry_policy(
    policy: &RetryPolicy,
    options: &RequestOptions,
) -> RetryPolicy {
    let mut policy = policy.clone();
    if policy.require_idempotency_key && options.idempotency_key.is_none() {
        policy.max_attempts = 1;
    }
    policy
}

pub(crate) fn audit_outcome_for_error(error: &Error) -> AuditOutcome {
    if matches!(error, Error::Cancelled) {
        AuditOutcome::Cancelled
    } else {
        AuditOutcome::Failure
    }
}

pub(crate) fn effective_provider_request_options(
    profile: Option<&ProviderProfile>,
    request: &ProviderRequestOptions,
) -> ProviderRequestOptions {
    let mut effective = profile
        .map(|profile| profile.request_options.clone())
        .unwrap_or_default();
    if request.tool_call_content.is_some() {
        effective.tool_call_content = request.tool_call_content;
    }
    if request.require_provider_parameters.is_some() {
        effective.require_provider_parameters = request.require_provider_parameters;
    }
    if request.do_sample.is_some() {
        effective.do_sample = request.do_sample;
    }
    if request.include_stream_usage.is_some() {
        effective.include_stream_usage = request.include_stream_usage;
    }
    if request.prompt_cache_key.is_some() {
        effective
            .prompt_cache_key
            .clone_from(&request.prompt_cache_key);
    }
    if request.user.is_some() {
        effective.user.clone_from(&request.user);
    }
    effective
}

pub(crate) fn apply_provider_request_options(
    body: &mut Value,
    protocol: Protocol,
    options: &ProviderRequestOptions,
    streaming: bool,
) -> Result<()> {
    let body = body.as_object_mut().ok_or_else(|| {
        Error::InvalidProfile(format!("{protocol:?} request body must be a JSON object"))
    })?;

    if let Some(prompt_cache_key) = &options.prompt_cache_key
        && matches!(protocol, Protocol::OpenAiChat | Protocol::OpenAiResponses)
    {
        body.insert(
            "prompt_cache_key".to_string(),
            Value::String(prompt_cache_key.clone()),
        );
    }
    if let Some(user) = &options.user
        && protocol == Protocol::OpenAiChat
    {
        body.insert("user".to_string(), Value::String(user.clone()));
    }
    if let Some(do_sample) = options.do_sample {
        body.insert("do_sample".to_string(), Value::Bool(do_sample));
    }
    if let Some(require_provider_parameters) = options.require_provider_parameters {
        body.insert(
            "require_provider_parameters".to_string(),
            Value::Bool(require_provider_parameters),
        );
    }
    if streaming
        && protocol == Protocol::OpenAiChat
        && let Some(include_usage) = options.include_stream_usage
    {
        let stream_options = body
            .entry("stream_options".to_string())
            .or_insert_with(|| Value::Object(serde_json::Map::new()));
        if let Some(stream_options) = stream_options.as_object_mut() {
            stream_options.insert("include_usage".to_string(), Value::Bool(include_usage));
        }
    }

    if protocol == Protocol::OpenAiChat
        && let Some(mode) = options.tool_call_content
        && let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut)
    {
        for message in messages {
            if !message.get("tool_calls").is_some_and(Value::is_array) {
                continue;
            }
            match mode {
                ToolCallContentMode::Auto => {}
                ToolCallContentMode::Null => {
                    message["content"] = Value::Null;
                }
                ToolCallContentMode::Omit => {
                    if let Some(message) = message.as_object_mut() {
                        message.remove("content");
                    }
                }
                ToolCallContentMode::Empty => {
                    message["content"] = Value::String(String::new());
                }
            }
        }
    }

    Ok(())
}

pub(crate) fn apply_request_transform<'a>(
    request: &'a ChatRequest,
    options: &RequestOptions,
) -> Result<std::borrow::Cow<'a, ChatRequest>> {
    match &options.request_transform {
        Some(transform) => Ok(std::borrow::Cow::Owned(transform.transform(request)?)),
        None => Ok(std::borrow::Cow::Borrowed(request)),
    }
}

pub(crate) fn completion_report(
    protocol: Protocol,
    request: &ChatRequest,
    normalization: crate::normalize::NormalizeStats,
    transform_applied: bool,
) -> CompletionReport {
    let mut report = CompletionReport::from_normalization(normalization);
    report.transform_applied = transform_applied;
    let mut downgraded_tool_results = 0_u32;
    let mut unsupported_provider_items = 0_u32;
    let mut unsupported_tool_metadata = 0_u32;
    for message in &request.messages {
        for part in &message.parts {
            match part {
                crate::types::Part::ToolResult(result) => {
                    let has_structured = result
                        .parts
                        .iter()
                        .any(|part| !matches!(part, crate::types::ToolResultPart::Text { .. }));
                    if has_structured
                        && matches!(protocol, Protocol::OpenAiChat | Protocol::OpenAiResponses)
                    {
                        downgraded_tool_results += 1;
                    }
                    if result.status.is_some() || result.metadata.is_some() {
                        unsupported_tool_metadata += 1;
                    }
                }
                crate::types::Part::ProviderItem(item)
                    if item.provider_state.format != protocol.provider_state_format() =>
                {
                    unsupported_provider_items += 1;
                }
                _ => {}
            }
        }
    }
    if downgraded_tool_results > 0 {
        report.wire.push(WireAction::Downgraded {
            feature: WireFeature::StructuredToolResult,
            to: "text".to_string(),
            count: downgraded_tool_results,
        });
    }
    if unsupported_provider_items > 0 {
        report.wire.push(WireAction::Unsupported {
            feature: WireFeature::ProviderItem,
            count: unsupported_provider_items,
        });
    }
    if unsupported_tool_metadata > 0 {
        report.wire.push(WireAction::Unsupported {
            feature: WireFeature::RequestField,
            count: unsupported_tool_metadata,
        });
    }
    if !request.server_tools.is_empty() && protocol != Protocol::OpenAiResponses {
        report.wire.push(WireAction::Unsupported {
            feature: WireFeature::ServerTool,
            count: request.server_tools.len() as u32,
        });
    }
    report
}

pub(crate) struct ReconnectRequest {
    provider: &'static str,
    client: reqwest::Client,
    retry_policy: RetryPolicy,
    retry_provider: RetryProvider,
    transport: TransportConfig,
    options: RequestOptions,
    audit: Option<AuditContext>,
    url: String,
    base_headers: HeaderMap,
    query: Vec<(String, String)>,
    body: Value,
}

pub(crate) struct ReconnectRequestParts {
    pub provider: &'static str,
    pub client: reqwest::Client,
    pub retry_policy: RetryPolicy,
    pub retry_provider: RetryProvider,
    pub transport: TransportConfig,
    pub options: RequestOptions,
    pub audit: Option<AuditContext>,
    pub url: String,
    pub base_headers: HeaderMap,
    pub query: Vec<(String, String)>,
    pub body: Value,
}

impl ReconnectRequest {
    pub(crate) fn new(parts: ReconnectRequestParts) -> Self {
        Self {
            provider: parts.provider,
            client: parts.client,
            retry_policy: parts.retry_policy,
            retry_provider: parts.retry_provider,
            transport: parts.transport,
            options: parts.options,
            audit: parts.audit,
            url: parts.url,
            base_headers: parts.base_headers,
            query: parts.query,
            body: parts.body,
        }
    }

    fn connect(
        &self,
        last_event_id: Option<String>,
    ) -> Pin<Box<dyn Future<Output = Result<reqwest::Response>> + Send>> {
        let provider = self.provider;
        let client = self.client.clone();
        let retry_policy = self.retry_policy.clone();
        let retry_provider = self.retry_provider;
        let transport = self.transport.clone();
        let options = self.options.clone();
        let audit = self.audit.clone();
        let url = self.url.clone();
        let base_headers = self.base_headers.clone();
        let query = self.query.clone();
        let body = self.body.clone();

        Box::pin(async move {
            let cancellation = options.cancellation.clone();
            send_stream_retry(
                provider,
                retry_provider,
                &retry_policy,
                audit.as_ref(),
                cancellation.as_ref(),
                || async {
                    let mut headers = prepare_headers(&transport, &base_headers, &options).await?;
                    if let Some(last_event_id) = &last_event_id {
                        headers.insert(
                            HeaderName::from_static("last-event-id"),
                            HeaderValue::from_str(last_event_id)?,
                        );
                    }
                    let mut request_builder =
                        client.post(&url).headers(headers).query(&query).json(&body);
                    if let Some(timeout) = options.timeout {
                        request_builder = request_builder.timeout(timeout);
                    }
                    Ok(request_builder)
                },
            )
            .await
        })
    }
}

pub(crate) fn finish_sse_stream<M>(
    response: reqwest::Response,
    mapper: M,
    policy: Option<StreamReconnectPolicy>,
    reconnect: Option<ReconnectRequest>,
    cancellation: Option<CancellationToken>,
    idle_timeout: Option<std::time::Duration>,
    audit: Option<AuditContext>,
) -> ModelStream
where
    M: crate::stream::SseMapper,
{
    match (policy, reconnect) {
        (Some(policy), Some(reconnect)) => sse_stream_with_reconnect(
            response,
            mapper,
            policy,
            move |last_event_id| reconnect.connect(last_event_id),
            cancellation,
            idle_timeout,
            audit,
        ),
        _ => crate::stream::sse_stream_with_audit(
            response,
            mapper,
            cancellation,
            idle_timeout,
            audit,
        ),
    }
}

pub(crate) async fn send_json(
    provider: &'static str,
    retry_provider: RetryProvider,
    request: reqwest::RequestBuilder,
    audit: Option<&AuditContext>,
    attempt: u32,
    attempt_started: Instant,
    cancellation: Option<&CancellationToken>,
) -> Result<(Value, ResponseMetadata)> {
    let response = send_request(request, cancellation).await?;
    let metadata = ResponseMetadata::from_response(&response);
    let status = response.status();
    if let Some(audit) = audit {
        audit.response_headers(
            attempt,
            attempt_started.elapsed(),
            status.as_u16(),
            metadata.request_id.clone(),
            status.is_success(),
        );
    }
    let retry_directive = parse_retry_headers(retry_provider, response.headers());
    let retry_after = retry_directive.map(|directive| directive.delay);
    let retry_source = retry_directive.map(|directive| directive.source);
    let text = read_response_text(response, cancellation).await?;
    if let Some(audit) = audit {
        audit.add_response_bytes(text.len());
    }

    if !status.is_success() {
        return Err(Error::Api {
            provider,
            status: status.as_u16(),
            body: text,
            retry_after,
            retry_source,
            request_id: metadata.request_id.clone(),
        });
    }

    let value = serde_json::from_str(&text).map_err(Error::from)?;
    Ok((value, metadata))
}

async fn send_request(
    request: reqwest::RequestBuilder,
    cancellation: Option<&CancellationToken>,
) -> Result<reqwest::Response> {
    if let Some(cancellation) = cancellation {
        if cancellation.is_cancelled() {
            return Err(Error::Cancelled);
        }
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => Err(Error::Cancelled),
            response = request.send() => response.map_err(Error::from),
        }
    } else {
        request.send().await.map_err(Error::from)
    }
}

async fn read_response_text(
    response: reqwest::Response,
    cancellation: Option<&CancellationToken>,
) -> Result<String> {
    if let Some(cancellation) = cancellation {
        if cancellation.is_cancelled() {
            return Err(Error::Cancelled);
        }
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => Err(Error::Cancelled),
            text = response.text() => text.map_err(Error::from),
        }
    } else {
        response.text().await.map_err(Error::from)
    }
}

async fn sleep_or_cancel(
    delay: std::time::Duration,
    cancellation: Option<&CancellationToken>,
) -> Result<()> {
    if let Some(cancellation) = cancellation {
        if cancellation.is_cancelled() {
            return Err(Error::Cancelled);
        }
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => Err(Error::Cancelled),
            _ = tokio::time::sleep(delay) => Ok(()),
        }
    } else {
        tokio::time::sleep(delay).await;
        Ok(())
    }
}

async fn build_request_or_cancel<Fut>(
    future: Fut,
    cancellation: Option<&CancellationToken>,
) -> Result<reqwest::RequestBuilder>
where
    Fut: Future<Output = Result<reqwest::RequestBuilder>>,
{
    if let Some(cancellation) = cancellation {
        if cancellation.is_cancelled() {
            return Err(Error::Cancelled);
        }
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => Err(Error::Cancelled),
            request = future => request,
        }
    } else {
        future.await
    }
}

pub(crate) async fn send_json_retry<F, Fut>(
    provider: &'static str,
    retry_provider: RetryProvider,
    policy: &RetryPolicy,
    audit: Option<&AuditContext>,
    cancellation: Option<&CancellationToken>,
    mut make_request: F,
) -> Result<(Value, ResponseMetadata)>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<reqwest::RequestBuilder>>,
{
    if policy.max_attempts == 0 {
        return Err(Error::InvalidRetryPolicy(
            "max_attempts must be at least 1".to_string(),
        ));
    }

    for attempt in 0..policy.max_attempts {
        if cancellation.is_some_and(CancellationToken::is_cancelled) {
            return Err(Error::Cancelled);
        }
        if let Some(audit) = audit {
            audit.attempt_started(attempt);
        }
        let attempt_started = Instant::now();
        let request = match build_request_or_cancel(make_request(), cancellation).await {
            Ok(request) => request,
            Err(error) => {
                if let Some(audit) = audit {
                    audit.attempt_finished_with_request_id(
                        attempt,
                        attempt_started.elapsed(),
                        None,
                        None,
                        Some(&error),
                    );
                }
                let Some(delay) = next_delay(policy, attempt, &error) else {
                    return Err(error);
                };
                if let Some(audit) = audit {
                    audit.retry_scheduled(attempt, attempt + 1, delay, &error);
                }
                sleep_or_cancel(delay, cancellation).await?;
                continue;
            }
        };

        match send_json(
            provider,
            retry_provider,
            request,
            audit,
            attempt,
            attempt_started,
            cancellation,
        )
        .await
        {
            Ok(response) => {
                if let Some(audit) = audit {
                    audit.attempt_finished_with_request_id(
                        attempt,
                        attempt_started.elapsed(),
                        Some(response.1.status),
                        response.1.request_id.clone(),
                        None,
                    );
                }
                return Ok(response);
            }
            Err(error) => {
                if let Some(audit) = audit {
                    audit.attempt_finished_with_request_id(
                        attempt,
                        attempt_started.elapsed(),
                        error.status(),
                        error.request_id().map(str::to_string),
                        Some(&error),
                    );
                }
                let Some(delay) = next_delay(policy, attempt, &error) else {
                    return Err(error);
                };
                if let Some(audit) = audit {
                    audit.retry_scheduled(attempt, attempt + 1, delay, &error);
                }
                sleep_or_cancel(delay, cancellation).await?;
            }
        }
    }

    unreachable!("the loop returns on success or on the final error")
}

pub(crate) async fn send_stream_retry<F, Fut>(
    provider: &'static str,
    retry_provider: RetryProvider,
    policy: &RetryPolicy,
    audit: Option<&AuditContext>,
    cancellation: Option<&CancellationToken>,
    mut make_request: F,
) -> Result<reqwest::Response>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<reqwest::RequestBuilder>>,
{
    if policy.max_attempts == 0 {
        return Err(Error::InvalidRetryPolicy(
            "max_attempts must be at least 1".to_string(),
        ));
    }

    for attempt in 0..policy.max_attempts {
        if cancellation.is_some_and(CancellationToken::is_cancelled) {
            return Err(Error::Cancelled);
        }
        if let Some(audit) = audit {
            audit.attempt_started(attempt);
        }
        let attempt_started = Instant::now();
        let request = match build_request_or_cancel(make_request(), cancellation).await {
            Ok(request) => request,
            Err(error) => {
                if let Some(audit) = audit {
                    audit.attempt_finished_with_request_id(
                        attempt,
                        attempt_started.elapsed(),
                        None,
                        None,
                        Some(&error),
                    );
                }
                let Some(delay) = next_delay(policy, attempt, &error) else {
                    return Err(error);
                };
                if let Some(audit) = audit {
                    audit.retry_scheduled(attempt, attempt + 1, delay, &error);
                }
                sleep_or_cancel(delay, cancellation).await?;
                continue;
            }
        };

        let response = match send_request(request, cancellation).await {
            Ok(response) => response,
            Err(error) => {
                if let Some(audit) = audit {
                    audit.attempt_finished_with_request_id(
                        attempt,
                        attempt_started.elapsed(),
                        error.status(),
                        None,
                        Some(&error),
                    );
                }
                let Some(delay) = next_delay(policy, attempt, &error) else {
                    return Err(error);
                };
                if let Some(audit) = audit {
                    audit.retry_scheduled(attempt, attempt + 1, delay, &error);
                }
                sleep_or_cancel(delay, cancellation).await?;
                continue;
            }
        };

        let status = response.status();
        let metadata = ResponseMetadata::from_response(&response);
        if let Some(audit) = audit {
            audit.response_headers(
                attempt,
                attempt_started.elapsed(),
                status.as_u16(),
                metadata.request_id.clone(),
                status.is_success(),
            );
        }
        if status.is_success() {
            if let Some(content_type) = response.headers().get(CONTENT_TYPE)
                && let Ok(content_type) = content_type.to_str()
                && !content_type
                    .to_ascii_lowercase()
                    .contains("text/event-stream")
            {
                let error = Error::StreamProtocol(format!(
                    "{provider} returned {content_type}, expected text/event-stream"
                ));
                if let Some(audit) = audit {
                    audit.attempt_finished_with_request_id(
                        attempt,
                        attempt_started.elapsed(),
                        Some(status.as_u16()),
                        metadata.request_id,
                        Some(&error),
                    );
                }
                return Err(error);
            }
            if let Some(audit) = audit {
                audit.attempt_finished_with_request_id(
                    attempt,
                    attempt_started.elapsed(),
                    Some(status.as_u16()),
                    metadata.request_id,
                    None,
                );
            }
            return Ok(response);
        }

        let retry_directive = parse_retry_headers(retry_provider, response.headers());
        let retry_after = retry_directive.map(|directive| directive.delay);
        let retry_source = retry_directive.map(|directive| directive.source);
        let request_id = ResponseMetadata::from_response(&response).request_id;
        let body = read_response_text(response, cancellation).await?;
        if let Some(audit) = audit {
            audit.add_response_bytes(body.len());
        }
        let error = Error::Api {
            provider,
            status: status.as_u16(),
            body,
            retry_after,
            retry_source,
            request_id,
        };
        if let Some(audit) = audit {
            audit.attempt_finished_with_request_id(
                attempt,
                attempt_started.elapsed(),
                error.status(),
                error.request_id().map(str::to_string),
                Some(&error),
            );
        }

        let Some(delay) = next_delay(policy, attempt, &error) else {
            return Err(error);
        };
        if let Some(audit) = audit {
            audit.retry_scheduled(attempt, attempt + 1, delay, &error);
        }
        sleep_or_cancel(delay, cancellation).await?;
    }

    unreachable!("the loop returns on success or on the final error")
}

/// Normalize a request with protocol defaults plus profile overrides.
///
/// System placement is an OpenAI-protocol capability: Anthropic and Gemini
/// keep merged top-level system behavior regardless of the profile.
pub(crate) fn normalize_for_protocol(
    request: &crate::types::ChatRequest,
    protocol: crate::normalize::Protocol,
    profile: Option<&crate::profile::ProviderProfile>,
) -> crate::error::Result<(
    crate::normalize::NormalizedChat,
    crate::normalize::NormalizeReport,
)> {
    let mut options = crate::normalize::NormalizeOptions::for_protocol(protocol);
    if let Some(profile) = profile
        && matches!(
            protocol,
            crate::normalize::Protocol::OpenAiChat | crate::normalize::Protocol::OpenAiResponses
        )
    {
        options.system_placement = profile.normalize.system_placement;
    }
    crate::normalize::normalize_with_options(request, options)
}
