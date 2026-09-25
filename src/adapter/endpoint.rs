use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderName, HeaderValue};
use serde_json::json;

use super::anthropic::{
    AnthropicStreamMapper, from_anthropic_response, to_anthropic_body, validate_anthropic_request,
};
use super::gemini::{
    GeminiStreamMapper, from_gemini_response, to_gemini_body, validate_gemini_request,
};
use super::openai_chat::{
    OpenAiChatStreamMapper, from_openai_chat_response_with_aliases,
    to_openai_chat_body_with_profile,
};
use super::openai_responses::{
    ResponsesStreamMapper, from_responses_response, to_responses_body, validate_responses_request,
};
use super::{
    ModelAdapter, ReconnectRequest, ReconnectRequestParts, apply_provider_request_options,
    apply_request_transform, audit_outcome_for_error, completion_report,
    effective_provider_request_options, effective_retry_policy, finish_sse_stream, json_body_bytes,
    merge_extra_body, prepare_headers, protocol_name, send_json_retry, send_stream_retry,
    validate_profile_selector,
};
use crate::audit::{AuditContext, ProfileAuditSnapshot};
use crate::error::{Error, Result};
use crate::headers::{RequestOptions, TransportConfig};
use crate::normalize::{NormalizeReport, Protocol, normalize};
use crate::profile::{
    AuthStyle, Capabilities, EndpointSpec, ModelProfile, ReasoningAliases, ReasoningReplayPolicy,
};
use crate::report::CompletionReport;
use crate::retry::{RetryPolicy, RetryProvider};
use crate::stream::ModelStream;
use crate::types::{ChatRequest, ChatResponse, Part, ResponseMetadata};

const ANTHROPIC_VERSION: &str = "2023-06-01";

/// A protocol-dispatching adapter for explicitly configured endpoints.
///
/// The caller selects the protocol, path, and auth through [`EndpointSpec`].
/// The adapter never infers them from a base URL.
pub struct EndpointAdapter {
    model: String,
    endpoint: EndpointSpec,
    provider_profile: Option<crate::profile::ProviderProfile>,
    api_key: Option<String>,
    api_key_env: Option<String>,
    model_profiles: Vec<ModelProfile>,
    retry_policy: RetryPolicy,
    transport: TransportConfig,
    client: reqwest::Client,
}

/// Backwards-compatible name for the OpenAI Chat-focused first iteration.
///
/// New code should prefer [`EndpointAdapter`] because it also dispatches
/// Responses, Anthropic Messages, and Gemini generateContent.
pub type OpenAICompatible = EndpointAdapter;

impl EndpointAdapter {
    pub fn new(model: impl Into<String>, endpoint: EndpointSpec) -> Self {
        let provider_profile = endpoint.resolved_profile();
        Self {
            model: model.into(),
            endpoint,
            provider_profile,
            api_key: None,
            api_key_env: None,
            model_profiles: Vec::new(),
            retry_policy: RetryPolicy::default(),
            transport: TransportConfig::default(),
            client: reqwest::Client::new(),
        }
    }

    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn api_key_from_env(mut self, env_name: impl Into<String>) -> Self {
        self.api_key_env = Some(env_name.into());
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

    /// Add a model-level profile. The first matching profile wins.
    pub fn model_profile(mut self, profile: ModelProfile) -> Self {
        self.model_profiles.push(profile);
        self
    }

    pub fn model_profiles(mut self, profiles: impl IntoIterator<Item = ModelProfile>) -> Self {
        self.model_profiles.extend(profiles);
        self
    }

    pub fn endpoint(&self) -> &EndpointSpec {
        &self.endpoint
    }

    fn ensure_supported(&self) -> Result<()> {
        validate_profile_selector(&self.endpoint.profile)
    }

    fn selected_model_profile(&self) -> Option<&ModelProfile> {
        self.model_profiles
            .iter()
            .find(|profile| profile.matches(&self.model))
    }

    fn capabilities(&self) -> Capabilities {
        self.selected_model_profile()
            .and_then(|profile| profile.capabilities.clone())
            .or_else(|| {
                self.provider_profile
                    .as_ref()
                    .map(|profile| profile.capabilities.clone())
            })
            .unwrap_or_default()
    }

    fn audit_profile_snapshot(&self) -> ProfileAuditSnapshot {
        ProfileAuditSnapshot {
            provider_profile_id: self
                .provider_profile
                .as_ref()
                .map(|profile| profile.id.to_string()),
            model_profile_selected: self.selected_model_profile().is_some(),
            capabilities: Some(self.capabilities()),
        }
    }

    fn reasoning_aliases(&self) -> ReasoningAliases {
        self.selected_model_profile()
            .filter(|profile| !profile.reasoning_aliases.is_empty())
            .map(|profile| profile.reasoning_aliases.clone())
            .or_else(|| {
                self.provider_profile
                    .as_ref()
                    .map(|profile| profile.reasoning.aliases.clone())
            })
            .unwrap_or_default()
    }

    fn reasoning_replay_policy(&self) -> ReasoningReplayPolicy {
        self.selected_model_profile()
            .and_then(|profile| profile.replay)
            .or_else(|| {
                self.provider_profile
                    .as_ref()
                    .map(|profile| profile.reasoning.replay)
            })
            .unwrap_or(ReasoningReplayPolicy::SameProvider)
    }

    fn ensure_request_supported(&self, request: &ChatRequest, streaming: bool) -> Result<()> {
        let capabilities = self.capabilities();

        if request.reasoning.is_some() && !capabilities.reasoning.supported {
            return Err(Error::Unsupported(
                "reasoning is disabled by the selected profile".to_string(),
            ));
        }
        if let Some(reasoning) = &request.reasoning {
            if reasoning.summary.is_some() && !capabilities.reasoning.summary {
                return Err(Error::Unsupported(
                    "reasoning summaries are disabled by the selected profile".to_string(),
                ));
            }
            if reasoning.include_text == Some(true) && !capabilities.reasoning.text {
                return Err(Error::Unsupported(
                    "readable reasoning text is disabled by the selected profile".to_string(),
                ));
            }
            if reasoning.include_encrypted == Some(true) && !capabilities.reasoning.encrypted {
                return Err(Error::Unsupported(
                    "encrypted reasoning is disabled by the selected profile".to_string(),
                ));
            }
        }

        let expected_state = self.endpoint.protocol.provider_state_format();
        let has_replay_state = request.messages.iter().any(|message| {
            message.parts.iter().any(|part| match part {
                Part::Reasoning(reasoning) => reasoning
                    .state
                    .as_ref()
                    .is_some_and(|state| state.format == expected_state),
                Part::Text { provider_state, .. } => provider_state
                    .as_ref()
                    .is_some_and(|state| state.format == expected_state),
                Part::ToolCall(call) => call
                    .provider_state
                    .as_ref()
                    .is_some_and(|state| state.format == expected_state),
                _ => false,
            })
        });
        if has_replay_state {
            let policy = self.reasoning_replay_policy();
            if self.endpoint.protocol != Protocol::OpenAiChat
                && !matches!(
                    policy,
                    ReasoningReplayPolicy::SameProvider
                        | ReasoningReplayPolicy::PreserveInHistory
                        | ReasoningReplayPolicy::ModelDefined
                )
            {
                return Err(Error::Unsupported(format!(
                    "replay policy {policy:?} is not implemented for {:?}",
                    self.endpoint.protocol
                )));
            }
            if !capabilities.reasoning.replay
                && !(self.endpoint.protocol == Protocol::OpenAiChat
                    && policy == ReasoningReplayPolicy::Never)
            {
                return Err(Error::Unsupported(
                    "reasoning replay is disabled by the selected profile".to_string(),
                ));
            }
        }

        if !request.tools.is_empty() && !capabilities.tools.supported {
            return Err(Error::Unsupported(
                "tool calling is disabled by the selected profile".to_string(),
            ));
        }
        if !request.server_tools.is_empty() {
            if !capabilities.server_side_state {
                return Err(Error::Unsupported(
                    "server-side tools are disabled by the selected profile".to_string(),
                ));
            }
            if self.endpoint.protocol != Protocol::OpenAiResponses {
                return Err(Error::Unsupported(
                    "the selected protocol does not implement generic server-side tools"
                        .to_string(),
                ));
            }
        }
        if request.tool_choice.is_some() && !capabilities.tool_choice {
            return Err(Error::Unsupported(
                "tool choice is disabled by the selected profile".to_string(),
            ));
        }
        if request.response_format.is_some() && !capabilities.structured_output {
            return Err(Error::Unsupported(
                "structured output is disabled by the selected profile".to_string(),
            ));
        }
        if request.seed.is_some() && !capabilities.seed {
            return Err(Error::Unsupported(
                "seed is disabled by the selected profile".to_string(),
            ));
        }
        if !request.stop.is_empty() && !capabilities.stop {
            return Err(Error::Unsupported(
                "stop sequences are disabled by the selected profile".to_string(),
            ));
        }
        if streaming && !request.tools.is_empty() && !capabilities.tools.streaming {
            return Err(Error::Unsupported(
                "streaming tool calls are disabled by the selected profile".to_string(),
            ));
        }
        if streaming && !capabilities.streaming.supported {
            return Err(Error::Unsupported(
                "streaming is disabled by the selected profile".to_string(),
            ));
        }
        if !capabilities.multimodal
            && request.messages.iter().any(|message| {
                message
                    .parts
                    .iter()
                    .any(|part| matches!(part, Part::ImageUrl { .. }))
            })
        {
            return Err(Error::Unsupported(
                "multimodal input is disabled by the selected profile".to_string(),
            ));
        }

        Ok(())
    }

    fn resolve_api_key(&self) -> Result<Option<String>> {
        if self.endpoint.auth == AuthStyle::None {
            return Ok(None);
        }

        if let Some(key) = &self.api_key
            && !key.trim().is_empty()
        {
            return Ok(Some(key.clone()));
        }

        if let Some(env_name) = &self.api_key_env
            && let Ok(key) = std::env::var(env_name)
            && !key.trim().is_empty()
        {
            return Ok(Some(key));
        }

        Err(Error::MissingApiKey(
            self.api_key_env
                .clone()
                .unwrap_or_else(|| "generic endpoint API key".to_string()),
        ))
    }

    async fn request_parts(
        &self,
        options: &RequestOptions,
        streaming: bool,
    ) -> Result<(HeaderMap, Vec<(String, String)>)> {
        self.ensure_supported()?;
        let profile = self.provider_profile.as_ref();
        let key = self.resolve_api_key()?;

        let mut headers = profile
            .map(|profile| profile.request.extra_headers.clone())
            .unwrap_or_default();
        let mut query = profile
            .map(|profile| profile.request.extra_query.clone())
            .unwrap_or_default();
        query.extend(options.extra_query.iter().cloned());

        let reserved_query: &[&str] = match self.endpoint.protocol {
            Protocol::GeminiGenerateContent => &["key", "alt"],
            _ => &[],
        };
        for (name, _) in &query {
            if reserved_query.contains(&name.as_str()) {
                return Err(Error::ReservedExtraQuery {
                    protocol: protocol_name(self.endpoint.protocol),
                    parameter: name.clone(),
                });
            }
        }

        if let Some(key) = key {
            match &self.endpoint.auth {
                AuthStyle::Bearer => {
                    headers.insert(
                        AUTHORIZATION,
                        HeaderValue::from_str(&format!("Bearer {key}"))?,
                    );
                }
                AuthStyle::XApiKey { header } => {
                    headers.insert(header.clone(), HeaderValue::from_str(&key)?);
                }
                AuthStyle::QueryKey { parameter } => {
                    query.push((parameter.clone(), key));
                }
                AuthStyle::Custom {
                    header,
                    value_prefix,
                } => {
                    let value = format!("{}{key}", value_prefix.as_deref().unwrap_or_default());
                    headers.insert(header.clone(), HeaderValue::from_str(&value)?);
                }
                AuthStyle::None => {}
            }
        }

        if self.endpoint.protocol == Protocol::AnthropicMessages {
            headers
                .entry(HeaderName::from_static("anthropic-version"))
                .or_insert_with(|| HeaderValue::from_static(ANTHROPIC_VERSION));
        }

        if streaming && self.endpoint.protocol == Protocol::GeminiGenerateContent {
            query.push(("alt".to_string(), "sse".to_string()));
        }

        Ok((headers, query))
    }

    #[cfg(test)]
    fn build_body(
        &self,
        request: &ChatRequest,
        streaming: bool,
    ) -> Result<(serde_json::Value, NormalizeReport)> {
        self.build_body_with_options(request, streaming, &RequestOptions::default())
    }

    fn build_body_with_options(
        &self,
        request: &ChatRequest,
        streaming: bool,
        options: &RequestOptions,
    ) -> Result<(serde_json::Value, NormalizeReport)> {
        let profile = self.provider_profile.as_ref();
        let model_profile = self.selected_model_profile();
        let (normalized, report) = normalize(request, self.endpoint.protocol)?;

        let (mut body, report) = match self.endpoint.protocol {
            Protocol::OpenAiChat => {
                let mut body = to_openai_chat_body_with_profile(
                    &self.model,
                    &normalized,
                    request,
                    profile,
                    model_profile,
                )?;
                if streaming {
                    body["stream"] = json!(true);
                    body["stream_options"] = json!({"include_usage": true});
                }
                (body, report)
            }
            Protocol::OpenAiResponses => {
                validate_responses_request(request)?;
                let mut body = to_responses_body(&self.model, &normalized, request);
                merge_extra_body(&mut body, request, Protocol::OpenAiResponses, profile)?;
                if streaming {
                    body["stream"] = json!(true);
                }
                (body, report)
            }
            Protocol::AnthropicMessages => {
                validate_anthropic_request(request)?;
                let mut body = to_anthropic_body(&self.model, &normalized, request);
                merge_extra_body(&mut body, request, Protocol::AnthropicMessages, profile)?;
                if streaming {
                    body["stream"] = json!(true);
                }
                (body, report)
            }
            Protocol::GeminiGenerateContent => {
                validate_gemini_request(request)?;
                let mut body = to_gemini_body(&normalized, request);
                merge_extra_body(&mut body, request, Protocol::GeminiGenerateContent, profile)?;
                (body, report)
            }
        };
        let provider_request =
            effective_provider_request_options(profile, &options.provider_request);
        apply_provider_request_options(
            &mut body,
            self.endpoint.protocol,
            &provider_request,
            streaming,
        )?;
        Ok((body, report))
    }

    fn retry_provider(&self) -> RetryProvider {
        match self.endpoint.protocol {
            Protocol::OpenAiChat | Protocol::OpenAiResponses => RetryProvider::OpenAi,
            Protocol::AnthropicMessages => RetryProvider::Anthropic,
            Protocol::GeminiGenerateContent => RetryProvider::Google,
        }
    }

    fn parse_response(&self, value: serde_json::Value) -> Result<ChatResponse> {
        match self.endpoint.protocol {
            Protocol::OpenAiChat => {
                from_openai_chat_response_with_aliases(value, &self.reasoning_aliases())
            }
            Protocol::OpenAiResponses => from_responses_response(value),
            Protocol::AnthropicMessages => from_anthropic_response(value),
            Protocol::GeminiGenerateContent => from_gemini_response(value),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn stream_mapper(
        &self,
        response: reqwest::Response,
        policy: Option<crate::stream::StreamReconnectPolicy>,
        reconnect: Option<ReconnectRequest>,
        cancellation: Option<crate::CancellationToken>,
        idle_timeout: Option<std::time::Duration>,
        report: CompletionReport,
        audit: Option<AuditContext>,
    ) -> ModelStream {
        let metadata = ResponseMetadata::from_response(&response);
        match self.endpoint.protocol {
            Protocol::OpenAiChat => finish_sse_stream(
                response,
                OpenAiChatStreamMapper::with_aliases(
                    self.model.clone(),
                    metadata,
                    self.reasoning_aliases(),
                )
                .with_report(report),
                policy,
                reconnect,
                cancellation,
                idle_timeout,
                audit,
            ),
            Protocol::OpenAiResponses => finish_sse_stream(
                response,
                ResponsesStreamMapper::new(self.model.clone(), metadata).with_report(report),
                policy,
                reconnect,
                cancellation,
                idle_timeout,
                audit,
            ),
            Protocol::AnthropicMessages => finish_sse_stream(
                response,
                AnthropicStreamMapper::new(self.model.clone(), metadata).with_report(report),
                policy,
                reconnect,
                cancellation,
                idle_timeout,
                audit,
            ),
            Protocol::GeminiGenerateContent => finish_sse_stream(
                response,
                GeminiStreamMapper::new(self.model.clone(), metadata).with_report(report),
                policy,
                reconnect,
                cancellation,
                idle_timeout,
                audit,
            ),
        }
    }
}

#[async_trait]
impl ModelAdapter for EndpointAdapter {
    fn provider_name(&self) -> &'static str {
        "endpoint-adapter"
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    async fn complete_with(
        &self,
        request: &ChatRequest,
        options: &RequestOptions,
    ) -> Result<ChatResponse> {
        self.ensure_supported()?;
        let transformed = apply_request_transform(request, options)?;
        let transform_applied = matches!(transformed, std::borrow::Cow::Owned(_));
        let request = transformed.as_ref();
        self.ensure_request_supported(request, false)?;
        let (body, report) = self.build_body_with_options(request, false, options)?;
        let request_body_bytes = json_body_bytes(&body)?;
        let (base_headers, query) = self.request_parts(options, false).await?;
        let url = self.endpoint.url_for(&self.model, false)?;
        let provider = protocol_name(self.endpoint.protocol);
        let retry_policy = effective_retry_policy(&self.retry_policy, options);
        let audit = self.transport.audit_context(
            self.provider_name(),
            provider,
            &self.model,
            options,
            Some(self.audit_profile_snapshot()),
            request_body_bytes,
        );
        if let Some(audit) = &audit {
            audit.request_started();
            if transform_applied {
                audit.transform_applied();
            }
            audit.normalization(report.stats());
        }

        let result = send_json_retry(
            provider,
            self.retry_provider(),
            &retry_policy,
            audit.as_ref(),
            options.cancellation.as_ref(),
            || async {
                let headers = prepare_headers(&self.transport, &base_headers, options).await?;
                let mut request_builder = self
                    .client
                    .post(&url)
                    .headers(headers)
                    .query(&query)
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

        let mut response = match self.parse_response(value) {
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
            self.endpoint.protocol,
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
        self.ensure_supported()?;
        let transformed = apply_request_transform(request, options)?;
        let transform_applied = matches!(transformed, std::borrow::Cow::Owned(_));
        let request = transformed.as_ref();
        self.ensure_request_supported(request, true)?;
        let (body, report) = self.build_body_with_options(request, true, options)?;
        let request_body_bytes = json_body_bytes(&body)?;
        let (base_headers, query) = self.request_parts(options, true).await?;
        let url = self.endpoint.url_for(&self.model, true)?;
        let provider = protocol_name(self.endpoint.protocol);
        let retry_policy = effective_retry_policy(&self.retry_policy, options);
        let audit = self.transport.audit_context(
            self.provider_name(),
            provider,
            &self.model,
            options,
            Some(self.audit_profile_snapshot()),
            request_body_bytes,
        );
        if let Some(audit) = &audit {
            audit.request_started();
            if transform_applied {
                audit.transform_applied();
            }
            audit.normalization(report.stats());
        }

        let response = match send_stream_retry(
            provider,
            self.retry_provider(),
            &retry_policy,
            audit.as_ref(),
            options.cancellation.as_ref(),
            || async {
                let headers = prepare_headers(&self.transport, &base_headers, options).await?;
                let mut request_builder = self
                    .client
                    .post(&url)
                    .headers(headers)
                    .query(&query)
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

        let reconnect = options.stream_reconnect.map(|_| {
            ReconnectRequest::new(ReconnectRequestParts {
                provider,
                client: self.client.clone(),
                retry_policy: retry_policy.clone(),
                retry_provider: self.retry_provider(),
                transport: self.transport.clone(),
                options: options.clone(),
                audit: audit.clone(),
                url,
                base_headers,
                query,
                body,
            })
        });
        Ok(self.stream_mapper(
            response,
            options.stream_reconnect,
            reconnect,
            options.cancellation.clone(),
            options.idle_timeout,
            completion_report(
                self.endpoint.protocol,
                request,
                report.stats(),
                transform_applied,
            ),
            audit,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::{Capabilities, ModelMatcher, ProfileSelector, ProviderProfile};
    use crate::types::{Message, Role, ToolSpec};

    fn endpoint(profile: ProviderProfile) -> EndpointSpec {
        EndpointSpec::new(
            Protocol::OpenAiChat,
            "http://127.0.0.1:9/v1",
            "/chat/completions",
            AuthStyle::None,
        )
        .profile(profile)
    }

    #[test]
    fn provider_capabilities_gate_tools_reasoning_and_multimodal_input() {
        let mut profile = ProviderProfile::new("restricted");
        profile.capabilities.tools.supported = false;
        profile.capabilities.reasoning.supported = false;
        profile.capabilities.multimodal = false;
        let model = EndpointAdapter::new("model-a", endpoint(profile));

        let tools = ChatRequest::user("hi").tools([ToolSpec::new(
            "lookup",
            "lookup",
            json!({"type": "object"}),
        )]);
        assert!(matches!(
            model.ensure_request_supported(&tools, false),
            Err(Error::Unsupported(message)) if message.contains("tool calling")
        ));

        let reasoning = ChatRequest::user("hi").reasoning(crate::types::ReasoningConfig::new());
        assert!(matches!(
            model.ensure_request_supported(&reasoning, false),
            Err(Error::Unsupported(message)) if message.contains("reasoning")
        ));

        let multimodal = ChatRequest::new([Message::new(
            Role::User,
            vec![
                Part::text("describe"),
                Part::image_url("https://example.test/image.png"),
            ],
        )]);
        assert!(matches!(
            model.ensure_request_supported(&multimodal, false),
            Err(Error::Unsupported(message)) if message.contains("multimodal")
        ));
    }

    #[test]
    fn model_profile_capabilities_override_provider_capabilities() {
        let mut provider = ProviderProfile::new("restricted");
        provider.capabilities.tools.supported = false;

        let model_profile = ModelProfile {
            matcher: ModelMatcher::Prefix("qwen3".to_string()),
            capabilities: Some(Capabilities::default()),
            ..ModelProfile::default()
        };
        let model =
            EndpointAdapter::new("qwen3-235b", endpoint(provider)).model_profile(model_profile);

        let request = ChatRequest::user("hi").tools([ToolSpec::new(
            "lookup",
            "lookup",
            json!({"type": "object"}),
        )]);
        assert!(model.ensure_request_supported(&request, false).is_ok());
    }

    #[test]
    fn streaming_capability_is_checked_before_sending() {
        let mut profile = ProviderProfile::new("non-streaming");
        profile.capabilities.streaming.supported = false;
        let model = EndpointAdapter::new("model-a", endpoint(profile));

        assert!(matches!(
            model.ensure_request_supported(&ChatRequest::user("hi"), true),
            Err(Error::Unsupported(message)) if message.contains("streaming")
        ));
    }

    #[test]
    fn builtin_profile_is_used_by_endpoint_adapter() {
        let endpoint = EndpointSpec::openai_chat("https://example.test/v1").profile_selector(
            ProfileSelector::Builtin(crate::profile::ProfileId::from("qwen")),
        );
        let model = EndpointAdapter::new("qwen3", endpoint);
        let request = ChatRequest::user("hi")
            .reasoning(crate::types::ReasoningConfig::new().budget_tokens(1024));

        let (body, _) = model.build_body(&request, false).unwrap();
        assert_eq!(body["enable_thinking"], true);
        assert_eq!(body["thinking_budget"], 1024);
    }

    #[test]
    fn unknown_builtin_profile_is_rejected() {
        let endpoint = EndpointSpec::openai_chat("https://example.test/v1").profile_selector(
            ProfileSelector::Builtin(crate::profile::ProfileId::from("unknown")),
        );
        let model = EndpointAdapter::new("model-a", endpoint);

        assert!(matches!(
            model.ensure_supported(),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn unsupported_non_chat_replay_policies_are_rejected() {
        let mut profile = ProviderProfile::new("anthropic-gateway");
        profile.reasoning.replay = ReasoningReplayPolicy::Never;
        let endpoint = EndpointSpec::anthropic_messages("https://example.test/v1").profile(profile);
        let model = EndpointAdapter::new("claude", endpoint);
        let request = ChatRequest::new([Message::new(
            Role::Assistant,
            vec![Part::reasoning(
                crate::types::Reasoning::text("private").with_state(
                    crate::types::ProviderStateFormat::AnthropicMessages,
                    json!({"type": "thinking"}),
                ),
            )],
        )]);

        assert!(matches!(
            model.ensure_request_supported(&request, false),
            Err(Error::Unsupported(message)) if message.contains("replay policy")
        ));
    }

    #[test]
    fn endpoint_adapter_builds_all_protocol_bodies() {
        let request = ChatRequest::user("hi");

        let (responses, _) = EndpointAdapter::new(
            "gpt-5",
            EndpointSpec::openai_responses("https://example.test/v1"),
        )
        .build_body(&request, false)
        .unwrap();
        assert_eq!(responses["model"], "gpt-5");
        assert!(responses.get("input").is_some());

        let (anthropic, _) = EndpointAdapter::new(
            "claude",
            EndpointSpec::anthropic_messages("https://example.test/v1"),
        )
        .build_body(&request, false)
        .unwrap();
        assert_eq!(anthropic["model"], "claude");
        assert!(anthropic.get("messages").is_some());

        let (gemini, _) = EndpointAdapter::new(
            "gemini-3",
            EndpointSpec::gemini_generate_content("https://example.test/v1beta"),
        )
        .build_body(&request, false)
        .unwrap();
        assert!(gemini.get("contents").is_some());
    }

    #[test]
    fn gemini_streaming_uses_stream_path_and_alt_query() {
        let endpoint = EndpointSpec::gemini_generate_content("https://example.test/v1beta");
        assert_eq!(
            endpoint.url_for("gemini-3", true).unwrap(),
            "https://example.test/v1beta/models/gemini-3:streamGenerateContent"
        );
    }
}
