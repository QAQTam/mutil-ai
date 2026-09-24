use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use reqwest::header::{
    ACCEPT, AUTHORIZATION, CONTENT_LENGTH, CONTENT_TYPE, HOST, HeaderMap, HeaderName, HeaderValue,
    USER_AGENT,
};

use crate::audit::{AuditConfig, AuditContext, AuditRequestMeta, AuditSink, ProfileAuditSnapshot};
use crate::error::{Error, Result};
use crate::stream::StreamReconnectPolicy;

pub const SDK_USER_AGENT: &str = concat!("mutil-ai/", env!("CARGO_PKG_VERSION"));

const IDEMPOTENCY_KEY: &str = "idempotency-key";

/// Product identity attached to requests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientInfo {
    pub name: String,
    pub version: String,
    pub instance_id: Option<String>,
}

impl ClientInfo {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            instance_id: None,
        }
    }

    pub fn instance_id(mut self, instance_id: impl Into<String>) -> Self {
        self.instance_id = Some(instance_id.into());
        self
    }

    pub fn user_agent(&self) -> Result<HeaderValue> {
        HeaderValue::from_str(&format!(
            "{}/{} {}",
            sanitize_product_token(&self.name),
            sanitize_product_token(&self.version),
            SDK_USER_AGENT
        ))
        .map_err(Error::from)
    }
}

/// Per-request context. It is intentionally not automatically mapped to
/// provider-specific headers such as `x-sessionid`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RequestContext {
    pub request_id: Option<String>,
    pub session_id: Option<String>,
    pub conversation_id: Option<String>,
    pub tenant_id: Option<String>,
    pub trace_id: Option<String>,
    pub attributes: BTreeMap<String, String>,
}

impl RequestContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }

    pub fn session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub fn tenant_id(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    pub fn trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
    }

    pub fn attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }
}

/// Per-request transport options.
#[derive(Debug, Clone, Default)]
pub struct RequestOptions {
    pub headers: HeaderMap,
    pub user_agent: Option<HeaderValue>,
    pub context: RequestContext,
    pub idempotency_key: Option<String>,
    pub timeout: Option<Duration>,
    /// Additional query parameters appended to the endpoint URL.
    ///
    /// Duplicate names are preserved. Protocol-reserved names are validated by
    /// the adapter before the request is sent.
    pub extra_query: Vec<(String, String)>,
    /// Opt-in reconnect policy for a stream after it has started.
    pub stream_reconnect: Option<StreamReconnectPolicy>,
    /// Per-request override for transport audit switches.
    pub audit_config: Option<AuditConfig>,
}

impl RequestOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn header(mut self, name: impl AsRef<str>, value: impl AsRef<str>) -> Result<Self> {
        let name = HeaderName::from_bytes(name.as_ref().as_bytes())?;
        let value = HeaderValue::from_str(value.as_ref())?;
        ensure_ascii_header_value(name.as_str(), value.as_bytes())?;
        self.headers.insert(name, value);
        Ok(self)
    }

    /// Merge a set of request-level headers.
    ///
    /// This is an alias for callers that already build a [`HeaderMap`]. Header
    /// policy and protected-header validation are still applied when the
    /// request is sent.
    pub fn extra_headers(mut self, headers: HeaderMap) -> Self {
        self.headers.extend(headers);
        self
    }

    /// Append one query parameter without rewriting the endpoint URL.
    pub fn extra_query(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_query.push((name.into(), value.into()));
        self
    }

    pub fn stream_reconnect(mut self, policy: StreamReconnectPolicy) -> Self {
        self.stream_reconnect = Some(policy);
        self
    }

    pub fn audit_config(mut self, audit_config: AuditConfig) -> Self {
        self.audit_config = Some(audit_config);
        self
    }

    pub fn user_agent(mut self, user_agent: impl AsRef<str>) -> Result<Self> {
        let user_agent = HeaderValue::from_str(user_agent.as_ref())?;
        ensure_ascii_header_value("user-agent", user_agent.as_bytes())?;
        self.user_agent = Some(user_agent);
        Ok(self)
    }

    pub fn context(mut self, context: RequestContext) -> Self {
        self.context = context;
        self
    }

    pub fn idempotency_key(mut self, key: impl Into<String>) -> Self {
        self.idempotency_key = Some(key.into());
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    pub(crate) fn validate_query(&self, protocol: &'static str, reserved: &[&str]) -> Result<()> {
        for (name, _) in &self.extra_query {
            if reserved.contains(&name.as_str()) {
                return Err(Error::ReservedExtraQuery {
                    protocol,
                    parameter: name.clone(),
                });
            }
        }
        Ok(())
    }
}

/// Header names that an application must explicitly opt into overriding.
#[derive(Debug, Clone)]
pub struct HeaderPolicy {
    protected: Vec<HeaderName>,
    allow_override: Vec<HeaderName>,
}

impl Default for HeaderPolicy {
    fn default() -> Self {
        Self {
            protected: vec![AUTHORIZATION, CONTENT_TYPE, CONTENT_LENGTH, HOST, ACCEPT],
            allow_override: Vec::new(),
        }
    }
}

impl HeaderPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn protect(mut self, name: HeaderName) -> Self {
        self.protected.push(name);
        self
    }

    pub fn allow_override(mut self, name: HeaderName) -> Self {
        self.allow_override.push(name);
        self
    }

    pub fn is_protected(&self, name: &HeaderName) -> bool {
        self.protected.iter().any(|protected| protected == name)
    }

    pub fn can_override(&self, name: &HeaderName) -> bool {
        self.allow_override.iter().any(|allowed| allowed == name)
    }
}

/// Shared transport defaults for an adapter.
#[derive(Clone)]
pub struct TransportConfig {
    pub user_agent: HeaderValue,
    pub default_headers: HeaderMap,
    pub client_info: Option<ClientInfo>,
    pub header_policy: HeaderPolicy,
    pub header_injector: Option<Arc<dyn HeaderInjector>>,
    pub audit_sink: Option<Arc<dyn AuditSink>>,
    pub audit_config: AuditConfig,
    audit_sample_counter: Arc<AtomicU64>,
}

impl fmt::Debug for TransportConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TransportConfig")
            .field("user_agent", &self.user_agent)
            .field("default_headers", &self.default_headers)
            .field("client_info", &self.client_info)
            .field("header_policy", &self.header_policy)
            .field(
                "header_injector",
                &self.header_injector.as_ref().map(|_| "<injector>"),
            )
            .field(
                "audit_sink",
                &self.audit_sink.as_ref().map(|_| "<audit-sink>"),
            )
            .field("audit_config", &self.audit_config)
            .finish()
    }
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            user_agent: HeaderValue::from_static(SDK_USER_AGENT),
            default_headers: HeaderMap::new(),
            client_info: None,
            header_policy: HeaderPolicy::default(),
            header_injector: None,
            audit_sink: None,
            audit_config: AuditConfig::default(),
            audit_sample_counter: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl TransportConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn user_agent(mut self, user_agent: impl AsRef<str>) -> Result<Self> {
        let user_agent = HeaderValue::from_str(user_agent.as_ref())?;
        ensure_ascii_header_value("user-agent", user_agent.as_bytes())?;
        self.user_agent = user_agent;
        Ok(self)
    }

    pub fn header(mut self, name: impl AsRef<str>, value: impl AsRef<str>) -> Result<Self> {
        let name = HeaderName::from_bytes(name.as_ref().as_bytes())?;
        let value = HeaderValue::from_str(value.as_ref())?;
        ensure_ascii_header_value(name.as_str(), value.as_bytes())?;
        self.default_headers.insert(name, value);
        Ok(self)
    }

    pub fn client_info(mut self, client_info: ClientInfo) -> Self {
        self.client_info = Some(client_info);
        self
    }

    pub fn header_policy(mut self, header_policy: HeaderPolicy) -> Self {
        self.header_policy = header_policy;
        self
    }

    pub fn header_injector(mut self, header_injector: Arc<dyn HeaderInjector>) -> Self {
        self.header_injector = Some(header_injector);
        self
    }

    pub fn audit_sink(mut self, audit_sink: Arc<dyn AuditSink>) -> Self {
        self.audit_sink = Some(audit_sink);
        self
    }

    pub fn audit_config(mut self, audit_config: AuditConfig) -> Self {
        self.audit_config = audit_config;
        self
    }

    pub(crate) fn audit_context(
        &self,
        provider: &'static str,
        protocol: &'static str,
        model: &str,
        options: &RequestOptions,
        profile: Option<ProfileAuditSnapshot>,
        request_body_bytes: u64,
    ) -> Option<AuditContext> {
        let config = options.audit_config.unwrap_or(self.audit_config);
        if !config.enabled {
            return None;
        }
        let sample = self.audit_sample_counter.fetch_add(1, Ordering::Relaxed) + 1;
        if config.sample_every > 1 && !sample.is_multiple_of(config.sample_every) {
            return None;
        }
        self.audit_sink.clone().map(|sink| {
            AuditContext::new(
                sink,
                config,
                options,
                AuditRequestMeta {
                    provider,
                    protocol,
                    model,
                    profile,
                    request_body_bytes,
                },
            )
        })
    }

    /// Apply static header layers.
    ///
    /// This does not call [`Self::header_injector`]. Built-in adapters use
    /// [`Self::apply_to_async`] so dynamic headers are included.
    pub fn apply_to(&self, headers: &mut HeaderMap, options: &RequestOptions) -> Result<()> {
        headers
            .entry(USER_AGENT)
            .or_insert_with(|| self.user_agent.clone());

        if let Some(client_info) = &self.client_info {
            headers.insert(
                HeaderName::from_static("x-client-name"),
                HeaderValue::from_str(&sanitize_product_token(&client_info.name))?,
            );
            headers.insert(
                HeaderName::from_static("x-client-version"),
                HeaderValue::from_str(&sanitize_product_token(&client_info.version))?,
            );
            if let Some(instance_id) = &client_info.instance_id {
                let value = HeaderValue::from_str(instance_id)?;
                ensure_ascii_header_value("x-client-instance-id", value.as_bytes())?;
                headers.insert(HeaderName::from_static("x-client-instance-id"), value);
            }
        }

        apply_headers(headers, &self.default_headers, &self.header_policy)?;

        if let Some(user_agent) = &options.user_agent {
            headers.insert(USER_AGENT, user_agent.clone());
        }
        apply_headers(headers, &options.headers, &self.header_policy)?;

        if let Some(idempotency_key) = &options.idempotency_key {
            let value = HeaderValue::from_str(idempotency_key)?;
            ensure_ascii_header_value(IDEMPOTENCY_KEY, value.as_bytes())?;
            headers.insert(HeaderName::from_static(IDEMPOTENCY_KEY), value);
        }

        Ok(())
    }

    /// Apply static headers, then trusted dynamic headers.
    ///
    /// Dynamic headers are allowed to set protected headers such as
    /// `Authorization`; the injector is part of the trusted transport
    /// configuration.
    pub async fn apply_to_async(
        &self,
        headers: &mut HeaderMap,
        options: &RequestOptions,
    ) -> Result<()> {
        self.apply_to(headers, options)?;
        let idempotency_key = headers
            .get(HeaderName::from_static(IDEMPOTENCY_KEY))
            .cloned();
        if let Some(injector) = &self.header_injector {
            let injected = injector.inject(&options.context).await?;
            apply_trusted_headers(headers, &injected);
        }
        if let Some(idempotency_key) = idempotency_key {
            headers.insert(HeaderName::from_static(IDEMPOTENCY_KEY), idempotency_key);
        }
        Ok(())
    }
}

/// Dynamic header provider for authentication refresh and request-specific
/// routing.
#[async_trait]
pub trait HeaderInjector: Send + Sync {
    async fn inject(&self, context: &RequestContext) -> Result<HeaderMap>;
}

pub fn apply_headers(
    target: &mut HeaderMap,
    layer: &HeaderMap,
    policy: &HeaderPolicy,
) -> Result<()> {
    for (name, value) in layer {
        if policy.is_protected(name) && !policy.can_override(name) {
            return Err(Error::ProtectedHeader(name.to_string()));
        }
        target.insert(name.clone(), value.clone());
    }
    Ok(())
}

fn apply_trusted_headers(target: &mut HeaderMap, layer: &HeaderMap) {
    for (name, value) in layer {
        target.insert(name.clone(), value.clone());
    }
}

fn ensure_ascii_header_value(name: &str, value: &[u8]) -> Result<()> {
    if value.is_ascii() {
        Ok(())
    } else {
        Err(Error::NonAsciiHeaderValue {
            name: name.to_string(),
        })
    }
}

fn sanitize_product_token(value: &str) -> String {
    value
        .chars()
        .filter(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_' | '+' | '~')
        })
        .collect()
}
