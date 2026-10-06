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
use crate::cancel::CancellationToken;
use crate::error::{Error, Result};
use crate::profile::ProviderRequestOptions;
use crate::stream::StreamReconnectPolicy;
use crate::transform::RequestTransform;

/// Default `User-Agent` value, e.g. `mutil-ai/0.1.0`.
pub const SDK_USER_AGENT: &str = concat!("mutil-ai/", env!("CARGO_PKG_VERSION"));

const IDEMPOTENCY_KEY: &str = "idempotency-key";

/// Product identity attached to requests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientInfo {
    /// Application name; sanitized to a valid HTTP product token.
    pub name: String,
    /// Application version; sanitized to a valid HTTP product token.
    pub version: String,
    /// Optional per-deployment identifier, sent as `x-client-instance-id`.
    pub instance_id: Option<String>,
}

impl ClientInfo {
    /// Creates client identity from an application name and version.
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            instance_id: None,
        }
    }

    /// Sets an optional per-deployment instance identifier.
    pub fn instance_id(mut self, instance_id: impl Into<String>) -> Self {
        self.instance_id = Some(instance_id.into());
        self
    }

    /// Builds the combined `User-Agent` value:
    /// `<name>/<version> <sdk-user-agent>`.
    ///
    /// # Errors
    ///
    /// Returns an error if the resulting value is not a valid HTTP header
    /// value.
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
    /// Correlation identifier for a single logical request.
    pub request_id: Option<String>,
    /// Session identifier, kept provider-neutral (not auto-mapped to
    /// `x-sessionid`).
    pub session_id: Option<String>,
    /// Identifier grouping messages into one conversation.
    pub conversation_id: Option<String>,
    /// Multi-tenant identifier forwarded to header injectors.
    pub tenant_id: Option<String>,
    /// Distributed tracing identifier.
    pub trace_id: Option<String>,
    /// Free-form key/value pairs exposed to [`HeaderInjector::inject`].
    pub attributes: BTreeMap<String, String>,
}

impl RequestContext {
    /// Creates an empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the request correlation identifier.
    pub fn request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }

    /// Sets the session identifier.
    pub fn session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Sets the tenant identifier.
    pub fn tenant_id(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    /// Sets the distributed tracing identifier.
    pub fn trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
    }

    /// Sets one free-form attribute.
    pub fn attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }
}

/// Per-request transport options.
#[derive(Clone, Default)]
pub struct RequestOptions {
    /// Request-level headers, subject to [`HeaderPolicy`] protection checks.
    pub headers: HeaderMap,
    /// Per-request `User-Agent` override.
    pub user_agent: Option<HeaderValue>,
    /// Correlation identifiers and attributes exposed to header injectors.
    pub context: RequestContext,
    /// Adds an `idempotency-key` header and enables automatic replay of
    /// idempotent requests.
    pub idempotency_key: Option<String>,
    /// Total request timeout covering the whole exchange.
    pub timeout: Option<Duration>,
    /// Maximum time between bytes while reading an SSE stream.
    pub idle_timeout: Option<Duration>,
    /// Additional query parameters appended to the endpoint URL.
    ///
    /// Duplicate names are preserved. Protocol-reserved names are validated by
    /// the adapter before the request is sent.
    pub extra_query: Vec<(String, String)>,
    /// Opt-in reconnect policy for a stream after it has started.
    pub stream_reconnect: Option<StreamReconnectPolicy>,
    /// Cooperative cancellation shared by all attempts and stream reads.
    pub cancellation: Option<CancellationToken>,
    /// Optional transform applied once before capability gating and
    /// normalization.
    pub request_transform: Option<Arc<dyn RequestTransform>>,
    /// Typed provider request switches for this logical request.
    pub provider_request: ProviderRequestOptions,
    /// Per-request override for transport audit switches.
    pub audit_config: Option<AuditConfig>,
}

impl fmt::Debug for RequestOptions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RequestOptions")
            .field("headers", &self.headers)
            .field("user_agent", &self.user_agent)
            .field("context", &self.context)
            .field("idempotency_key", &self.idempotency_key)
            .field("timeout", &self.timeout)
            .field("idle_timeout", &self.idle_timeout)
            .field("extra_query", &self.extra_query)
            .field("stream_reconnect", &self.stream_reconnect)
            .field("cancellation", &self.cancellation)
            .field(
                "request_transform",
                &self.request_transform.as_ref().map(|_| "<configured>"),
            )
            .field("provider_request", &self.provider_request)
            .field("audit_config", &self.audit_config)
            .finish()
    }
}

impl RequestOptions {
    /// Creates empty per-request options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one request-level header, replacing any existing value with the
    /// same name.
    ///
    /// # Errors
    ///
    /// Returns an error if the name is not a valid [`HeaderName`], the value
    /// is not a valid [`HeaderValue`], or the value contains non-ASCII bytes.
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

    /// Enables opt-in reconnect for the stream after it has started.
    pub fn stream_reconnect(mut self, policy: StreamReconnectPolicy) -> Self {
        self.stream_reconnect = Some(policy);
        self
    }

    /// Attaches a cooperative cancellation token shared by all attempts and
    /// stream reads.
    pub fn cancellation(mut self, token: CancellationToken) -> Self {
        self.cancellation = Some(token);
        self
    }

    /// Alias for [`RequestOptions::cancellation`].
    pub fn cancellation_token(self, token: CancellationToken) -> Self {
        self.cancellation(token)
    }

    /// Installs a request transform applied once before capability gating and
    /// normalization.
    pub fn request_transform(mut self, transform: impl RequestTransform + 'static) -> Self {
        self.request_transform = Some(Arc::new(transform));
        self
    }

    /// Replaces the typed provider request switches for this logical request.
    pub fn provider_request(mut self, provider_request: ProviderRequestOptions) -> Self {
        self.provider_request = provider_request;
        self
    }

    /// Overrides the transport-level audit switches for this request.
    pub fn audit_config(mut self, audit_config: AuditConfig) -> Self {
        self.audit_config = Some(audit_config);
        self
    }

    /// Overrides the `User-Agent` for this request.
    ///
    /// # Errors
    ///
    /// Returns an error if the value is not a valid [`HeaderValue`] or
    /// contains non-ASCII bytes.
    pub fn user_agent(mut self, user_agent: impl AsRef<str>) -> Result<Self> {
        let user_agent = HeaderValue::from_str(user_agent.as_ref())?;
        ensure_ascii_header_value("user-agent", user_agent.as_bytes())?;
        self.user_agent = Some(user_agent);
        Ok(self)
    }

    /// Replaces the per-request context.
    pub fn context(mut self, context: RequestContext) -> Self {
        self.context = context;
        self
    }

    /// Sets the idempotency key sent as `idempotency-key`.
    pub fn idempotency_key(mut self, key: impl Into<String>) -> Self {
        self.idempotency_key = Some(key.into());
        self
    }

    /// Sets the total request timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Sets the maximum time between bytes while reading an SSE stream.
    pub fn idle_timeout(mut self, idle_timeout: Duration) -> Self {
        self.idle_timeout = Some(idle_timeout);
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
    /// Creates the default policy (standard sensitive headers protected).
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `name` to the protected set; requests may not override it unless
    /// it is also explicitly allowed via [`Self::allow_override`].
    pub fn protect(mut self, name: HeaderName) -> Self {
        self.protected.push(name);
        self
    }

    /// Explicitly allows overriding one protected header name.
    pub fn allow_override(mut self, name: HeaderName) -> Self {
        self.allow_override.push(name);
        self
    }

    /// Returns whether `name` is protected and not open for override.
    pub fn is_protected(&self, name: &HeaderName) -> bool {
        self.protected.iter().any(|protected| protected == name)
    }

    /// Returns whether `name` was explicitly allowed to override protection.
    pub fn can_override(&self, name: &HeaderName) -> bool {
        self.allow_override.iter().any(|allowed| allowed == name)
    }
}

/// Shared transport defaults for an adapter.
#[derive(Clone)]
pub struct TransportConfig {
    /// Default `User-Agent` used when a request does not set its own.
    pub user_agent: HeaderValue,
    /// Static headers applied to every request from this transport.
    pub default_headers: HeaderMap,
    /// Optional client identity reported via `x-client-*` headers.
    pub client_info: Option<ClientInfo>,
    /// Policy deciding which headers applications may override.
    pub header_policy: HeaderPolicy,
    /// Trusted dynamic header provider (e.g. token refresh), applied by
    /// [`TransportConfig::apply_to_async`].
    pub header_injector: Option<Arc<dyn HeaderInjector>>,
    /// Sink receiving audit events for this transport.
    pub audit_sink: Option<Arc<dyn AuditSink>>,
    /// Default audit switches, overridable per request.
    pub audit_config: AuditConfig,
    /// Shared sampler backing the `sample_every` audit setting.
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
    /// Creates the default transport configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the default `User-Agent`.
    ///
    /// # Errors
    ///
    /// Returns an error if the value is not a valid [`HeaderValue`] or
    /// contains non-ASCII bytes.
    pub fn user_agent(mut self, user_agent: impl AsRef<str>) -> Result<Self> {
        let user_agent = HeaderValue::from_str(user_agent.as_ref())?;
        ensure_ascii_header_value("user-agent", user_agent.as_bytes())?;
        self.user_agent = user_agent;
        Ok(self)
    }

    /// Adds one static default header applied to every request.
    ///
    /// # Errors
    ///
    /// Returns an error if the name is not a valid [`HeaderName`], the value
    /// is not a valid [`HeaderValue`], or the value contains non-ASCII bytes.
    pub fn header(mut self, name: impl AsRef<str>, value: impl AsRef<str>) -> Result<Self> {
        let name = HeaderName::from_bytes(name.as_ref().as_bytes())?;
        let value = HeaderValue::from_str(value.as_ref())?;
        ensure_ascii_header_value(name.as_str(), value.as_bytes())?;
        self.default_headers.insert(name, value);
        Ok(self)
    }

    /// Attaches client identity reported via `x-client-*` headers.
    pub fn client_info(mut self, client_info: ClientInfo) -> Self {
        self.client_info = Some(client_info);
        self
    }

    /// Replaces the header protection policy.
    pub fn header_policy(mut self, header_policy: HeaderPolicy) -> Self {
        self.header_policy = header_policy;
        self
    }

    /// Installs a trusted dynamic header provider.
    pub fn header_injector(mut self, header_injector: Arc<dyn HeaderInjector>) -> Self {
        self.header_injector = Some(header_injector);
        self
    }

    /// Installs the audit sink receiving request and stream events.
    pub fn audit_sink(mut self, audit_sink: Arc<dyn AuditSink>) -> Self {
        self.audit_sink = Some(audit_sink);
        self
    }

    /// Replaces the transport-level audit switches.
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
    /// Returns the headers to inject for this request.
    ///
    /// Injected headers are trusted: they may set protected headers such as
    /// `Authorization`.
    ///
    /// # Errors
    ///
    /// Returns an error if dynamic header computation fails (for example a
    /// token refresh could not be completed).
    async fn inject(&self, context: &RequestContext) -> Result<HeaderMap>;
}

/// Merges one header layer into `target`, enforcing the [`HeaderPolicy`].
///
/// # Errors
///
/// Returns [`Error::ProtectedHeader`] if the layer tries to set a protected
/// header that is not allowed to be overridden.
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
