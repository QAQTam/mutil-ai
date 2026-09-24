use std::sync::Arc;

use async_trait::async_trait;
use mutil_ai::{
    ClientInfo, HeaderInjector, HeaderPolicy, RequestContext, RequestOptions, SDK_USER_AGENT,
    TransportConfig, apply_headers,
};
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderName, HeaderValue, USER_AGENT};

#[test]
fn default_user_agent_is_set() {
    let transport = TransportConfig::default();
    let mut headers = HeaderMap::new();
    transport
        .apply_to(&mut headers, &RequestOptions::default())
        .unwrap();

    assert_eq!(headers.get(USER_AGENT).unwrap(), SDK_USER_AGENT);
}

#[test]
fn client_and_request_user_agents_override_default() {
    let transport = TransportConfig::new().user_agent("client/1.0.0").unwrap();
    let mut headers = HeaderMap::new();
    transport
        .apply_to(&mut headers, &RequestOptions::default())
        .unwrap();
    assert_eq!(headers.get(USER_AGENT).unwrap(), "client/1.0.0");

    let options = RequestOptions::new().user_agent("request/2.0.0").unwrap();
    transport.apply_to(&mut headers, &options).unwrap();
    assert_eq!(headers.get(USER_AGENT).unwrap(), "request/2.0.0");
}

#[test]
fn client_info_adds_product_headers() {
    let transport = TransportConfig::new()
        .client_info(ClientInfo::new("my-app", "1.2.3").instance_id("node-7"));
    let mut headers = HeaderMap::new();
    transport
        .apply_to(&mut headers, &RequestOptions::default())
        .unwrap();

    assert_eq!(headers.get("x-client-name").unwrap(), "my-app");
    assert_eq!(headers.get("x-client-version").unwrap(), "1.2.3");
    assert_eq!(headers.get("x-client-instance-id").unwrap(), "node-7");
}

#[test]
fn request_header_overrides_default_header() {
    let transport = TransportConfig::new().header("x-route", "default").unwrap();
    let options = RequestOptions::new().header("x-route", "request").unwrap();
    let mut headers = HeaderMap::new();
    transport.apply_to(&mut headers, &options).unwrap();

    assert_eq!(headers.get("x-route").unwrap(), "request");
}

#[test]
fn protected_headers_are_rejected_by_default() {
    let transport = TransportConfig::default();
    let options = RequestOptions::new()
        .header("authorization", "Bearer user-value")
        .unwrap();
    let mut headers = HeaderMap::new();
    headers.insert(
        reqwest::header::AUTHORIZATION,
        HeaderValue::from_static("Bearer sdk-value"),
    );

    let error = transport.apply_to(&mut headers, &options).unwrap_err();
    assert!(matches!(error, mutil_ai::Error::ProtectedHeader(_)));
}

#[test]
fn explicit_policy_can_allow_protected_override() {
    let policy = HeaderPolicy::new().allow_override(reqwest::header::AUTHORIZATION);
    let mut target = HeaderMap::new();
    target.insert(
        reqwest::header::AUTHORIZATION,
        HeaderValue::from_static("Bearer sdk-value"),
    );
    let mut layer = HeaderMap::new();
    layer.insert(
        reqwest::header::AUTHORIZATION,
        HeaderValue::from_static("Bearer user-value"),
    );

    apply_headers(&mut target, &layer, &policy).unwrap();
    assert_eq!(
        target.get(reqwest::header::AUTHORIZATION).unwrap(),
        "Bearer user-value"
    );
}

#[test]
fn non_ascii_header_values_are_rejected() {
    let result = RequestOptions::new().header("x-session-name", "广州");
    assert!(result.is_err());
}

#[test]
fn custom_header_names_are_validated() {
    let result = RequestOptions::new().header("bad header", "value");
    assert!(result.is_err());

    let name = HeaderName::from_static("x-sessionid");
    let value = HeaderValue::from_static("session-123");
    let mut headers = HeaderMap::new();
    headers.insert(name, value);
    assert_eq!(headers.get("x-sessionid").unwrap(), "session-123");
}

#[test]
fn idempotency_key_is_sent_as_a_header() {
    let transport = TransportConfig::default();
    let options = RequestOptions::new().idempotency_key("idem-123");
    let mut headers = HeaderMap::new();
    transport.apply_to(&mut headers, &options).unwrap();

    assert_eq!(headers.get("idempotency-key").unwrap(), "idem-123");
}

struct TestInjector;

#[async_trait]
impl HeaderInjector for TestInjector {
    async fn inject(&self, context: &RequestContext) -> mutil_ai::Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_static("Bearer dynamic-token"),
        );
        headers.insert(
            HeaderName::from_static("x-sessionid"),
            HeaderValue::from_str(context.session_id.as_deref().unwrap_or_default())?,
        );
        Ok(headers)
    }
}

#[tokio::test]
async fn injector_can_set_protected_and_context_headers() {
    let transport = TransportConfig::new().header_injector(Arc::new(TestInjector));
    let options = RequestOptions::new().context(RequestContext::new().session_id("session-9"));
    let mut headers = HeaderMap::new();
    headers.insert(AUTHORIZATION, HeaderValue::from_static("Bearer sdk-token"));

    transport
        .apply_to_async(&mut headers, &options)
        .await
        .unwrap();

    assert_eq!(headers.get(AUTHORIZATION).unwrap(), "Bearer dynamic-token");
    assert_eq!(headers.get("x-sessionid").unwrap(), "session-9");
}

struct IdempotencyInjector;

#[async_trait]
impl HeaderInjector for IdempotencyInjector {
    async fn inject(&self, _context: &RequestContext) -> mutil_ai::Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("idempotency-key"),
            HeaderValue::from_static("injected-key"),
        );
        Ok(headers)
    }
}

#[tokio::test]
async fn injector_cannot_change_request_idempotency_key() {
    let transport = TransportConfig::new().header_injector(Arc::new(IdempotencyInjector));
    let options = RequestOptions::new().idempotency_key("stable-key");
    let mut headers = HeaderMap::new();

    transport
        .apply_to_async(&mut headers, &options)
        .await
        .unwrap();

    assert_eq!(headers.get("idempotency-key").unwrap(), "stable-key");
}
