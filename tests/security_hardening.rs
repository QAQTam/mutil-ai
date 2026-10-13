//! Regression tests for the 0.4 security boundaries. Local servers and fake secrets only.
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use mutil_ai::{
    AnthropicMessages, AuthStyle, ChatRequest, EndpointAdapter, EndpointSpec, Error,
    GeminiGenerateContent, HeaderInjector, Message, ModelAdapter, OpenAIChat, Part, Protocol,
    RequestContext, RequestOptions, ResponseLimits, RetryPolicy, StreamEvent,
    StreamReconnectPolicy, ToolCall, ToolResult, TransportConfig, collect_stream, next_event,
    normalize,
};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

async fn read_request(socket: &mut TcpStream) -> String {
    let mut bytes = Vec::new();
    loop {
        let mut buf = [0; 4096];
        let n = socket.read(&mut buf).await.unwrap();
        assert_ne!(n, 0);
        bytes.extend_from_slice(&buf[..n]);
        if let Some(start) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&bytes[..start]).to_ascii_lowercase();
            let len = headers
                .lines()
                .find_map(|line| {
                    line.strip_prefix("content-length: ")
                        .and_then(|v| v.parse::<usize>().ok())
                })
                .unwrap_or(0);
            if bytes.len() >= start + 4 + len {
                break;
            }
        }
    }
    String::from_utf8(bytes).unwrap()
}

async fn respond(socket: &mut TcpStream, status: &str, headers: &str, body: &str) {
    let response = format!(
        "HTTP/1.1 {status}\r\n{headers}content-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = socket.write_all(response.as_bytes()).await;
    let _ = socket.shutdown().await;
}

async fn serve_once(body: String, content_type: &str) -> (String, tokio::task::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let headers = format!("content-type: {content_type}\r\n");
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let request = read_request(&mut socket).await;
        respond(&mut socket, "200 OK", &headers, &body).await;
        request
    });
    (url, server)
}

fn chat_delta(text: &str) -> String {
    format!(
        "data: {}\n\n",
        json!({"choices":[{"delta":{"content":text}}]})
    )
}

#[tokio::test]
async fn cross_origin_307_does_not_forward_api_key_or_body() {
    let sink = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let sink_address = sink.local_addr().unwrap();
    let source = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let source_address = source.local_addr().unwrap();
    // Different host strings AND ports: localhost -> 127.0.0.1.
    let source_server = tokio::spawn(async move {
        let (mut socket, _) = source.accept().await.unwrap();
        read_request(&mut socket).await;
        respond(
            &mut socket,
            "307 Temporary Redirect",
            &format!("location: http://{sink_address}/capture\r\n"),
            "",
        )
        .await;
    });
    let adapter = AnthropicMessages::new("test")
        .base_url(format!("http://localhost:{}/v1", source_address.port()))
        .api_key("audit-fake-secret");
    let error = tokio::time::timeout(
        Duration::from_secs(5),
        adapter.complete(&ChatRequest::user("PRIVATE_PROMPT")),
    )
    .await
    .unwrap()
    .unwrap_err();
    assert_eq!(error.status(), Some(307));
    assert!(
        tokio::time::timeout(Duration::from_millis(100), sink.accept())
            .await
            .is_err()
    );
    source_server.await.unwrap();
}

#[tokio::test]
async fn gemini_transport_error_redacts_query_api_key() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        read_request(&mut socket).await;
        // Close without an HTTP response: trigger a deterministic transport error.
        let _ = socket.shutdown().await;
    });
    let adapter = GeminiGenerateContent::new("test")
        .api_key("AUDIT_FAKE_QUERY_SECRET")
        .base_url(format!("http://{address}/v1beta"));
    let error = adapter
        .complete(&ChatRequest::user("hello"))
        .await
        .unwrap_err();
    assert!(!format!("{error:?}").contains("AUDIT_FAKE_QUERY_SECRET"));
    assert!(!error.to_string().contains("AUDIT_FAKE_QUERY_SECRET"));
    server.await.unwrap();
}

#[tokio::test]
async fn provider_stream_error_redacts_logs_and_keeps_opt_in_raw() {
    let body = format!(
        "data: {}\n\n",
        json!({"error":{"message":"PRIVATE_PROMPT_ECHO"}})
    );
    let (url, server) = serve_once(body, "text/event-stream").await;
    let adapter = OpenAIChat::new("test").api_key("fake").base_url(url);
    let error = collect_stream(adapter.stream(&ChatRequest::user("hello")).await.unwrap())
        .await
        .unwrap_err();
    assert!(!error.to_string().contains("PRIVATE_PROMPT_ECHO"));
    assert!(error.raw_body().unwrap().contains("PRIVATE_PROMPT_ECHO"));
    assert!(!format!("{error:?}").contains("PRIVATE_PROMPT_ECHO"));
    server.await.unwrap();
}

#[tokio::test]
async fn request_options_cannot_replace_anthropic_auth_header_without_opt_in() {
    let (url, server) = serve_once(
        r#"{"content":[{"type":"text","text":"ok"}]}"#.to_string(),
        "application/json",
    )
    .await;
    let adapter = AnthropicMessages::new("test")
        .api_key("trusted-key")
        .base_url(url);
    let options = RequestOptions::new()
        .header("x-api-key", "untrusted-key")
        .unwrap();
    let error = adapter
        .complete_with(&ChatRequest::user("hello"), &options)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::ProtectedHeader(name) if name == "x-api-key"));
    server.abort();
}

#[test]
fn duplicate_tool_result_is_dropped() {
    let request = ChatRequest::new([
        Message::assistant_with_tools("", [ToolCall::new("lookup", json!({})).with_id("a")]),
        Message::tool_result(ToolResult::new(Some("a".into()), "lookup", "first")),
        Message::tool_result(ToolResult::new(Some("a".into()), "lookup", "replayed")),
    ]);
    let (clean, _) = normalize(&request, Protocol::OpenAiChat).unwrap();
    let results = clean
        .messages
        .iter()
        .flat_map(|m| &m.parts)
        .filter(|p| matches!(p, Part::ToolResult(_)))
        .count();
    assert_eq!(results, 1);
}

#[test]
fn explicit_wrong_call_id_is_not_matched_by_name() {
    let request = ChatRequest::new([
        Message::assistant_with_tools("", [ToolCall::new("lookup", json!({})).with_id("expected")]),
        Message::tool_result(ToolResult::new(
            Some("foreign".into()),
            "lookup",
            "foreign result",
        )),
    ]);
    let (clean, _) = normalize(&request, Protocol::OpenAiChat).unwrap();
    let results: Vec<_> = clean
        .messages
        .iter()
        .flat_map(|m| &m.parts)
        .filter_map(|p| match p {
            Part::ToolResult(r) => Some(r),
            _ => None,
        })
        .collect();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].call_id.as_deref(), Some("expected"));
    assert!(results[0].is_error);
}

#[test]
fn generated_tool_call_id_avoids_existing_id() {
    let request = ChatRequest::new([Message::assistant_with_tools(
        "",
        [
            ToolCall::new("first", json!({})).with_id("call_1"),
            ToolCall::new("second", json!({})),
        ],
    )]);
    let (clean, _) = normalize(&request, Protocol::OpenAiChat).unwrap();
    let ids: Vec<_> = clean
        .messages
        .iter()
        .flat_map(|m| &m.parts)
        .filter_map(|p| match p {
            Part::ToolCall(c) => c.id.as_deref(),
            _ => None,
        })
        .collect();
    assert_eq!(ids, vec!["call_1", "call_2"]);
}

#[tokio::test]
async fn inherited_sse_id_preserves_repeated_deltas() {
    let body = format!(
        "id: cursor\n{}{}data: [DONE]\n\n",
        chat_delta("ha"),
        chat_delta("ha")
    );
    let (url, server) = serve_once(body, "text/event-stream").await;
    let adapter = OpenAIChat::new("test").api_key("fake").base_url(url);
    let response = collect_stream(adapter.stream(&ChatRequest::user("hello")).await.unwrap())
        .await
        .unwrap();
    assert_eq!(response.text(), "haha");
    server.await.unwrap();
}

struct SlowInjector;
#[async_trait]
impl HeaderInjector for SlowInjector {
    async fn inject(&self, _: &RequestContext) -> mutil_ai::Result<HeaderMap> {
        tokio::time::sleep(Duration::from_millis(100)).await;
        Ok(HeaderMap::new())
    }
}

#[tokio::test]
async fn total_timeout_covers_header_injector() {
    let adapter = OpenAIChat::new("test")
        .api_key("fake")
        .base_url("http://127.0.0.1:9/v1")
        .transport(TransportConfig::new().header_injector(Arc::new(SlowInjector)));
    let options = RequestOptions::new().timeout(Duration::from_millis(10));
    let request = ChatRequest::user("hello");
    // Stop before the injector completes, so NO request to the public endpoint is sent.
    let result = tokio::time::timeout(
        Duration::from_millis(500),
        adapter.complete_with(&request, &options),
    )
    .await;
    assert!(matches!(result.unwrap(), Err(Error::Timeout(_))));
}

#[tokio::test]
async fn total_timeout_covers_retry_after_sleep() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        read_request(&mut socket).await;
        respond(
            &mut socket,
            "429 Too Many Requests",
            "content-type: application/json\r\nretry-after-ms: 150\r\n",
            "{}",
        )
        .await;
        assert!(
            tokio::time::timeout(Duration::from_millis(200), listener.accept())
                .await
                .is_err()
        );
    });
    let adapter = OpenAIChat::new("test")
        .api_key("fake")
        .base_url(format!("http://{address}/v1"))
        .retry_policy(RetryPolicy::default().max_attempts(2));
    let options = RequestOptions::new()
        .timeout(Duration::from_millis(100))
        .idempotency_key("audit");
    let error = adapter
        .complete_with(&ChatRequest::user("hello"), &options)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Timeout(_)));
    server.await.unwrap();
}

#[tokio::test]
async fn streaming_enforces_total_decoded_byte_limit() {
    // 40 separate valid ~60 KiB events: bounded experiment, no OOM attempt.
    let text = "x".repeat(60 * 1024);
    let body = format!("{}data: [DONE]\n\n", chat_delta(&text).repeat(40));
    let (url, server) = serve_once(body, "text/event-stream").await;
    let adapter = OpenAIChat::new("test").api_key("fake").base_url(url);
    let mut limits = ResponseLimits::default();
    limits.max_body_bytes = 1024 * 1024;
    let options = RequestOptions::new().response_limits(limits);
    let error = collect_stream(
        adapter
            .stream_with(&ChatRequest::user("hello"), &options)
            .await
            .unwrap(),
    )
    .await
    .unwrap_err();
    assert!(matches!(
        error,
        Error::ResponseLimit {
            resource: "decoded body bytes",
            ..
        }
    ));
    server.await.unwrap();
}

#[tokio::test]
async fn oversized_sse_retry_hint_refuses_reconnect() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        read_request(&mut socket).await;
        let body = format!("retry: 86400000\nid: cursor\n{}", chat_delta("partial"));
        // Truncated body triggers reconnect scheduling.
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len() + 100
        );
        let _ = socket.write_all(response.as_bytes()).await;
        let _ = socket.shutdown().await;
    });
    let adapter = OpenAIChat::new("test")
        .api_key("fake")
        .base_url(format!("http://{address}/v1"))
        .retry_policy(RetryPolicy::default().max_retry_after(Duration::from_secs(1)));
    let options =
        RequestOptions::new().stream_reconnect(StreamReconnectPolicy::new().delay(Duration::ZERO));
    let stream = adapter
        .stream_with(&ChatRequest::user("hello"), &options)
        .await
        .unwrap();
    let error = tokio::time::timeout(Duration::from_secs(2), collect_stream(stream))
        .await
        .unwrap()
        .unwrap_err();
    assert!(matches!(error, Error::Http(_)));
    server.await.unwrap();
}

#[test]
fn sensitive_custom_headers_are_hidden_in_options_debug() {
    let options = RequestOptions::new()
        .header("x-api-key", "AUDIT_FAKE_HEADER_SECRET")
        .unwrap();
    assert!(!format!("{options:?}").contains("AUDIT_FAKE_HEADER_SECRET"));
    let mut headers = HeaderMap::new();
    headers.insert(
        HeaderName::from_static("x-api-key"),
        HeaderValue::from_static("key"),
    );
    assert!(matches!(
        TransportConfig::new().apply_to(&mut headers, &options),
        Err(Error::ProtectedHeader(_))
    ));
}

#[tokio::test]
async fn custom_auth_headers_and_query_keys_are_protected() {
    let endpoint = EndpointSpec::new(
        Protocol::OpenAiChat,
        "http://127.0.0.1:9",
        "/chat/completions",
        AuthStyle::Custom {
            header: HeaderName::from_static("x-private-token"),
            value_prefix: None,
        },
    );
    let adapter = EndpointAdapter::new("test", endpoint)
        .api_key("trusted")
        .transport(TransportConfig::new());
    let options = RequestOptions::new()
        .header("x-private-token", "untrusted")
        .unwrap();
    assert!(
        matches!(adapter.complete_with(&ChatRequest::user("hello"), &options).await,
        Err(Error::ProtectedHeader(name)) if name == "x-private-token")
    );
    let endpoint = EndpointSpec::new(
        Protocol::OpenAiChat,
        "http://127.0.0.1:9",
        "/chat/completions",
        AuthStyle::QueryKey {
            parameter: "token".into(),
        },
    );
    let adapter = EndpointAdapter::new("test", endpoint).api_key("trusted");
    let options = RequestOptions::new().extra_query("token", "untrusted");
    assert!(
        matches!(adapter.complete_with(&ChatRequest::user("hello"), &options).await,
        Err(Error::ReservedExtraQuery { parameter, .. }) if parameter == "token")
    );
}

#[tokio::test]
async fn decoded_body_budget_is_shared_across_http_attempts() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        read_request(&mut socket).await;
        let body = json!({"error":{"message":"x".repeat(700)}}).to_string();
        respond(
            &mut socket,
            "429 Too Many Requests",
            "content-type: application/json\r\nretry-after: 0\r\n",
            &body,
        )
        .await;
        let (mut socket, _) = listener.accept().await.unwrap();
        read_request(&mut socket).await;
        let body = json!({"choices":[{"message":{"content":"x".repeat(700)}}]}).to_string();
        respond(
            &mut socket,
            "200 OK",
            "content-type: application/json\r\n",
            &body,
        )
        .await;
    });
    let adapter = OpenAIChat::new("test")
        .api_key("fake")
        .base_url(format!("http://{address}/v1"));
    let mut limits = ResponseLimits::default();
    limits.max_body_bytes = 1024;
    let options = RequestOptions::new()
        .response_limits(limits)
        .idempotency_key("retry-budget");
    assert!(matches!(
        adapter
            .complete_with(&ChatRequest::user("hello"), &options)
            .await,
        Err(Error::ResponseLimit {
            resource: "decoded body bytes",
            ..
        })
    ));
    server.await.unwrap();
}

#[tokio::test]
async fn gzip_body_limit_applies_after_decompression() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let compressed = include_bytes!("fixtures/security-response.json.gz");
    assert!(compressed.len() < 512);
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        read_request(&mut socket).await;
        let mut response = format!("HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-encoding: gzip\r\ncontent-length: {}\r\nconnection: close\r\n\r\n", compressed.len()).into_bytes();
        response.extend_from_slice(compressed);
        let _ = socket.write_all(&response).await;
    });
    let adapter = OpenAIChat::new("test")
        .api_key("fake")
        .base_url(format!("http://{address}/v1"));
    let mut limits = ResponseLimits::default();
    limits.max_body_bytes = 512;
    assert!(matches!(
        adapter
            .complete_with(
                &ChatRequest::user("hello"),
                &RequestOptions::new().response_limits(limits)
            )
            .await,
        Err(Error::ResponseLimit { .. })
    ));
    server.await.unwrap();
}

#[tokio::test]
async fn stream_event_budget_is_enforced() {
    let body = format!("{}data: [DONE]\n\n", chat_delta("a").repeat(3));
    let (url, server) = serve_once(body, "text/event-stream").await;
    let adapter = OpenAIChat::new("test").api_key("fake").base_url(url);
    let mut limits = ResponseLimits::default();
    limits.max_stream_events = 2;
    let stream = adapter
        .stream_with(
            &ChatRequest::user("hello"),
            &RequestOptions::new().response_limits(limits),
        )
        .await
        .unwrap();
    assert!(matches!(
        collect_stream(stream).await,
        Err(Error::ResponseLimit {
            resource: "SSE events",
            limit: 2
        })
    ));
    server.await.unwrap();
}

#[tokio::test]
async fn tool_progress_is_bounded_without_losing_arguments_and_raw_can_be_disabled() {
    let arguments = json!({"value":"x".repeat(100)}).to_string();
    let mut body = format!(
        "data: {}\n\n",
        json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call","function":{"name":"lookup","arguments":""}}]}}]})
    );
    for ch in arguments.chars() {
        body.push_str(&format!("data: {}\n\n", json!({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":ch.to_string()}}]}}]})));
    }
    body.push_str("data: [DONE]\n\n");
    let (url, server) = serve_once(body, "text/event-stream").await;
    let adapter = OpenAIChat::new("test").api_key("fake").base_url(url);
    let mut limits = ResponseLimits::default();
    limits.retain_raw = false;
    let mut stream = adapter
        .stream_with(
            &ChatRequest::user("hello"),
            &RequestOptions::new().response_limits(limits),
        )
        .await
        .unwrap();
    let mut snapshots = Vec::new();
    let mut deltas = String::new();
    let mut response = None;
    while let Some(event) = next_event(&mut stream).await {
        match event.unwrap() {
            StreamEvent::ToolCallDelta {
                arguments_delta, ..
            } => deltas.push_str(&arguments_delta),
            StreamEvent::ToolCallProgress {
                arguments_so_far, ..
            } => snapshots.push(arguments_so_far),
            StreamEvent::Done { response: done, .. } => response = Some(done),
            _ => {}
        }
    }
    assert_eq!(deltas, arguments);
    assert_eq!(snapshots.len(), 2);
    assert_eq!(snapshots.last(), Some(&arguments));
    let response = response.unwrap();
    assert!(response.raw.is_null());
    assert_eq!(
        response.tool_calls().next().unwrap().arguments,
        serde_json::from_str::<serde_json::Value>(&arguments).unwrap()
    );
    server.await.unwrap();
}

#[tokio::test]
async fn total_deadline_survives_stream_reconnect() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        read_request(&mut socket).await;
        let body = format!("id: 1\n{}", chat_delta("partial"));
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\n\r\n{body}",
            body.len() + 100
        );
        socket.write_all(response.as_bytes()).await.unwrap();
        socket.shutdown().await.unwrap();
        let (mut socket, _) = listener.accept().await.unwrap();
        read_request(&mut socket).await;
        socket.write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: 1000\r\n\r\n").await.unwrap();
        tokio::time::sleep(Duration::from_secs(1)).await;
    });
    let adapter = OpenAIChat::new("test")
        .api_key("fake")
        .base_url(format!("http://{address}/v1"));
    let options = RequestOptions::new()
        .timeout(Duration::from_millis(150))
        .stream_reconnect(
            StreamReconnectPolicy::new()
                .max_attempts(1)
                .delay(Duration::from_millis(100)),
        );
    let stream = adapter
        .stream_with(&ChatRequest::user("hello"), &options)
        .await
        .unwrap();
    assert!(matches!(
        collect_stream(stream).await,
        Err(Error::Timeout(_))
    ));
    server.abort();
}

#[test]
fn explicit_duplicate_call_ids_are_rejected() {
    let request = ChatRequest::new([Message::assistant_with_tools(
        "",
        [
            ToolCall::new("a", json!({})).with_id("same"),
            ToolCall::new("b", json!({})).with_id("same"),
        ],
    )]);
    assert!(matches!(
        normalize(&request, Protocol::OpenAiChat),
        Err(Error::Normalize(_))
    ));
}
