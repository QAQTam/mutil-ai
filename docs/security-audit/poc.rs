//! Security audit probes, 2026-10-13.
//! These assertions demonstrate CURRENT UNSAFE BEHAVIOR, not desired behavior.
//! Run in the isolated audit harness described in the report, not the main test suite.
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use mutil_ai::{
    AnthropicMessages, ChatRequest, GeminiGenerateContent, HeaderInjector, Message, ModelAdapter,
    OpenAIChat, Part, Protocol, RequestContext, RequestOptions, RetryPolicy, StreamEvent,
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
    socket.write_all(response.as_bytes()).await.unwrap();
    socket.shutdown().await.unwrap();
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
async fn cross_origin_307_forwards_api_key_and_private_body() {
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
    let sink_server = tokio::spawn(async move {
        let (mut socket, _) = sink.accept().await.unwrap();
        let request = read_request(&mut socket).await;
        respond(
            &mut socket,
            "200 OK",
            "content-type: application/json\r\n",
            r#"{"content":[{"type":"text","text":"ok"}]}"#,
        )
        .await;
        request
    });
    let adapter = AnthropicMessages::new("test")
        .base_url(format!("http://localhost:{}/v1", source_address.port()))
        .api_key("audit-fake-secret");
    tokio::time::timeout(
        Duration::from_secs(5),
        adapter.complete(&ChatRequest::user("PRIVATE_PROMPT")),
    )
    .await
    .unwrap()
    .unwrap();
    let captured = sink_server.await.unwrap();
    assert!(captured.contains("x-api-key: audit-fake-secret"));
    assert!(captured.contains("PRIVATE_PROMPT"));
    assert!(captured.starts_with("POST /capture"));
    source_server.await.unwrap();
}

#[tokio::test]
async fn gemini_transport_error_debug_contains_query_api_key() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        read_request(&mut socket).await;
        // Close without an HTTP response: trigger a deterministic transport error.
        socket.shutdown().await.unwrap();
    });
    let adapter = GeminiGenerateContent::new("test")
        .api_key("AUDIT_FAKE_QUERY_SECRET")
        .base_url(format!("http://{address}/v1beta"));
    let error = adapter
        .complete(&ChatRequest::user("hello"))
        .await
        .unwrap_err();
    assert!(format!("{error:?}").contains("AUDIT_FAKE_QUERY_SECRET"));
    assert!(error.to_string().contains("AUDIT_FAKE_QUERY_SECRET"));
    server.await.unwrap();
}

#[tokio::test]
async fn provider_stream_error_display_and_debug_leak_raw_message() {
    let body = format!(
        "data: {}\n\n",
        json!({"error":{"message":"PRIVATE_PROMPT_ECHO"}})
    );
    let (url, server) = serve_once(body, "text/event-stream").await;
    let adapter = OpenAIChat::new("test").api_key("fake").base_url(url);
    let error = collect_stream(adapter.stream(&ChatRequest::user("hello")).await.unwrap())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("PRIVATE_PROMPT_ECHO"));
    assert!(format!("{error:?}").contains("PRIVATE_PROMPT_ECHO"));
    server.await.unwrap();
}

#[tokio::test]
async fn request_options_can_replace_anthropic_auth_header_without_opt_in() {
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
    adapter
        .complete_with(&ChatRequest::user("hello"), &options)
        .await
        .unwrap();
    let captured = server.await.unwrap();
    assert!(captured.contains("x-api-key: untrusted-key"));
    assert!(!captured.contains("x-api-key: trusted-key\r\n"));
}

#[test]
fn duplicate_tool_result_is_accepted_for_already_fulfilled_call() {
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
    assert_eq!(results, 2);
}

#[test]
fn explicit_wrong_call_id_falls_back_to_name_but_keeps_wrong_id() {
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
    assert_eq!(results[0].call_id.as_deref(), Some("foreign"));
    assert!(!results[0].is_error);
}

#[test]
fn generated_tool_call_id_collides_with_existing_id() {
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
    assert_eq!(ids, vec!["call_1", "call_1"]);
}

#[tokio::test]
async fn inherited_sse_id_drops_legitimate_repeated_deltas_without_reconnect() {
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
    // Correct response is "haha", but the second event inherits id: cursor and is discarded.
    assert_eq!(response.text(), "ha");
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
async fn total_timeout_does_not_cover_header_injector() {
    let adapter = OpenAIChat::new("test")
        .api_key("fake")
        .transport(TransportConfig::new().header_injector(Arc::new(SlowInjector)));
    let options = RequestOptions::new().timeout(Duration::from_millis(10));
    let request = ChatRequest::user("hello");
    // Stop before the injector completes, so NO request to the public endpoint is sent.
    let result = tokio::time::timeout(
        Duration::from_millis(50),
        adapter.complete_with(&request, &options),
    )
    .await;
    assert!(
        result.is_err(),
        "SDK timeout should already have returned an error at 10 ms"
    );
}

#[tokio::test]
async fn total_timeout_does_not_cover_retry_after_sleep() {
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
        let (mut socket, _) = listener.accept().await.unwrap();
        read_request(&mut socket).await;
        respond(
            &mut socket,
            "200 OK",
            "content-type: application/json\r\n",
            r#"{"choices":[{"message":{"content":"ok"}}]}"#,
        )
        .await;
    });
    let adapter = OpenAIChat::new("test")
        .api_key("fake")
        .base_url(format!("http://{address}/v1"))
        .retry_policy(RetryPolicy::default().max_attempts(2));
    let options = RequestOptions::new()
        .timeout(Duration::from_millis(100))
        .idempotency_key("audit");
    let started = Instant::now();
    let response = adapter
        .complete_with(&ChatRequest::user("hello"), &options)
        .await
        .unwrap();
    assert_eq!(response.text(), "ok");
    assert!(started.elapsed() >= Duration::from_millis(150));
    server.await.unwrap();
}

#[tokio::test]
async fn streaming_accumulates_more_than_parser_event_limit() {
    // 40 separate valid ~60 KiB events: bounded experiment, no OOM attempt.
    let text = "x".repeat(60 * 1024);
    let body = format!("{}data: [DONE]\n\n", chat_delta(&text).repeat(40));
    let (url, server) = serve_once(body, "text/event-stream").await;
    let adapter = OpenAIChat::new("test").api_key("fake").base_url(url);
    let response = collect_stream(adapter.stream(&ChatRequest::user("hello")).await.unwrap())
        .await
        .unwrap();
    assert_eq!(response.text().len(), 40 * 60 * 1024);
    assert_eq!(response.raw.as_array().unwrap().len(), 40);
    server.await.unwrap();
}

#[tokio::test]
async fn reconnect_retry_hint_bypasses_http_retry_after_cap() {
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
        socket.write_all(response.as_bytes()).await.unwrap();
        socket.shutdown().await.unwrap();
    });
    let adapter = OpenAIChat::new("test")
        .api_key("fake")
        .base_url(format!("http://{address}/v1"))
        .retry_policy(RetryPolicy::default().max_retry_after(Duration::from_secs(1)));
    let options =
        RequestOptions::new().stream_reconnect(StreamReconnectPolicy::new().delay(Duration::ZERO));
    let mut stream = adapter
        .stream_with(&ChatRequest::user("hello"), &options)
        .await
        .unwrap();
    let scheduled = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if let Some(Ok(StreamEvent::Retrying { delay, .. })) = next_event(&mut stream).await {
                break delay;
            }
        }
    })
    .await
    .unwrap();
    assert_eq!(scheduled, Duration::from_secs(86400));
    // Drop immediately; do not wait for the requested day-long delay.
    drop(stream);
    server.await.unwrap();
}

#[test]
fn sensitive_custom_headers_are_visible_in_options_debug() {
    let options = RequestOptions::new()
        .header("x-api-key", "AUDIT_FAKE_HEADER_SECRET")
        .unwrap();
    assert!(format!("{options:?}").contains("AUDIT_FAKE_HEADER_SECRET"));
    let mut headers = HeaderMap::new();
    headers.insert(
        HeaderName::from_static("x-api-key"),
        HeaderValue::from_static("key"),
    );
    TransportConfig::new()
        .apply_to(&mut headers, &options)
        .unwrap();
}
