use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use mutil_ai::{
    ChatRequest, HeaderInjector, ModelAdapter, OpenAIChat, RequestContext, RequestOptions,
    RetryPolicy, TransportConfig, collect_stream,
};
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

async fn read_request(socket: &mut TcpStream) -> String {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 4096];

    loop {
        let read = socket.read(&mut buffer).await.unwrap();
        if read == 0 {
            break;
        }
        request.extend_from_slice(&buffer[..read]);
        if request.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
    }

    String::from_utf8_lossy(&request).to_string()
}

async fn write_response(socket: &mut TcpStream, status: &str, headers: &str, body: &str) {
    let response = format!(
        "HTTP/1.1 {status}\r\n\
         content-type: application/json\r\n\
         {headers}\
         content-length: {}\r\n\
         connection: close\r\n\r\n{body}",
        body.len()
    );
    socket.write_all(response.as_bytes()).await.unwrap();
    socket.shutdown().await.unwrap();
}

struct RefreshingInjector {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl HeaderInjector for RefreshingInjector {
    async fn inject(&self, context: &RequestContext) -> mutil_ai::Result<HeaderMap> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer token-{call}"))?,
        );
        headers.insert(
            reqwest::header::HeaderName::from_static("x-sessionid"),
            HeaderValue::from_str(context.session_id.as_deref().unwrap_or_default())?,
        );
        Ok(headers)
    }
}

fn retry_policy() -> RetryPolicy {
    RetryPolicy::default()
        .max_attempts(2)
        .base_delay(Duration::ZERO)
}

fn request_options() -> RequestOptions {
    RequestOptions::new()
        .context(RequestContext::new().session_id("session-refresh"))
        .idempotency_key("idem-refresh")
}

#[tokio::test]
async fn json_retry_refreshes_dynamic_headers_but_keeps_session_and_idempotency() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let calls = Arc::new(AtomicUsize::new(0));

    let server = tokio::spawn(async move {
        let (mut first, _) = listener.accept().await.unwrap();
        let first_request = read_request(&mut first).await.to_ascii_lowercase();
        assert!(first_request.contains("authorization: bearer token-1"));
        assert!(first_request.contains("x-sessionid: session-refresh"));
        assert!(first_request.contains("idempotency-key: idem-refresh"));
        write_response(
            &mut first,
            "429 Too Many Requests",
            "retry-after: 0\r\n",
            r#"{"error":{"message":"retry"}}"#,
        )
        .await;

        let (mut second, _) = listener.accept().await.unwrap();
        let second_request = read_request(&mut second).await.to_ascii_lowercase();
        assert!(second_request.contains("authorization: bearer token-2"));
        assert!(second_request.contains("x-sessionid: session-refresh"));
        assert!(second_request.contains("idempotency-key: idem-refresh"));
        write_response(
            &mut second,
            "200 OK",
            "",
            r#"{"choices":[{"message":{"role":"assistant","content":"ok"}}]}"#,
        )
        .await;
    });

    let transport = TransportConfig::new().header_injector(Arc::new(RefreshingInjector {
        calls: calls.clone(),
    }));
    let model = OpenAIChat::new("test-model")
        .api_key("static-key")
        .base_url(format!("http://{address}/v1"))
        .retry_policy(retry_policy())
        .transport(transport);

    let response = model
        .complete_with(&ChatRequest::user("hello"), &request_options())
        .await
        .unwrap();

    assert_eq!(response.text(), "ok");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    server.await.unwrap();
}

#[tokio::test]
async fn streaming_retry_refreshes_dynamic_headers_before_stream_starts() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let calls = Arc::new(AtomicUsize::new(0));

    let server = tokio::spawn(async move {
        let (mut first, _) = listener.accept().await.unwrap();
        let first_request = read_request(&mut first).await.to_ascii_lowercase();
        assert!(first_request.contains("authorization: bearer token-1"));
        write_response(
            &mut first,
            "429 Too Many Requests",
            "retry-after: 0\r\n",
            r#"{"error":{"message":"retry"}}"#,
        )
        .await;

        let (mut second, _) = listener.accept().await.unwrap();
        let second_request = read_request(&mut second).await.to_ascii_lowercase();
        assert!(second_request.contains("authorization: bearer token-2"));
        assert!(second_request.contains("x-sessionid: session-refresh"));
        assert!(second_request.contains("idempotency-key: idem-refresh"));

        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let response = format!(
            "HTTP/1.1 200 OK\r\n\
             content-type: text/event-stream\r\n\
             content-length: {}\r\n\
             connection: close\r\n\r\n{}",
            body.len(),
            body
        );
        second.write_all(response.as_bytes()).await.unwrap();
        second.shutdown().await.unwrap();
    });

    let transport = TransportConfig::new().header_injector(Arc::new(RefreshingInjector {
        calls: calls.clone(),
    }));
    let model = OpenAIChat::new("test-model")
        .api_key("static-key")
        .base_url(format!("http://{address}/v1"))
        .retry_policy(retry_policy())
        .transport(transport);

    let stream = model
        .stream_with(&ChatRequest::user("hello"), &request_options())
        .await
        .unwrap();
    let response = collect_stream(stream).await.unwrap();

    assert_eq!(response.text(), "ok");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    server.await.unwrap();
}
