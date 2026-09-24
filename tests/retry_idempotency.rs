use std::time::Duration;

use mutil_ai::{ChatRequest, ErrorKind, ModelAdapter, OpenAIChat, RequestOptions, RetryPolicy};
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

#[tokio::test]
async fn retry_without_idempotency_key_sends_only_once_by_default() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        let (mut first, _) = listener.accept().await.unwrap();
        let _ = read_request(&mut first).await;
        write_response(
            &mut first,
            "429 Too Many Requests",
            "retry-after: 0\r\n",
            r#"{"error":{"message":"retry"}}"#,
        )
        .await;

        let second = tokio::time::timeout(Duration::from_millis(200), listener.accept()).await;
        assert!(
            second.is_err(),
            "request without idempotency key was replayed"
        );
    });

    let model = OpenAIChat::new("test-model")
        .api_key("test-key")
        .base_url(format!("http://{address}/v1"))
        .retry_policy(
            RetryPolicy::default()
                .max_attempts(2)
                .base_delay(Duration::ZERO),
        );

    let error = model
        .complete_with(&ChatRequest::user("hello"), &RequestOptions::new())
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::RateLimited);
    server.await.unwrap();
}

#[tokio::test]
async fn retry_without_idempotency_key_can_be_explicitly_enabled() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        let (mut first, _) = listener.accept().await.unwrap();
        let _ = read_request(&mut first).await;
        write_response(
            &mut first,
            "429 Too Many Requests",
            "retry-after: 0\r\n",
            r#"{"error":{"message":"retry"}}"#,
        )
        .await;

        let (mut second, _) = listener.accept().await.unwrap();
        let _ = read_request(&mut second).await;
        write_response(
            &mut second,
            "200 OK",
            "",
            r#"{"choices":[{"message":{"role":"assistant","content":"ok"}}]}"#,
        )
        .await;
    });

    let model = OpenAIChat::new("test-model")
        .api_key("test-key")
        .base_url(format!("http://{address}/v1"))
        .retry_policy(
            RetryPolicy::default()
                .max_attempts(2)
                .base_delay(Duration::ZERO)
                .require_idempotency_key(false),
        );

    let response = model
        .complete_with(&ChatRequest::user("hello"), &RequestOptions::new())
        .await
        .unwrap();

    assert_eq!(response.text(), "ok");
    server.await.unwrap();
}
