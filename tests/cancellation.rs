use std::time::Duration;

use mutil_ai::{
    CancellationToken, ChatRequest, ErrorKind, ModelAdapter, OpenAIChat, RequestOptions,
    RetryPolicy, StreamEvent, next_event,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

async fn read_request(socket: &mut TcpStream) -> String {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let read = socket.read(&mut buffer).await.unwrap();
        assert_ne!(read, 0, "connection closed before request headers");
        request.extend_from_slice(&buffer[..read]);
        if request.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
    }
    String::from_utf8_lossy(&request).to_string()
}

#[tokio::test]
async fn complete_cancellation_returns_structured_error() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (release, wait) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let _ = read_request(&mut socket).await;
        let _ = wait.await;
    });

    let token = CancellationToken::new();
    let cancel = token.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancel.cancel();
    });
    let model = OpenAIChat::new("cancel-test")
        .api_key("secret")
        .base_url(format!("http://{address}/v1"));

    let error = tokio::time::timeout(
        Duration::from_secs(1),
        model.complete_with(
            &ChatRequest::user("cancel"),
            &RequestOptions::new().cancellation(token),
        ),
    )
    .await
    .expect("cancellation must finish promptly")
    .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Cancelled);
    let _ = release.send(());
    server.await.unwrap();
}

#[tokio::test]
async fn stream_cancellation_stops_reading_without_reconnect() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (release, wait) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let _ = read_request(&mut socket).await;
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\n\
                  content-type: text/event-stream\r\n\
                  connection: close\r\n\r\n\
                  data: {\"choices\":[{\"delta\":{\"content\":\"one\"},\"finish_reason\":null}]}\n\n",
            )
            .await
            .unwrap();
        let _ = wait.await;
        let second = tokio::time::timeout(Duration::from_millis(150), listener.accept()).await;
        assert!(second.is_err(), "cancelled stream must not reconnect");
    });

    let token = CancellationToken::new();
    let model = OpenAIChat::new("cancel-stream")
        .api_key("secret")
        .base_url(format!("http://{address}/v1"));
    let mut stream = model
        .stream_with(
            &ChatRequest::user("cancel"),
            &RequestOptions::new()
                .cancellation(token.clone())
                .stream_reconnect(mutil_ai::StreamReconnectPolicy::new().delay(Duration::ZERO)),
        )
        .await
        .unwrap();

    loop {
        match next_event(&mut stream).await.unwrap().unwrap() {
            StreamEvent::TextDelta { .. } => break,
            StreamEvent::Done { .. } => panic!("stream ended before cancellation"),
            _ => {}
        }
    }
    token.cancel();
    let error = tokio::time::timeout(Duration::from_secs(1), next_event(&mut stream))
        .await
        .expect("cancelled stream must finish promptly")
        .unwrap()
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Cancelled);

    let _ = release.send(());
    server.await.unwrap();
}

#[tokio::test]
async fn stream_idle_timeout_returns_timeout() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (release, wait) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let _ = read_request(&mut socket).await;
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\n\
                  content-type: text/event-stream\r\n\
                  connection: close\r\n\r\n",
            )
            .await
            .unwrap();
        let _ = wait.await;
    });

    let model = OpenAIChat::new("idle-test")
        .api_key("secret")
        .base_url(format!("http://{address}/v1"));
    let mut stream = model
        .stream_with(
            &ChatRequest::user("idle"),
            &RequestOptions::new().idle_timeout(Duration::from_millis(50)),
        )
        .await
        .unwrap();
    let error = tokio::time::timeout(Duration::from_secs(1), next_event(&mut stream))
        .await
        .expect("idle timeout must fire promptly")
        .unwrap()
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Timeout);

    let _ = release.send(());
    server.await.unwrap();
}

#[tokio::test]
async fn cancellation_interrupts_retry_delay_and_does_not_send_again() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let _ = read_request(&mut socket).await;
        let body = r#"{"error":{"code":"rate_limit_error","message":"retry"}}"#;
        let response = format!(
            "HTTP/1.1 429 Too Many Requests\r\n\
             content-type: application/json\r\n\
             retry-after: 5\r\n\
             content-length: {}\r\n\
             connection: close\r\n\r\n{body}",
            body.len()
        );
        socket.write_all(response.as_bytes()).await.unwrap();
        socket.shutdown().await.unwrap();

        let second = tokio::time::timeout(Duration::from_millis(200), listener.accept()).await;
        assert!(second.is_err(), "cancelled retry must not send again");
    });

    let token = CancellationToken::new();
    let cancel = token.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancel.cancel();
    });
    let model = OpenAIChat::new("cancel-retry")
        .api_key("secret")
        .base_url(format!("http://{address}/v1"))
        .retry_policy(
            RetryPolicy::new(3)
                .base_delay(Duration::ZERO)
                .jitter_ratio(0.0)
                .max_retry_after(Duration::from_secs(10)),
        );

    let error = tokio::time::timeout(
        Duration::from_secs(1),
        model.complete_with(
            &ChatRequest::user("cancel"),
            &RequestOptions::new()
                .idempotency_key("cancel-retry")
                .cancellation(token),
        ),
    )
    .await
    .expect("cancelled retry must finish promptly")
    .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Cancelled);
    server.await.unwrap();
}
