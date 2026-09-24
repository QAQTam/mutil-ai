use std::time::Duration;

use mutil_ai::{
    ChatRequest, ModelAdapter, OpenAIChat, RequestOptions, StreamEvent, StreamReconnectPolicy,
    next_event,
};
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

async fn write_sse_response(socket: &mut TcpStream, body: &str, declared_length: usize) {
    let response = format!(
        "HTTP/1.1 200 OK\r\n\
         content-type: text/event-stream\r\n\
         content-length: {declared_length}\r\n\
         connection: close\r\n\r\n{body}"
    );
    socket.write_all(response.as_bytes()).await.unwrap();
    socket.shutdown().await.unwrap();
}

#[tokio::test]
async fn stream_reconnect_requires_an_event_id_by_default() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();

    let first_event =
        "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n";

    let server = tokio::spawn(async move {
        let (mut first, _) = listener.accept().await.unwrap();
        let _ = read_request(&mut first).await;
        write_sse_response(&mut first, first_event, first_event.len() + 100).await;

        let second = tokio::time::timeout(Duration::from_millis(200), listener.accept()).await;
        assert!(second.is_err(), "reconnect should require an event id");
    });

    let model = OpenAIChat::new("test-model")
        .api_key("test-key")
        .base_url(format!("http://{address}/v1"));
    let options = RequestOptions::new().stream_reconnect(
        StreamReconnectPolicy::new()
            .max_attempts(1)
            .delay(Duration::ZERO),
    );

    let stream = model
        .stream_with(&ChatRequest::user("hello"), &options)
        .await
        .unwrap();
    let error = mutil_ai::collect_stream(stream).await.unwrap_err();
    assert!(matches!(
        error.kind(),
        mutil_ai::ErrorKind::Connection | mutil_ai::ErrorKind::Decode
    ));

    server.await.unwrap();
}

#[tokio::test]
async fn stream_reconnect_reuses_mapper_and_deduplicates_last_event() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();

    let first_event = concat!(
        "id: 1\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n\n"
    );

    let server = tokio::spawn(async move {
        let (mut first, _) = listener.accept().await.unwrap();
        let first_request = read_request(&mut first).await.to_ascii_lowercase();
        assert!(first_request.contains("x-sessionid: session-reconnect"));
        assert!(first_request.contains("idempotency-key: idem-reconnect"));
        write_sse_response(&mut first, first_event, first_event.len() + 100).await;

        let (mut second, _) = listener.accept().await.unwrap();
        let second_request = read_request(&mut second).await.to_ascii_lowercase();
        assert!(second_request.contains("last-event-id: 1"));
        assert!(second_request.contains("x-sessionid: session-reconnect"));
        assert!(second_request.contains("idempotency-key: idem-reconnect"));

        let body = concat!(
            "id: 1\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n\n",
            "id: 2\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        write_sse_response(&mut second, body, body.len()).await;
    });

    let model = OpenAIChat::new("test-model")
        .api_key("test-key")
        .base_url(format!("http://{address}/v1"));
    let options = RequestOptions::new()
        .header("x-sessionid", "session-reconnect")
        .unwrap()
        .idempotency_key("idem-reconnect")
        .stream_reconnect(
            StreamReconnectPolicy::new()
                .max_attempts(1)
                .delay(Duration::ZERO),
        );

    let mut stream = model
        .stream_with(&ChatRequest::user("hello"), &options)
        .await
        .unwrap();

    let mut saw_retry = false;
    let mut response = None;
    while let Some(event) = next_event(&mut stream).await {
        match event.unwrap() {
            StreamEvent::Retry { .. } => saw_retry = true,
            StreamEvent::Done { response: done, .. } => {
                response = Some(done);
                break;
            }
            _ => {}
        }
    }

    assert!(saw_retry);
    assert_eq!(response.unwrap().text(), "hello world");
    server.await.unwrap();
}
