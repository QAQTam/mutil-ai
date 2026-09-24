use std::time::Duration;

use mutil_ai::{ChatRequest, ModelAdapter, OpenAIChat, RequestOptions, RetryPolicy};
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
async fn retries_reuse_the_same_session_and_idempotency_key() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        let (mut first, _) = listener.accept().await.unwrap();
        let first_request = read_request(&mut first).await;
        assert!(first_request.contains("x-sessionid: session-A"));
        assert!(first_request.contains("idempotency-key: idem-A"));
        write_response(
            &mut first,
            "429 Too Many Requests",
            "retry-after: 0\r\n",
            r#"{"error":{"message":"retry"}}"#,
        )
        .await;

        let (mut second, _) = listener.accept().await.unwrap();
        let second_request = read_request(&mut second).await;
        assert!(second_request.contains("x-sessionid: session-A"));
        assert!(second_request.contains("idempotency-key: idem-A"));
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
                .base_delay(Duration::ZERO),
        );
    let options = RequestOptions::new()
        .header("x-sessionid", "session-A")
        .unwrap()
        .idempotency_key("idem-A");

    let response = model
        .complete_with(&ChatRequest::user("hello"), &options)
        .await
        .unwrap();

    assert_eq!(response.text(), "ok");
    server.await.unwrap();
}
