use mutil_ai::{ChatRequest, ModelAdapter, OpenAIChat, collect_stream};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

#[tokio::test]
async fn openai_chat_streams_over_http_and_keeps_metadata() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = vec![0_u8; 8192];
        let read = socket.read(&mut request).await.unwrap();
        let request = String::from_utf8_lossy(&request[..read]);
        assert!(request.contains("\"stream\":true"));

        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"先想\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"你好\"},\"finish_reason\":\"stop\"}],",
            "\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}\n\n",
            "data: [DONE]\n\n"
        );
        let response = format!(
            "HTTP/1.1 200 OK\r\n\
             content-type: text/event-stream; charset=utf-8\r\n\
             x-request-id: req_integration\r\n\
             content-length: {}\r\n\
             connection: close\r\n\r\n{}",
            body.len(),
            body
        );
        socket.write_all(response.as_bytes()).await.unwrap();
        socket.shutdown().await.unwrap();
    });

    let model = OpenAIChat::new("test-model")
        .api_key("test-key")
        .base_url(format!("http://{address}/v1"));
    let stream = model.stream(&ChatRequest::user("hi")).await.unwrap();
    let response = collect_stream(stream).await.unwrap();

    assert_eq!(response.text(), "你好");
    assert_eq!(response.reasoning_text(), "先想");
    assert_eq!(
        response
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.request_id.as_deref()),
        Some("req_integration")
    );

    server.await.unwrap();
}
