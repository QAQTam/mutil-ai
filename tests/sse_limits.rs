use mutil_ai::{ChatRequest, ModelAdapter, OpenAIResponses, RequestOptions, collect_stream};
use tokio::{io::{AsyncReadExt, AsyncWriteExt}, net::TcpListener};

async fn request_with_limits(line: usize, event: usize, body_limit: usize) -> mutil_ai::Result<mutil_ai::ChatResponse> {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buffer = [0;8192];
        socket.read(&mut buffer).await.unwrap();
        let text = "x".repeat(128 * 1024);
        let item = serde_json::json!({"type":"response.completed","response":{"id":"r1","status":"completed","model":"test","output":[{"id":"m1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":text,"annotations":[]}]}]}});
        let body = format!("data: {item}\n\n");
        let response = format!("HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",body.len());
        let _ = socket.write_all(response.as_bytes()).await;
    });
    let model = OpenAIResponses::new("test").api_key("test").base_url(format!("http://{address}"));
    let mut options = RequestOptions::default();
    options.response_limits.max_sse_line_bytes = line;
    options.response_limits.max_sse_event_bytes = event;
    options.response_limits.max_body_bytes = body_limit;
    let result = match model.stream_with(&ChatRequest::user("inspect"), &options).await {
        Ok(stream) => collect_stream(stream).await,
        Err(error) => Err(error),
    };
    server.await.unwrap();
    result
}

#[tokio::test]
async fn configured_sse_limits_apply_without_disabling_total_body_budget() {
    assert_eq!(request_with_limits(256*1024, 512*1024, 1024*1024).await.unwrap().text().len(),128*1024);
    assert!(request_with_limits(64*1024, 512*1024, 1024*1024).await.is_err());
    assert!(request_with_limits(256*1024, 64*1024, 1024*1024).await.is_err());
    assert!(request_with_limits(256*1024, 512*1024, 64*1024).await.is_err());
}
