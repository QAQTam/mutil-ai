use mutil_ai::{
    ChatRequest, EndpointAdapter, EndpointSpec, ModelAdapter, ProviderProfile, collect_stream,
};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

async fn read_request(socket: &mut TcpStream) -> Vec<u8> {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 4096];
    let header_end = loop {
        let read = socket.read(&mut buffer).await.unwrap();
        assert_ne!(read, 0, "connection closed before request headers");
        request.extend_from_slice(&buffer[..read]);

        if let Some(index) = request.windows(4).position(|window| window == b"\r\n\r\n") {
            break index + 4;
        }
    };

    let headers = String::from_utf8_lossy(&request[..header_end]);
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or_default();

    while request.len() < header_end + content_length {
        let read = socket.read(&mut buffer).await.unwrap();
        assert_ne!(read, 0, "connection closed before request body");
        request.extend_from_slice(&buffer[..read]);
    }

    request
}

async fn write_response(socket: &mut TcpStream, content_type: &str, body: &str) {
    let response = format!(
        "HTTP/1.1 200 OK\r\n\
         content-type: {content_type}\r\n\
         content-length: {}\r\n\
         connection: close\r\n\r\n{}",
        body.len(),
        body
    );
    socket.write_all(response.as_bytes()).await.unwrap();
    socket.shutdown().await.unwrap();
}

fn request_parts(request: &[u8]) -> (String, String) {
    let text = String::from_utf8_lossy(request);
    let (headers, body) = text.split_once("\r\n\r\n").unwrap();
    (headers.to_string(), body.to_string())
}

#[tokio::test]
async fn endpoint_adapter_dispatches_openai_responses() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let request = read_request(&mut socket).await;
        let (headers, body) = request_parts(&request);
        assert!(headers.starts_with("POST /v1/responses HTTP/1.1"));
        assert!(
            headers
                .to_ascii_lowercase()
                .contains("authorization: bearer secret")
        );

        let body: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(body["model"], "gpt-5");
        assert_eq!(body["store"], true);
        assert_eq!(body["metadata"]["trace"], "req-1");
        assert_eq!(body["input"][0]["role"], "user");

        write_response(
            &mut socket,
            "application/json",
            r#"{"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"responses-ok"}]}]}"#,
        )
        .await;
    });

    let mut profile = ProviderProfile::new("responses-gateway");
    profile
        .request
        .extra_body
        .insert("store".to_string(), json!(true));
    let endpoint = EndpointSpec::openai_responses(format!("http://{address}/v1")).profile(profile);
    let model = EndpointAdapter::new("gpt-5", endpoint).api_key("secret");
    let request = ChatRequest::user("hi").extra_body("metadata", json!({"trace": "req-1"}));

    let response = model.complete(&request).await.unwrap();
    assert_eq!(response.text(), "responses-ok");
    server.await.unwrap();
}

#[tokio::test]
async fn endpoint_adapter_dispatches_anthropic_messages() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let request = read_request(&mut socket).await;
        let (headers, body) = request_parts(&request);
        let lower = headers.to_ascii_lowercase();
        assert!(headers.starts_with("POST /v1/messages HTTP/1.1"));
        assert!(lower.contains("x-api-key: secret"));
        assert!(lower.contains("anthropic-version: 2023-06-01"));

        let body: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(body["model"], "claude-sonnet");
        assert_eq!(body["max_tokens"], 4096);
        assert_eq!(body["messages"][0]["role"], "user");

        write_response(
            &mut socket,
            "application/json",
            r#"{"content":[{"type":"text","text":"anthropic-ok"}]}"#,
        )
        .await;
    });

    let endpoint = EndpointSpec::anthropic_messages(format!("http://{address}/v1"));
    let model = EndpointAdapter::new("claude-sonnet", endpoint).api_key("secret");

    let response = model.complete(&ChatRequest::user("hi")).await.unwrap();
    assert_eq!(response.text(), "anthropic-ok");
    server.await.unwrap();
}

#[tokio::test]
async fn endpoint_adapter_dispatches_gemini_generate_content() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let request = read_request(&mut socket).await;
        let (headers, body) = request_parts(&request);
        assert!(
            headers.starts_with("POST /v1beta/models/gemini-3:generateContent?key=secret HTTP/1.1")
        );

        let body: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(body["contents"][0]["role"], "user");
        assert_eq!(body["contents"][0]["parts"][0]["text"], "hi");

        write_response(
            &mut socket,
            "application/json",
            r#"{"candidates":[{"content":{"parts":[{"text":"gemini-ok"}]}}]}"#,
        )
        .await;
    });

    let endpoint = EndpointSpec::gemini_generate_content(format!("http://{address}/v1beta"));
    let model = EndpointAdapter::new("gemini-3", endpoint).api_key("secret");

    let response = model.complete(&ChatRequest::user("hi")).await.unwrap();
    assert_eq!(response.text(), "gemini-ok");
    server.await.unwrap();
}

#[tokio::test]
async fn endpoint_adapter_dispatches_gemini_streaming() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let request = read_request(&mut socket).await;
        let (headers, _) = request_parts(&request);
        assert!(headers.starts_with(
            "POST /v1beta/models/gemini-3:streamGenerateContent?key=secret&alt=sse HTTP/1.1"
        ));

        let body = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"你好\"}]},\"finishReason\":\"STOP\"}]}\n\n";
        write_response(&mut socket, "text/event-stream", body).await;
    });

    let endpoint = EndpointSpec::gemini_generate_content(format!("http://{address}/v1beta"));
    let model = EndpointAdapter::new("gemini-3", endpoint).api_key("secret");
    let stream = model.stream(&ChatRequest::user("hi")).await.unwrap();
    let response = collect_stream(stream).await.unwrap();

    assert_eq!(response.text(), "你好");
    server.await.unwrap();
}
