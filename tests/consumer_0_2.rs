use mutil_ai::{
    AuthStyle, ChatRequest, EndpointAdapter, EndpointSpec, Error, ImagePart, Message, ModelAdapter,
    OpenAIChat, Part, ProfileId, ProtocolSurface, ProviderProfile, ProviderRequestOptions,
    RequestOptions, RequestTransform, Result, ServerTool, ServerToolItem, ToolCall, ToolResult,
    ToolResultPart,
};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

async fn read_request(socket: &mut TcpStream) -> Value {
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
    serde_json::from_slice(&request[header_end..]).unwrap()
}

async fn write_json(socket: &mut TcpStream, body: &Value) {
    let body = body.to_string();
    let response = format!(
        "HTTP/1.1 200 OK\r\n\
         content-type: application/json\r\n\
         content-length: {}\r\n\
         connection: close\r\n\r\n{body}",
        body.len()
    );
    socket.write_all(response.as_bytes()).await.unwrap();
    socket.shutdown().await.unwrap();
}

#[derive(Debug)]
struct AppendUser;

impl RequestTransform for AppendUser {
    fn transform(&self, request: &ChatRequest) -> Result<ChatRequest> {
        let mut transformed = request.clone();
        transformed.messages.push(Message::user("transformed"));
        Ok(transformed)
    }
}

#[tokio::test]
async fn transform_and_typed_request_options_apply_before_send() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let body = read_request(&mut socket).await;
        assert_eq!(body["messages"][1]["content"], "transformed");
        assert_eq!(body["prompt_cache_key"], "request-key");
        assert_eq!(body["user"], "user-1");
        assert_eq!(body["do_sample"], false);
        write_json(
            &mut socket,
            &json!({"choices":[{"message":{"role":"assistant","content":"ok"}}]}),
        )
        .await;
    });

    let mut profile = ProviderProfile::new(ProfileId::from("typed-profile"));
    profile.request_options.prompt_cache_key = Some("profile-key".to_string());
    profile.request_options.do_sample = Some(false);
    let endpoint = EndpointSpec::new(
        ProtocolSurface::OpenAiChat,
        format!("http://{address}/v1"),
        "/chat/completions",
        AuthStyle::None,
    )
    .profile(profile);
    let model = EndpointAdapter::new("typed-model", endpoint);
    let options = RequestOptions::new()
        .request_transform(AppendUser)
        .provider_request(ProviderRequestOptions {
            prompt_cache_key: Some("request-key".to_string()),
            user: Some("user-1".to_string()),
            ..ProviderRequestOptions::default()
        });

    let response = model
        .complete_with(&ChatRequest::user("hello"), &options)
        .await
        .unwrap();
    assert!(response.report.transform_applied);
    server.await.unwrap();
}

#[tokio::test]
async fn structured_tool_result_reports_openai_chat_downgrade() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let body = read_request(&mut socket).await;
        assert!(
            body["messages"][1]["content"]
                .as_str()
                .is_some_and(|content| content.contains("image"))
        );
        write_json(
            &mut socket,
            &json!({"choices":[{"message":{"role":"assistant","content":"ok"}}]}),
        )
        .await;
    });

    let model = EndpointAdapter::new(
        "tool-model",
        EndpointSpec::new(
            ProtocolSurface::OpenAiChat,
            format!("http://{address}/v1"),
            "/chat/completions",
            AuthStyle::None,
        ),
    );
    let request = ChatRequest::new([
        Message::assistant_with_tools("", [ToolCall::new("lookup", json!({})).with_id("call_1")]),
        Message::tool_result(ToolResult::from_parts(
            Some("call_1".to_string()),
            "lookup",
            vec![ToolResultPart::Image {
                image: ImagePart::url("https://example.test/image.png"),
            }],
        )),
    ]);

    let response = model.complete(&request).await.unwrap();
    assert!(!response.report.wire.actions.is_empty());
    server.await.unwrap();
}

#[tokio::test]
async fn responses_server_tool_item_round_trips_opaquely() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut first, _) = listener.accept().await.unwrap();
        let first_body = read_request(&mut first).await;
        assert_eq!(first_body["tools"][0]["type"], "web_search");
        write_json(
            &mut first,
            &json!({
                "output": [{
                    "id": "ws_1",
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {"type": "search", "query": "rust"}
                }]
            }),
        )
        .await;

        let (mut second, _) = listener.accept().await.unwrap();
        let second_body = read_request(&mut second).await;
        assert_eq!(second_body["input"][0]["type"], "web_search_call");
        assert_eq!(second_body["input"][0]["id"], "ws_1");
        write_json(
            &mut second,
            &json!({"output":[{"type":"message","content":[{"type":"output_text","text":"done"}]}]}),
        )
        .await;
    });

    let mut profile = ProviderProfile::new(ProfileId::from("responses-server-tools"));
    profile.capabilities.server_side_state = true;
    let model = EndpointAdapter::new(
        "responses-model",
        EndpointSpec::new(
            ProtocolSurface::OpenAiResponses,
            format!("http://{address}/v1"),
            "/responses",
            AuthStyle::None,
        )
        .profile(profile),
    );
    let request = ChatRequest::user("search").server_tools([ServerTool::WebSearch]);
    let response = model.complete(&request).await.unwrap();
    assert!(matches!(
        response.message.parts.as_slice(),
        [Part::ProviderItem(ServerToolItem { tool: Some(tool), .. })] if tool == "web_search"
    ));

    let followup = ChatRequest::new([response.message.clone(), Message::user("continue")]);
    let response = model.complete(&followup).await.unwrap();
    assert_eq!(response.text(), "done");
    server.await.unwrap();
}

#[tokio::test]
async fn openai_chat_preserves_extended_usage() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let _ = read_request(&mut socket).await;
        write_json(
            &mut socket,
            &json!({
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "prompt_tokens_details": {"cached_tokens": 7},
                    "completion_tokens_details": {"reasoning_tokens": 3}
                }
            }),
        )
        .await;
    });

    let model = OpenAIChat::new("usage-test")
        .api_key("secret")
        .base_url(format!("http://{address}/v1"));
    let response = model.complete(&ChatRequest::user("usage")).await.unwrap();
    let usage = response.usage.unwrap();
    assert_eq!(usage.total_tokens, Some(15));
    assert_eq!(usage.cache_read_tokens, Some(7));
    assert_eq!(usage.reasoning_tokens, Some(3));
    assert!(usage.raw.is_some());
    server.await.unwrap();
}

#[derive(Debug)]
struct FailTransform;

impl RequestTransform for FailTransform {
    fn transform(&self, _request: &ChatRequest) -> Result<ChatRequest> {
        Err(Error::InvalidRequest("transform rejected".to_string()))
    }
}

#[tokio::test]
async fn failed_transform_does_not_send_request() {
    let model = EndpointAdapter::new(
        "never-send",
        EndpointSpec::new(
            ProtocolSurface::OpenAiChat,
            "http://127.0.0.1:9/v1",
            "/chat/completions",
            AuthStyle::None,
        ),
    );
    let error = model
        .complete_with(
            &ChatRequest::user("hello"),
            &RequestOptions::new().request_transform(FailTransform),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidRequest(message) if message == "transform rejected"));
}
