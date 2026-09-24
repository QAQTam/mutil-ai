use mutil_ai::{
    AuthStyle, ChatRequest, EndpointSpec, Gemini, ModelAdapter, OpenAICompatible, ProfileId,
    ProtocolSurface, ProviderProfile, RequestOptions, TransportConfig,
};
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

async fn write_json_response(socket: &mut TcpStream, body: &str) {
    let response = format!(
        "HTTP/1.1 200 OK\r\n\
         content-type: application/json\r\n\
         content-length: {}\r\n\
         connection: close\r\n\r\n{}",
        body.len(),
        body
    );
    socket.write_all(response.as_bytes()).await.unwrap();
    socket.shutdown().await.unwrap();
}

#[tokio::test]
async fn generic_adapter_sends_explicit_endpoint_auth_headers_query_and_body_extras() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let request = read_request(&mut socket).await;
        let text = String::from_utf8_lossy(&request);
        let lower = text.to_ascii_lowercase();
        let (headers, body) = text.split_once("\r\n\r\n").unwrap();

        assert!(headers.starts_with("POST /gateway/custom/chat?tenant=profile&trace=abc HTTP/1.1"));
        assert!(lower.contains("authorization: bearer secret"));
        assert!(lower.contains("x-profile: qwen"));
        assert!(lower.contains("x-request: request-1"));

        let body: serde_json::Value = serde_json::from_str(body).unwrap();
        assert_eq!(body["model"], "qwen3");
        assert_eq!(body["enable_thinking"], true);
        assert_eq!(body["reasoning_format"], "parsed");
        assert_eq!(body["messages"][0]["content"], "你好");

        write_json_response(
            &mut socket,
            r#"{"choices":[{"message":{"role":"assistant","content":"ok"}}]}"#,
        )
        .await;
    });

    let mut profile = ProviderProfile::new(ProfileId::from("qwen"));
    profile
        .request
        .extra_headers
        .insert("x-profile", "qwen".parse().unwrap());
    profile.request.extra_body.insert(
        "enable_thinking".to_string(),
        serde_json::Value::Bool(false),
    );
    profile
        .request
        .extra_query
        .push(("tenant".to_string(), "profile".to_string()));

    let endpoint = EndpointSpec::new(
        ProtocolSurface::OpenAiChat,
        format!("http://{address}/gateway"),
        "/custom/chat",
        AuthStyle::Bearer,
    )
    .profile(profile);

    let model = OpenAICompatible::new("qwen3", endpoint)
        .api_key("secret")
        .transport(TransportConfig::new());

    let request = ChatRequest::user("你好")
        .extra_body("enable_thinking", true)
        .extra_body("reasoning_format", "parsed");
    let options = RequestOptions::new()
        .header("x-request", "request-1")
        .unwrap()
        .extra_query("trace", "abc");

    let response = model.complete_with(&request, &options).await.unwrap();
    assert_eq!(response.text(), "ok");
    server.await.unwrap();
}

#[tokio::test]
async fn gemini_rejects_extra_query_that_collides_with_its_api_key() {
    let model = Gemini::generate_content("gemini-3-flash")
        .api_key("secret")
        .base_url("http://127.0.0.1:9/v1beta");
    let options = RequestOptions::new().extra_query("key", "attacker");

    let error = model
        .complete_with(&ChatRequest::user("hi"), &options)
        .await
        .unwrap_err();

    assert!(matches!(
        error,
        mutil_ai::Error::ReservedExtraQuery { parameter, .. } if parameter == "key"
    ));
}
