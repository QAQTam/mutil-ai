use std::sync::{Arc, Mutex};
use std::time::Duration;

use mutil_ai::{
    AuditConfig, AuditEvent, AuditOutcome, AuditSink, ChatRequest, EndpointAdapter, EndpointSpec,
    FirstTokenKind, Message, ModelAdapter, OpenAIChat, ProviderProfile, RequestContext,
    RequestOptions, RetryPolicy, TransportConfig, bounded_audit_channel, collect_stream,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

#[derive(Default)]
struct RecordingSink {
    events: Mutex<Vec<AuditEvent>>,
}

impl AuditSink for RecordingSink {
    fn record(&self, event: AuditEvent) {
        self.events.lock().unwrap().push(event);
    }
}

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
async fn audit_records_stream_completion_without_deltas() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let _ = read_request(&mut socket).await;
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"secret stream text\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let response = format!(
            "HTTP/1.1 200 OK\r\n\
             content-type: text/event-stream\r\n\
             x-request-id: stream-request\r\n\
             content-length: {}\r\n\
             connection: close\r\n\r\n{}",
            body.len(),
            body
        );
        socket.write_all(response.as_bytes()).await.unwrap();
        socket.shutdown().await.unwrap();
    });

    let sink = Arc::new(RecordingSink::default());
    let transport = TransportConfig::new()
        .audit_sink(sink.clone())
        .audit_config(AuditConfig::enabled());
    let model = OpenAIChat::new("audit-stream")
        .api_key("secret")
        .base_url(format!("http://{address}/v1"))
        .transport(transport);

    let stream = model
        .stream(&ChatRequest::user("secret stream prompt"))
        .await
        .unwrap();
    let response = collect_stream(stream).await.unwrap();
    assert_eq!(response.text(), "secret stream text");
    server.await.unwrap();

    let events = sink.events.lock().unwrap();
    assert!(events.iter().any(|event| matches!(
        event,
        AuditEvent::ResponseHeaders {
            attempt: 0,
            status: 200,
            provider_request_id: Some(id),
            ..
        } if id == "stream-request"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AuditEvent::FirstToken {
            kind: FirstTokenKind::Text,
            ..
        }
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AuditEvent::RequestFinished {
            outcome: AuditOutcome::Success,
            status: Some(200),
            provider_request_id: Some(id),
            timing,
            bytes,
            ..
        } if id == "stream-request"
            && timing.time_to_headers.is_some()
            && timing.time_to_first_token.is_some()
            && timing.first_token_kind == Some(FirstTokenKind::Text)
            && bytes.request_body > 0
            && bytes.response_body > 0
    )));

    for event in events.iter() {
        let text = format!("{event:?}");
        assert!(!text.contains("secret stream text"));
        assert!(!text.contains("secret stream prompt"));
    }
}

#[tokio::test]
async fn audit_sampling_keeps_whole_logical_requests() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        for _ in 0..2 {
            let (mut socket, _) = listener.accept().await.unwrap();
            let _ = read_request(&mut socket).await;
            write_response(
                &mut socket,
                "200 OK",
                "",
                r#"{"choices":[{"message":{"role":"assistant","content":"ok"}}]}"#,
            )
            .await;
        }
    });

    let sink = Arc::new(RecordingSink::default());
    let transport = TransportConfig::new()
        .audit_sink(sink.clone())
        .audit_config(AuditConfig::enabled().sample_every(2));
    let model = OpenAIChat::new("audit-sample")
        .api_key("secret")
        .base_url(format!("http://{address}/v1"))
        .transport(transport);

    model.complete(&ChatRequest::user("first")).await.unwrap();
    model.complete(&ChatRequest::user("second")).await.unwrap();
    server.await.unwrap();

    let events = sink.events.lock().unwrap();
    let started = events
        .iter()
        .filter(|event| matches!(event, AuditEvent::RequestStarted { .. }))
        .count();
    assert_eq!(started, 1);
}

#[tokio::test]
async fn audit_records_early_stream_drop_as_cancelled() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let _ = read_request(&mut socket).await;
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"late\"}}]}\n\n";
        let response = format!(
            "HTTP/1.1 200 OK\r\n\
             content-type: text/event-stream\r\n\
             x-request-id: cancelled-stream\r\n\
             content-length: {}\r\n\
             connection: close\r\n\r\n{}",
            body.len(),
            body
        );
        socket.write_all(response.as_bytes()).await.unwrap();
        socket.shutdown().await.unwrap();
    });

    let sink = Arc::new(RecordingSink::default());
    let transport = TransportConfig::new()
        .audit_sink(sink.clone())
        .audit_config(AuditConfig::enabled());
    let model = OpenAIChat::new("audit-cancel")
        .api_key("secret")
        .base_url(format!("http://{address}/v1"))
        .transport(transport);

    let stream = model.stream(&ChatRequest::user("cancel")).await.unwrap();
    drop(stream);
    server.await.unwrap();

    let events = sink.events.lock().unwrap();
    assert!(events.iter().any(|event| matches!(
        event,
        AuditEvent::RequestFinished {
            outcome: AuditOutcome::Cancelled,
            status: Some(200),
            bytes,
            ..
        } if bytes.request_body > 0
    )));
}

#[tokio::test]
async fn audit_records_normalization_and_profile_without_context_names() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let _ = read_request(&mut socket).await;
        write_response(
            &mut socket,
            "200 OK",
            "",
            r#"{"choices":[{"message":{"role":"assistant","content":"ok"}}]}"#,
        )
        .await;
    });

    let sink = Arc::new(RecordingSink::default());
    let transport = TransportConfig::new()
        .audit_sink(sink.clone())
        .audit_config(AuditConfig::enabled().include_profile(true));
    let model = EndpointAdapter::new(
        "audit-profile-model",
        EndpointSpec::openai_chat(format!("http://{address}/v1"))
            .profile(ProviderProfile::new("private-tenant-profile")),
    )
    .api_key("secret")
    .transport(transport);
    let request = ChatRequest::new([
        Message::developer("secret developer prompt"),
        Message::custom("secret custom role", "secret custom prompt"),
    ]);

    model.complete(&request).await.unwrap();
    server.await.unwrap();

    let events = sink.events.lock().unwrap();
    assert!(events.iter().any(|event| matches!(
        event,
        AuditEvent::RequestStarted {
            profile: Some(profile),
            ..
        } if profile.provider_profile_id.as_deref() == Some("private-tenant-profile")
            && profile.capabilities.is_some()
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AuditEvent::Normalization { stats }
            if stats.total == 2
                && stats.downgraded_developer == 1
                && stats.downgraded_custom_role == 1
    )));

    for event in events.iter() {
        let text = format!("{event:?}");
        assert!(!text.contains("secret developer prompt"));
        assert!(!text.contains("secret custom role"));
        assert!(!text.contains("secret custom prompt"));
    }
}

#[test]
fn bounded_audit_channel_drops_without_blocking_and_tracks_stats() {
    let (sink, receiver, stats) = bounded_audit_channel(1);

    sink.record(AuditEvent::AttemptStarted { attempt: 0 });
    sink.record(AuditEvent::AttemptStarted { attempt: 1 });

    let snapshot = stats.snapshot();
    assert_eq!(snapshot.accepted, 1);
    assert_eq!(snapshot.dropped_full, 1);
    assert_eq!(snapshot.dropped(), 1);

    drop(receiver);
    sink.record(AuditEvent::AttemptStarted { attempt: 2 });
    assert_eq!(stats.snapshot().dropped_disconnected, 1);
}

#[tokio::test]
async fn audit_records_transport_outcomes_without_context_content() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        let (mut first, _) = listener.accept().await.unwrap();
        let first_request = read_request(&mut first).await;
        assert!(first_request.contains("idempotency-key: audit-idem"));
        write_response(
            &mut first,
            "429 Too Many Requests",
            "retry-after: 0\r\n",
            r#"{"error":{"message":"retry"}}"#,
        )
        .await;

        let (mut second, _) = listener.accept().await.unwrap();
        let second_request = read_request(&mut second).await;
        assert!(second_request.contains("idempotency-key: audit-idem"));
        write_response(
            &mut second,
            "200 OK",
            "x-request-id: provider-request-2\r\n",
            r#"{"choices":[{"message":{"role":"assistant","content":"ok"}}]}"#,
        )
        .await;
    });

    let sink = Arc::new(RecordingSink::default());
    let transport = TransportConfig::new()
        .audit_sink(sink.clone())
        .audit_config(AuditConfig::enabled());
    let model = OpenAIChat::new("audit-model")
        .api_key("secret")
        .base_url(format!("http://{address}/v1"))
        .retry_policy(
            RetryPolicy::default()
                .max_attempts(2)
                .base_delay(Duration::ZERO),
        )
        .transport(transport);
    let options = RequestOptions::new()
        .context(
            RequestContext::new()
                .session_id("sensitive-session")
                .request_id("sensitive-request"),
        )
        .idempotency_key("audit-idem");

    let response = model
        .complete_with(&ChatRequest::user("secret prompt"), &options)
        .await
        .unwrap();
    assert_eq!(response.text(), "ok");
    server.await.unwrap();

    let events = sink.events.lock().unwrap();
    assert!(events.iter().any(|event| matches!(
        event,
        AuditEvent::ResponseHeaders {
            attempt: 0,
            status: 429,
            ..
        }
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AuditEvent::ResponseHeaders {
            attempt: 1,
            status: 200,
            provider_request_id: Some(id),
            ..
        } if id == "provider-request-2"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AuditEvent::AttemptFinished {
            attempt: 0,
            outcome: AuditOutcome::Failure,
            status: Some(429),
            ..
        }
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AuditEvent::AttemptFinished {
            attempt: 1,
            outcome: AuditOutcome::Success,
            status: Some(200),
            provider_request_id: Some(id),
            ..
        } if id == "provider-request-2"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AuditEvent::RetryScheduled {
            attempt: 0,
            next_attempt: 1,
            ..
        }
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AuditEvent::RequestFinished {
            outcome: AuditOutcome::Success,
            status: Some(200),
            timing,
            bytes,
            ..
        } if timing.time_to_headers.is_some()
            && timing.time_to_first_token.is_none()
            && timing.first_token_kind.is_none()
            && bytes.request_body > 0
            && bytes.response_body > 0
    )));

    for event in events.iter() {
        let text = format!("{event:?}");
        assert!(!text.contains("secret prompt"));
        assert!(!text.contains("sensitive-session"));
        assert!(!text.contains("sensitive-request"));
    }
}
