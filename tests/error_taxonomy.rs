use std::time::Duration;

use mutil_ai::{Error, ErrorKind};

fn api_error(status: u16) -> Error {
    Error::Api {
        provider: "test-provider",
        status,
        body: "error".to_string(),
        retry_after: (status == 429).then_some(Duration::from_secs(7)),
        retry_source: None,
        request_id: Some(format!("req-{status}")),
    }
}

#[test]
fn api_errors_expose_stable_kind_status_provider_and_request_id() {
    let error = api_error(429);

    assert_eq!(error.kind(), ErrorKind::RateLimited);
    assert_eq!(error.kind().as_str(), "rate_limited");
    assert_eq!(error.status(), Some(429));
    assert_eq!(error.provider(), Some("test-provider"));
    assert_eq!(error.request_id(), Some("req-429"));
    assert_eq!(error.retry_after(), Some(Duration::from_secs(7)));
    assert!(error.is_retryable());
}

#[test]
fn api_error_display_and_debug_redact_provider_body() {
    let error = Error::Api {
        provider: "test-provider",
        status: 400,
        body: "secret prompt echoed by provider".to_string(),
        retry_after: None,
        retry_source: None,
        request_id: Some("req-400".to_string()),
    };

    let display = error.to_string();
    let debug = format!("{error:?}");
    assert_eq!(display, "test-provider returned HTTP 400");
    assert!(!display.contains("secret prompt"));
    assert!(!debug.contains("secret prompt"));
    assert!(debug.contains("<redacted 32 bytes>"));
    assert_eq!(error.raw_body(), Some("secret prompt echoed by provider"));
    assert_eq!(error.body_bytes(), Some(32));
}

#[test]
fn provider_error_summary_handles_common_provider_shapes_without_messages() {
    struct Case {
        body: &'static str,
        code: Option<&'static str>,
        error_type: Option<&'static str>,
        status: Option<&'static str>,
    }

    let cases = [
        Case {
            body: r#"{"error":{"message":"secret","type":"invalid_request_error","code":"context_length_exceeded"}}"#,
            code: Some("context_length_exceeded"),
            error_type: Some("invalid_request_error"),
            status: None,
        },
        Case {
            body: r#"{"type":"error","error":{"type":"rate_limit_error","message":"secret"}}"#,
            code: None,
            error_type: Some("rate_limit_error"),
            status: None,
        },
        Case {
            body: r#"{"error":{"code":400,"message":"secret","status":"INVALID_ARGUMENT","details":[{"x":1},{"x":2}]}}"#,
            code: Some("400"),
            error_type: None,
            status: Some("INVALID_ARGUMENT"),
        },
        Case {
            body: r#"{"error_code":"1001","error_msg":"secret"}"#,
            code: Some("1001"),
            error_type: None,
            status: None,
        },
    ];

    for Case {
        body,
        code,
        error_type,
        status,
    } in cases
    {
        let error = Error::Api {
            provider: "test-provider",
            status: 400,
            body: body.to_string(),
            retry_after: None,
            retry_source: None,
            request_id: None,
        };
        let info = error.provider_error().unwrap();
        assert_eq!(info.code.as_deref(), code);
        assert_eq!(info.error_type.as_deref(), error_type);
        assert_eq!(info.status.as_deref(), status);
        assert_eq!(info.message_bytes, Some("secret".len()));
        assert!(!format!("{info:?}").contains("secret"));
    }

    let gemini = Error::Api {
        provider: "gemini",
        status: 400,
        body:
            r#"{"error":{"code":400,"message":"secret","status":"INVALID_ARGUMENT","details":[1,2,3]}}"#
                .to_string(),
        retry_after: None,
        retry_source: None,
        request_id: None,
    }
    .provider_error()
    .unwrap();
    assert_eq!(gemini.details_count, 3);
}

#[test]
fn provider_error_summary_ignores_unsafe_or_oversized_tokens() {
    let error = Error::Api {
        provider: "test-provider",
        status: 400,
        body: r#"{"error":{"code":"secret prompt with spaces","type":"ok_type"}}"#.to_string(),
        retry_after: None,
        retry_source: None,
        request_id: None,
    };
    let info = error.provider_error().unwrap();
    assert_eq!(info.code, None);
    assert_eq!(info.error_type.as_deref(), Some("ok_type"));
}

#[test]
fn common_http_statuses_map_to_stable_kinds() {
    let cases = [
        (400, ErrorKind::InvalidRequest, false),
        (401, ErrorKind::Authentication, false),
        (403, ErrorKind::PermissionDenied, false),
        (404, ErrorKind::NotFound, false),
        (408, ErrorKind::Timeout, true),
        (425, ErrorKind::Overloaded, true),
        (429, ErrorKind::RateLimited, true),
        (503, ErrorKind::ProviderInternal, true),
        (599, ErrorKind::ProviderInternal, true),
    ];

    for (status, kind, retryable) in cases {
        let error = api_error(status);
        assert_eq!(error.kind(), kind, "status={status}");
        assert_eq!(error.is_retryable(), retryable, "status={status}");
    }
}

#[test]
fn structured_provider_tokens_classify_context_and_safety_errors() {
    let context = Error::Api {
        provider: "test-provider",
        status: 400,
        body: r#"{"error":{"code":"context_length_exceeded","message":"secret"}}"#.to_string(),
        retry_after: None,
        retry_source: None,
        request_id: None,
    };
    assert_eq!(context.kind(), ErrorKind::ContextLengthExceeded);
    assert!(!context.is_retryable());

    let safety = Error::Api {
        provider: "test-provider",
        status: 400,
        body: r#"{"error":{"type":"content_filter","message":"secret"}}"#.to_string(),
        retry_after: None,
        retry_source: None,
        request_id: None,
    };
    assert_eq!(safety.kind(), ErrorKind::ContentFiltered);
    assert!(!safety.is_retryable());
}

#[test]
fn cancellation_is_a_stable_non_retryable_kind() {
    let error = Error::Cancelled;
    assert_eq!(error.kind(), ErrorKind::Cancelled);
    assert_eq!(error.kind().as_str(), "cancelled");
    assert!(!error.is_retryable());
}

#[test]
fn non_api_errors_are_classified_without_string_matching() {
    let stream = Error::StreamProtocol("bad event".to_string());
    assert_eq!(stream.kind(), ErrorKind::StreamProtocol);
    assert!(!stream.is_retryable());

    let unsupported = Error::Unsupported("not supported".to_string());
    assert_eq!(unsupported.kind(), ErrorKind::Unsupported);

    let normalize = Error::Normalize("bad role".to_string());
    assert_eq!(normalize.kind(), ErrorKind::Normalization);

    let json = serde_json::from_str::<serde_json::Value>("{").unwrap_err();
    let decode = Error::Json(json);
    assert_eq!(decode.kind(), ErrorKind::Decode);
    assert!(!decode.is_retryable());
}

fn api_error_body(status: u16, body: &str) -> Error {
    Error::Api {
        provider: "test-provider",
        status,
        body: body.to_string(),
        retry_after: None,
        retry_source: None,
        request_id: None,
    }
}

#[test]
fn message_text_classifies_context_overflow_without_provider_codes() {
    let cases = [
        (
            400,
            r#"{"error":{"type":"invalid_request_error","message":"This model's maximum context length is 8192 tokens, however you requested 21000 tokens."}}"#,
        ),
        (
            400,
            r#"{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 210000 tokens > 200000 maximum"}}"#,
        ),
        (
            413,
            r#"{"error":{"message":"Request exceeds the model context window."}}"#,
        ),
        (400, "Range of input length should be [1, 128000]"),
        (
            400,
            r#"{"error":{"message":"input length and `max_tokens` exceed context limit"}}"#,
        ),
    ];

    for (status, body) in cases {
        assert_eq!(
            api_error_body(status, body).kind(),
            ErrorKind::ContextLengthExceeded,
            "status={status} body={body}"
        );
        assert!(
            !api_error_body(status, body).is_retryable(),
            "overflow is a request-shape rejection: status={status} body={body}"
        );
    }
}

#[test]
fn message_text_classifies_content_filtering() {
    let cases = [
        r#"{"error":{"type":"invalid_request_error","message":"The response was filtered due to the prompt triggering a content management policy."}}"#,
        r#"{"error":{"message":"The prompt was rejected by the content safety policy."}}"#,
        r#"{"message":"The input was blocked by the content safety policy."}"#,
    ];

    for body in cases {
        assert_eq!(
            api_error_body(400, body).kind(),
            ErrorKind::ContentFiltered,
            "body={body}"
        );
    }
}

#[test]
fn message_text_never_rewrites_a_specific_status() {
    let rate_limited = api_error_body(
        429,
        r#"{"error":{"message":"too many tokens per minute, reduce the length of your bursts"}}"#,
    );
    assert_eq!(rate_limited.kind(), ErrorKind::RateLimited);

    let overloaded = api_error_body(503, "prompt is too long for the primary replica");
    assert_eq!(overloaded.kind(), ErrorKind::ProviderInternal);
}

#[test]
fn echoed_request_content_cannot_change_the_kind() {
    // Only message-bearing fields are scanned; request content echoed under an
    // unrelated key must not classify as an overflow.
    let error = api_error_body(
        400,
        r#"{"error":{"type":"invalid_request_error","message":"value is not one of the allowed enum values","param":"tools[0].function"},"echoed_input":"please discuss the context window design"}"#,
    );
    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
}

#[test]
fn provider_stream_errors_classify_in_band_payloads() {
    let coded = Error::ProviderStream(
        r#"{"type":"response.failed","response":{"status":"failed","error":{"code":"context_length_exceeded","message":"hidden"}}}"#
            .to_string(),
    );
    assert_eq!(coded.kind(), ErrorKind::ContextLengthExceeded);

    let prose = Error::ProviderStream(
        r#"{"error":{"message":"This model's maximum context length is 4096 tokens"}}"#.to_string(),
    );
    assert_eq!(prose.kind(), ErrorKind::ContextLengthExceeded);

    let opaque = Error::ProviderStream(
        r#"{"error":{"type":"server_error","message":"boom"}}"#.to_string(),
    );
    assert_eq!(opaque.kind(), ErrorKind::ProviderInternal);
    assert!(opaque.is_retryable());
}
