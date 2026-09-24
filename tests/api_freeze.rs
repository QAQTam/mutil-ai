use mutil_ai::{
    AuditConfig, ChatRequest, EndpointSpec, Error, ExternalRole, Message, Protocol, Role, ToolCall,
    normalize,
};
use serde_json::json;

#[test]
fn frozen_normalization_contract_downgrades_roles_and_pairs_tools() {
    let request = ChatRequest::new([
        Message::developer("system rules"),
        Message::custom("planner", "legacy user input"),
        Message::assistant_with_tools("", [ToolCall::new("lookup", json!({"q": "rust"}))]),
    ]);

    let (clean, report) = normalize(&request, Protocol::OpenAiChat).unwrap();
    assert_eq!(clean.system.as_deref(), Some("system rules"));
    assert_eq!(clean.messages.len(), 3);
    assert_eq!(clean.messages[0].role, ExternalRole::User);
    assert_eq!(clean.messages[1].role, ExternalRole::Assistant);
    assert_eq!(clean.messages[2].role, ExternalRole::Tool);

    let stats = report.stats();
    assert_eq!(stats.downgraded_developer, 1);
    assert_eq!(stats.downgraded_custom_role, 1);
    assert_eq!(stats.assigned_tool_call_id, 1);
    assert_eq!(stats.synthesized_missing_tool_result, 1);
}

#[test]
fn frozen_error_contract_redacts_body_and_exposes_safe_summary() {
    let body = r#"{"error":{"message":"secret prompt","type":"invalid_request_error","code":"invalid_parameter"}}"#;
    let error = Error::Api {
        provider: "fixture",
        status: 400,
        body: body.to_string(),
        retry_after: None,
        retry_source: None,
        request_id: Some("req-1".to_string()),
    };

    assert_eq!(error.to_string(), "fixture returned HTTP 400");
    assert!(!format!("{error:?}").contains("secret prompt"));
    assert_eq!(error.body_bytes(), Some(body.len()));

    let provider = error.provider_error().unwrap();
    assert_eq!(provider.code.as_deref(), Some("invalid_parameter"));
    assert_eq!(
        provider.error_type.as_deref(),
        Some("invalid_request_error")
    );
    assert_eq!(provider.message_bytes, Some("secret prompt".len()));
}

#[test]
fn frozen_audit_defaults_do_not_record_context_identifiers() {
    let config = AuditConfig::enabled();
    assert!(!config.include_sdk_request_id);
    assert!(!config.include_session_id);
    assert!(!config.include_profile);
    assert!(config.include_provider_request_id);
    assert!(config.record_timing);
    assert!(config.record_normalization);
    assert!(config.record_cancellation);
}

#[test]
fn frozen_endpoint_contract_requires_explicit_protocol() {
    let endpoint = EndpointSpec::openai_chat("https://example.test/v1");
    assert_eq!(endpoint.protocol, Protocol::OpenAiChat);
    assert_eq!(
        endpoint.url("fixture-model").unwrap(),
        "https://example.test/v1/chat/completions"
    );

    let role = Role::Custom("agent-specific".to_string());
    assert_eq!(role, Role::Custom("agent-specific".to_string()));
}
