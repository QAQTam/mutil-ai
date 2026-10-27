use mutil_ai::{
    ChatRequest, Message, NativeToolCall, NativeToolCompletion, NativeToolKind, NativeToolPayload,
    Part, Protocol, Role, ToolResult, normalize,
};

fn call() -> NativeToolCall {
    NativeToolCall {
        kind: NativeToolKind::Custom,
        namespace: Some("functions".into()),
        name: "apply_patch".into(),
        call_id: "call_1".into(),
        item_id: Some("ctc_1".into()),
        payload: NativeToolPayload::RawText("*** Begin Patch\n+你好\n*** End Patch\n".into()),
        completion: NativeToolCompletion::Completed,
        provider_state: None,
    }
}

#[test]
fn native_calls_pair_only_by_real_call_id_and_cannot_cross_wires() {
    let assistant = Message::new(Role::Assistant, vec![Part::NativeToolCall(call())]);
    let result =
        Message::tool_results([ToolResult::new(Some("call_1".into()), "apply_patch", "ok")]);
    let request = ChatRequest::new([assistant.clone(), result]);
    let (normalized, _) = normalize(&request, Protocol::OpenAiResponses).unwrap();
    assert!(
        normalized
            .messages
            .iter()
            .any(|m| matches!(m.parts.first(), Some(Part::NativeToolCall(_))))
    );
    for protocol in [
        Protocol::OpenAiChat,
        Protocol::AnthropicMessages,
        Protocol::GeminiGenerateContent,
    ] {
        assert!(normalize(&request, protocol).is_err());
    }
    let missing_id = Message::tool_results([ToolResult::new(None, "apply_patch", "ok")]);
    assert!(
        normalize(
            &ChatRequest::new([assistant.clone(), missing_id]),
            Protocol::OpenAiResponses
        )
        .is_err()
    );
    assert!(normalize(&ChatRequest::new([assistant]), Protocol::OpenAiResponses).is_err());
}

#[test]
fn native_calls_on_user_role_are_rejected_with_explicit_options() {
    use mutil_ai::{NormalizeOptions, normalize_with_options};
    let request = ChatRequest::new([Message::new(Role::User, vec![Part::NativeToolCall(call())])]);
    assert!(
        normalize_with_options(
            &request,
            NormalizeOptions::for_protocol(Protocol::OpenAiResponses)
        )
        .is_err()
    );
}

#[test]
fn opaque_client_calls_are_rejected_even_when_a_typed_call_exists() {
    use mutil_ai::{ProviderState, ProviderStateFormat, ServerToolItem};
    use serde_json::json;

    for kind in ["function_call", "custom_tool_call", "function_call_output", "custom_tool_call_output"] {
        let raw = Part::ProviderItem(ServerToolItem::new(ProviderState::new(
            ProviderStateFormat::OpenAiResponses,
            json!({"type": kind, "call_id": "call_1", "name": "apply_patch"}),
        )));
        for parts in [vec![raw.clone()], vec![Part::NativeToolCall(call()), raw]] {
            let request = ChatRequest::new([Message::new(Role::Assistant, parts)]);
            assert!(normalize(&request, Protocol::OpenAiResponses).is_err());
        }
    }
}

#[test]
fn mixed_legacy_and_native_calls_pair_by_id_without_relabeling() {
    use mutil_ai::ToolCall;
    let legacy = ToolCall::new("apply_patch", serde_json::json!({"patch": "old"}))
        .with_id("legacy_1");
    let request = ChatRequest::new([
        Message::new(Role::Assistant, vec![Part::ToolCall(legacy), Part::NativeToolCall(call())]),
        Message::tool_results([
            ToolResult::new(Some("call_1".into()), "apply_patch", "denied"),
            ToolResult::new(Some("legacy_1".into()), "apply_patch", "ok"),
        ]),
    ]);
    let (normalized, _) = normalize(&request, Protocol::OpenAiResponses).unwrap();
    assert!(matches!(normalized.messages[0].parts[0], Part::ToolCall(_)));
    assert!(matches!(normalized.messages[0].parts[1], Part::NativeToolCall(_)));
    assert!(matches!(&normalized.messages[1].parts[0], Part::ToolResult(result) if result.call_id.as_deref() == Some("call_1") && result.content == "denied"));
    assert!(matches!(&normalized.messages[1].parts[1], Part::ToolResult(result) if result.call_id.as_deref() == Some("legacy_1") && result.content == "ok"));
}
