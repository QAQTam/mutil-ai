use mutil_ai::{
    CustomToolFormat, NativeToolCall, NativeToolCompletion, NativeToolKind, NativeToolPayload,
    ScopedToolDeclaration, ToolDeclaration, ToolSpec, validate_tool_declarations,
};
use serde_json::json;

fn function(name: &str) -> ScopedToolDeclaration {
    ToolSpec::new(name, "function", json!({"type": "object"})).into()
}

fn custom(name: &str) -> ScopedToolDeclaration {
    ScopedToolDeclaration::Custom {
        name: name.into(),
        description: "custom".into(),
        format: CustomToolFormat::Text,
    }
}

fn namespace(name: &str, tools: Vec<ScopedToolDeclaration>) -> ToolDeclaration {
    ToolDeclaration::Namespace {
        name: name.into(),
        description: "scope".into(),
        tools,
    }
}

#[test]
fn names_are_unique_per_scope_not_per_kind() {
    assert!(
        validate_tool_declarations(&[namespace(
            "functions",
            vec![function("patch"), custom("patch")],
        )])
        .is_err()
    );
    assert!(
        validate_tool_declarations(&[
            namespace("one", vec![function("patch")]),
            namespace("two", vec![custom("patch")]),
            ToolDeclaration::Tool {
                tool: custom("patch")
            },
        ])
        .is_ok()
    );
    assert!(
        validate_tool_declarations(&[
            namespace("same", vec![]),
            ToolDeclaration::Tool {
                tool: function("same")
            },
        ])
        .is_err()
    );
    assert!(validate_tool_declarations(&[ToolDeclaration::Tool { tool: custom("") }]).is_err());
}

#[test]
fn declarations_round_trip_without_changing_legacy_tool_spec() {
    let legacy = ToolSpec::new("read", "read file", json!({"type":"object"})).strict(true);
    let saved = serde_json::to_value(&legacy).unwrap();
    assert_eq!(serde_json::from_value::<ToolSpec>(saved).unwrap(), legacy);
    let declaration = namespace(
        "functions",
        vec![
            legacy.into(),
            ScopedToolDeclaration::Custom {
                name: "patch".into(),
                description: "apply".into(),
                format: CustomToolFormat::Grammar {
                    syntax: "lark".into(),
                    definition: "start: /.+/".into(),
                },
            },
        ],
    );
    let encoded = serde_json::to_value(&declaration).unwrap();
    assert_eq!(
        serde_json::from_value::<ToolDeclaration>(encoded).unwrap(),
        declaration
    );
}

#[test]
fn native_identity_completion_and_payload_are_not_inferred() {
    let mut call = NativeToolCall {
        kind: NativeToolKind::Custom,
        namespace: Some("functions".into()),
        name: "apply_patch".into(),
        call_id: "call_1".into(),
        item_id: Some("ctc_1".into()),
        payload: NativeToolPayload::RawText("中文\n\\quoted\n\n*** End Patch".into()),
        completion: NativeToolCompletion::Receiving,
        provider_state: None,
    };
    assert!(call.validate_completed().is_err());
    call.completion = NativeToolCompletion::Completed;
    assert!(call.validate_completed().is_ok());
    let json = serde_json::to_string(&call).unwrap();
    assert_eq!(serde_json::from_str::<NativeToolCall>(&json).unwrap(), call);
    call.call_id.clear();
    assert!(call.validate_completed().is_err());
    call.call_id = "call_1".into();
    call.payload = NativeToolPayload::JsonArguments("{}".into());
    assert!(call.validate_completed().is_err());
}

#[test]
fn native_requests_are_responses_only_and_detect_legacy_conflicts() {
    use mutil_ai::{ChatRequest, Protocol, normalize};
    let request = ChatRequest::user("test").tool_declarations([ToolDeclaration::Tool {
        tool: custom("apply_patch"),
    }]);
    // Responses is the only wire allowed to carry native declarations.
    let (normalized, _) = normalize(&request, Protocol::OpenAiResponses).unwrap();
    assert_eq!(normalized.tools.len(), 0);
    for protocol in [
        Protocol::OpenAiChat,
        Protocol::AnthropicMessages,
        Protocol::GeminiGenerateContent,
    ] {
        let error = normalize(&request, protocol).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unsupported_tool_representation")
        );
    }
    let conflicting = request.tools([ToolSpec::new("apply_patch", "legacy", json!({}))]);
    assert!(
        normalize(&conflicting, Protocol::OpenAiResponses)
            .unwrap_err()
            .to_string()
            .contains("tool_declaration_conflict")
    );
}
