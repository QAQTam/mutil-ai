use mutil_ai::{
    ChatRequest, ExternalRole, Message, MissingToolResultPolicy, NormalizeAction, NormalizeOptions,
    OrphanToolResultPolicy, Part, Protocol, ProviderStateFormat, Reasoning, Role, ToolCall,
    ToolResult, normalize, normalize_with_options,
};
use serde_json::json;

#[test]
fn developer_and_custom_roles_are_downgraded() {
    let request = ChatRequest::new([
        Message::developer("system rules"),
        Message::custom("reviewer", "legacy message"),
        Message::user("hello"),
    ]);

    let (clean, report) = normalize(&request, Protocol::OpenAiChat).unwrap();

    assert_eq!(clean.system.as_deref(), Some("system rules"));
    assert_eq!(clean.messages.len(), 2);
    assert_eq!(clean.messages[0].role, ExternalRole::User);
    assert_eq!(clean.messages[1].role, ExternalRole::User);
    assert!(
        report
            .actions
            .contains(&NormalizeAction::DowngradedDeveloper)
    );
    assert!(
        report
            .actions
            .contains(&NormalizeAction::DowngradedCustomRole {
                role: "reviewer".to_string()
            })
    );
}

#[test]
fn missing_tool_result_is_synthesized_before_the_next_user_turn() {
    let request = ChatRequest::new([
        Message::assistant_with_tools("", [ToolCall::new("lookup", json!({"q": "rust"}))]),
        Message::user("continue"),
    ]);

    let (clean, report) = normalize(&request, Protocol::OpenAiChat).unwrap();

    assert_eq!(clean.messages.len(), 3);
    assert_eq!(clean.messages[0].role, ExternalRole::Assistant);
    assert_eq!(clean.messages[1].role, ExternalRole::Tool);
    assert_eq!(clean.messages[2].role, ExternalRole::User);
    assert!(
        report
            .actions
            .iter()
            .any(|action| matches!(action, NormalizeAction::AssignedToolCallId { .. }))
    );
    assert!(
        report
            .actions
            .iter()
            .any(|action| matches!(action, NormalizeAction::SynthesizedMissingToolResult { .. }))
    );

    let Part::ToolResult(result) = &clean.messages[1].parts[0] else {
        panic!("expected a tool result");
    };
    assert!(result.is_error);
    assert_eq!(result.content, "missing tool result");
}

#[test]
fn orphan_tool_result_is_dropped_by_default() {
    let request = ChatRequest::new([
        Message::user("hello"),
        Message::tool_result(ToolResult::new(None, "lookup", "unexpected")),
    ]);

    let (clean, report) = normalize(&request, Protocol::OpenAiChat).unwrap();

    assert_eq!(clean.messages.len(), 1);
    assert_eq!(clean.messages[0].role, ExternalRole::User);
    assert!(report.actions.iter().any(|action| matches!(
        action,
        NormalizeAction::DroppedOrphanToolResult { name } if name == "lookup"
    )));
}

#[test]
fn missing_tool_result_id_is_filled_from_the_tool_name() {
    let request = ChatRequest::new([
        Message::assistant_with_tools("", [ToolCall::new("lookup", json!({"q": "rust"}))]),
        Message::tool_result(ToolResult::new(None, "lookup", "result")),
    ]);

    let (clean, report) = normalize(&request, Protocol::OpenAiChat).unwrap();
    let Part::ToolCall(call) = &clean.messages[0].parts[0] else {
        panic!("expected tool call");
    };
    let Part::ToolResult(result) = &clean.messages[1].parts[0] else {
        panic!("expected tool result");
    };

    assert!(call.id.is_some());
    assert_eq!(result.call_id, call.id);
    assert!(
        report
            .actions
            .iter()
            .any(|action| matches!(action, NormalizeAction::FilledToolResultId { .. }))
    );
}

#[test]
fn anthropic_merges_adjacent_user_messages() {
    let request = ChatRequest::new([
        Message::user("first"),
        Message::user("second"),
        Message::assistant("ok"),
    ]);

    let (clean, report) = normalize(&request, Protocol::AnthropicMessages).unwrap();

    assert_eq!(clean.messages.len(), 2);
    assert_eq!(clean.messages[0].role, ExternalRole::User);
    assert_eq!(clean.messages[0].parts.len(), 2);
    assert!(report.actions.iter().any(|action| matches!(
        action,
        NormalizeAction::MergedAdjacent {
            role: ExternalRole::User
        }
    )));
}

#[test]
fn explicit_policy_can_drop_unpaired_tool_calls() {
    let request = ChatRequest::new([
        Message::assistant_with_tools("answer", [ToolCall::new("lookup", json!({}))]),
        Message::user("continue"),
    ]);
    let options = NormalizeOptions {
        protocol: Protocol::OpenAiChat,
        developer_to_system: true,
        custom_to_user: true,
        orphan_tool_result: OrphanToolResultPolicy::Drop,
        missing_tool_result: MissingToolResultPolicy::DropToolCall,
        merge_adjacent_same_role: false,
        preserve_foreign_reasoning: false,
    };

    let (clean, report) = normalize_with_options(&request, options).unwrap();

    assert_eq!(clean.messages.len(), 2);
    assert_eq!(clean.messages[0].role, ExternalRole::Assistant);
    assert!(
        !clean.messages[0]
            .parts
            .iter()
            .any(|part| matches!(part, Part::ToolCall(_)))
    );
    assert!(report.actions.iter().any(|action| matches!(
        action,
        NormalizeAction::DroppedUnpairedToolCall { name, .. } if name == "lookup"
    )));
}

#[test]
fn foreign_reasoning_state_is_dropped_at_the_protocol_boundary() {
    let request = ChatRequest::new([Message::new(
        Role::Assistant,
        vec![Part::reasoning(
            Reasoning::text("private thinking").with_state(
                ProviderStateFormat::AnthropicMessages,
                json!({
                    "type": "thinking",
                    "thinking": "private thinking",
                    "signature": "anthropic-signature"
                }),
            ),
        )],
    )]);

    let (clean, report) = normalize(&request, Protocol::OpenAiChat).unwrap();

    assert!(clean.messages.is_empty());
    assert!(report.actions.iter().any(|action| matches!(
        action,
        NormalizeAction::DroppedForeignReasoning {
            format: ProviderStateFormat::AnthropicMessages,
            expected: ProviderStateFormat::OpenAiChat
        }
    )));
}

#[test]
fn reasoning_on_non_assistant_roles_is_not_promoted_to_user_text() {
    let request = ChatRequest::new([Message::new(
        Role::User,
        vec![
            Part::reasoning(Reasoning::summary("not user content")),
            Part::text("visible user content"),
        ],
    )]);

    let (clean, report) = normalize(&request, Protocol::OpenAiChat).unwrap();

    assert_eq!(
        clean.messages[0].parts,
        vec![Part::text("visible user content")]
    );
    assert!(report.actions.iter().any(|action| matches!(
        action,
        NormalizeAction::DroppedReasoningFromNonAssistant {
            role: ExternalRole::User
        }
    )));
}

#[test]
fn provider_state_is_removed_from_foreign_tool_calls() {
    let request = ChatRequest::new([Message::new(
        Role::Assistant,
        vec![Part::ToolCall(
            ToolCall::new("lookup", json!({"q": "rust"})).with_provider_state(
                ProviderStateFormat::GeminiGenerateContent,
                json!({"thoughtSignature": "gemini-only"}),
            ),
        )],
    )]);

    let (clean, report) = normalize(&request, Protocol::OpenAiChat).unwrap();

    let Part::ToolCall(call) = &clean.messages[0].parts[0] else {
        panic!("expected tool call");
    };
    assert!(call.provider_state.is_none());
    assert!(report.actions.iter().any(|action| matches!(
        action,
        NormalizeAction::DroppedForeignToolState {
            format: ProviderStateFormat::GeminiGenerateContent,
            expected: ProviderStateFormat::OpenAiChat
        }
    )));
}

#[test]
fn provider_state_is_removed_from_foreign_text_without_losing_text() {
    let request = ChatRequest::new([Message::new(
        Role::Assistant,
        vec![Part::text_with_provider_state(
            "visible answer",
            ProviderStateFormat::GeminiGenerateContent,
            json!({"text": "visible answer", "thoughtSignature": "gemini-only"}),
        )],
    )]);

    let (clean, report) = normalize(&request, Protocol::OpenAiChat).unwrap();

    assert_eq!(clean.messages[0].parts, vec![Part::text("visible answer")]);
    assert!(report.actions.iter().any(|action| matches!(
        action,
        NormalizeAction::DroppedForeignTextState {
            format: ProviderStateFormat::GeminiGenerateContent,
            expected: ProviderStateFormat::OpenAiChat
        }
    )));
}

#[test]
fn custom_role_is_kept_out_of_the_wire_role_set() {
    let request = ChatRequest::new([Message::new(
        Role::Custom("planner".to_string()),
        vec![Part::text("plan")],
    )]);

    let (clean, _) = normalize(&request, Protocol::GeminiGenerateContent).unwrap();

    assert_eq!(clean.messages[0].role, ExternalRole::User);
}
