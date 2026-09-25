use mutil_ai::{Message, Part, ProviderStateFormat, Reasoning, ReasoningKind, Role};
use serde_json::json;

fn main() {
    // Adapters produce this shape for every provider. The opaque state below
    // would contain an OpenAI reasoning item, an Anthropic signature, or a
    // Gemini thoughtSignature in a real response.
    let message = Message::new(
        Role::Assistant,
        vec![
            Part::reasoning(Reasoning {
                kind: ReasoningKind::Summary,
                summary: Some("先确认用户要的是 Rust 示例。".to_string()),
                text: None,
                state: Some(mutil_ai::ProviderState::new(
                    ProviderStateFormat::OpenAiResponses,
                    json!({
                        "id": "rs_example",
                        "type": "reasoning",
                        "summary": [{
                            "type": "summary_text",
                            "text": "先确认用户要的是 Rust 示例。"
                        }]
                    }),
                )),
            }),
            Part::text("下面是最终答案。"),
        ],
    );

    for part in &message.parts {
        match part {
            Part::Text { text, .. } => println!("message: {text}"),
            Part::Reasoning(reasoning) if reasoning.kind == ReasoningKind::Summary => {
                println!(
                    "summary: {}",
                    reasoning.summary.as_deref().unwrap_or_default()
                );
            }
            Part::Reasoning(reasoning) => {
                println!(
                    "reasoning: {}",
                    reasoning.text.as_deref().unwrap_or_default()
                );
            }
            Part::ToolCall(call) => println!("tool call: {}", call.name),
            Part::ToolResult(result) => println!("tool result: {}", result.content),
            Part::ImageUrl { image_url } => println!("image: {}", image_url.url),
            Part::Image { image } => println!("image source: {:?}", image.source),
            Part::ProviderItem(item) => println!("provider item: {:?}", item.tool),
        }
    }
}
