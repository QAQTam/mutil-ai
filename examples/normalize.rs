use mutil_ai::{ChatRequest, Message, Protocol, ToolCall, normalize};
use serde_json::json;

fn main() -> mutil_ai::Result<()> {
    let request = ChatRequest::new([
        Message::developer("你是一个严谨的助手。"),
        Message::custom("reviewer", "这段内容来自旧版 agent。"),
        Message::assistant_with_tools("", [ToolCall::new("get_weather", json!({"city": "广州"}))]),
        // Deliberately omit the matching tool result.
        Message::user("继续回答。"),
    ]);

    let (clean, report) = normalize(&request, Protocol::AnthropicMessages)?;

    println!("system = {:?}", clean.system);
    println!("messages = {:#?}", clean.messages);
    println!("repairs = {:#?}", report.actions);

    Ok(())
}
