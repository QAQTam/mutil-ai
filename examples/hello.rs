use mutil_ai::{ChatRequest, Message, ModelAdapter, OpenAI};

#[tokio::main]
async fn main() -> mutil_ai::Result<()> {
    // The adapter reads OPENAI_API_KEY from the environment.
    let model = OpenAI::responses("gpt-5.2");

    // Switch the adapter without changing the request code:
    // let model = OpenAI::chat("gpt-4.1");
    // let model = mutil_ai::Anthropic::messages("claude-sonnet-4-5");
    // let model = mutil_ai::Gemini::generate_content("gemini-3-flash");

    let request = ChatRequest::new([
        Message::system("你是一个简洁、耐心的 Rust 入门老师。"),
        Message::user("用三句话解释什么是 agent。"),
    ]);

    let response = model.complete(&request).await?;
    println!("{}", response.text());

    Ok(())
}
