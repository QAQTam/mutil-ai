use mutil_ai::{Agent, OpenAI};

#[tokio::main]
async fn main() -> mutil_ai::Result<()> {
    // The adapter reads OPENAI_API_KEY from the environment.
    let model = OpenAI::responses("gpt-5.2");

    // Switch the adapter without changing the agent code:
    // let model = OpenAI::chat("gpt-4.1");
    // let model = mutil_ai::Anthropic::messages("claude-sonnet-4-5");
    // let model = mutil_ai::Gemini::generate_content("gemini-3-flash");

    let mut agent = Agent::builder()
        .model(model)
        .system("你是一个简洁、耐心的 Rust 入门老师。")
        .build()?;

    let answer = agent.ask("用三句话解释什么是 agent。").await?;
    println!("{answer}");

    Ok(())
}
