use mutil_ai::{ChatRequest, ModelAdapter, OpenAI, StreamEvent, next_event};

#[tokio::main]
async fn main() -> mutil_ai::Result<()> {
    let model = OpenAI::chat("gpt-5.2");
    let request = ChatRequest::user("用三句话解释 Rust 的 ownership。");

    let mut stream = model.stream(&request).await?;
    while let Some(event) = next_event(&mut stream).await {
        match event? {
            StreamEvent::Start {
                provider, model, ..
            } => {
                println!("[{provider}/{model}] stream started");
            }
            StreamEvent::ReasoningDelta { text, .. } => {
                print!("[thinking] {text}");
            }
            StreamEvent::TextDelta { text } => {
                print!("{text}");
            }
            StreamEvent::ToolCallDelta {
                index,
                name,
                arguments_delta,
                ..
            } => {
                println!(
                    "[tool {index}] {} {arguments_delta}",
                    name.as_deref().unwrap_or("...")
                );
            }
            StreamEvent::ToolCallProgress {
                index,
                name,
                arguments_so_far,
                ..
            } => {
                println!(
                    "[tool {index}] {} accumulated {arguments_so_far}",
                    name.as_deref().unwrap_or("...")
                );
            }
            StreamEvent::ServerToolStatus { tool, state, .. } => {
                println!("[server tool {tool}] {state:?}");
            }
            StreamEvent::Usage { usage } => {
                println!("usage: {usage:?}");
            }
            StreamEvent::Retry { delay } => {
                println!("server requested retry after {delay:?}");
            }
            StreamEvent::Retrying {
                attempt,
                max_attempts,
                delay,
                reason,
            } => {
                println!("retry {attempt}/{max_attempts} after {delay:?}: {reason}");
            }
            StreamEvent::Error { error } => {
                eprintln!("stream error {:?}: {}", error.kind, error.message);
            }
            StreamEvent::Done { response, .. } => {
                println!("\n\nfinal message: {}", response.text());
            }
        }
    }

    Ok(())
}
