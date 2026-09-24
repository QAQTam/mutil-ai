use mutil_ai::{Agent, OpenAI, tool_fn};
use serde_json::json;

#[tokio::main]
async fn main() -> mutil_ai::Result<()> {
    let weather = tool_fn(
        "get_weather",
        "查询一个城市的天气",
        json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "城市名称" }
            },
            "required": ["city"],
            "additionalProperties": false
        }),
        |arguments| async move {
            let city = arguments
                .get("city")
                .and_then(|value| value.as_str())
                .unwrap_or("unknown");

            Ok::<_, mutil_ai::Error>(json!({
                "city": city,
                "weather": "sunny",
                "temperature_c": 26
            }))
        },
    );

    let mut agent = Agent::builder()
        .model(OpenAI::responses("gpt-5.2"))
        .system("需要天气信息时调用工具，不要猜。")
        .tool(weather)
        .max_steps(4)
        .build()?;

    let answer = agent.ask("广州现在天气怎么样？").await?;
    println!("{answer}");

    for step in agent.history() {
        println!("{step:#?}");
    }

    Ok(())
}
