//! A minimal agent runtime built entirely on the public `mutil-ai` API.
//!
//! This file is intentionally an example, not a crate export. It shows the
//! smallest useful loop a downstream project can copy and adapt:
//!
//! 1. keep its own conversation history;
//! 2. call [`ModelAdapter::complete_with`];
//! 3. execute requested tools;
//! 4. append tool results and continue until the model returns text.
//!
//! Production runtimes will usually replace `ToolRegistry` and the fixed
//! `max_steps` policy with their own permissions, cancellation, tracing, and
//! scheduling logic.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example minimal_agent
//! ```

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use async_trait::async_trait;
use mutil_ai::{
    ChatRequest, Error, Message, ModelAdapter, RequestOptions, Result, ToolCall, ToolResult,
    ToolSpec,
};
use serde_json::json;

/// The final result of one agent run.
#[derive(Debug, Clone)]
struct AgentReply {
    text: String,
    /// Every assistant message produced during the run, including tool-call
    /// turns.
    steps: Vec<Message>,
    /// Tool results collected during the run.
    tool_results: Vec<ToolResult>,
}

/// A tool implemented by the downstream runtime.
#[async_trait]
trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;

    async fn call(&self, arguments: serde_json::Value) -> Result<serde_json::Value>;
}

struct FunctionTool<F> {
    spec: ToolSpec,
    function: F,
}

impl<F> FunctionTool<F> {
    fn new(spec: ToolSpec, function: F) -> Self {
        Self { spec, function }
    }
}

#[async_trait]
impl<F, Fut> Tool for FunctionTool<F>
where
    F: Fn(serde_json::Value) -> Fut + Send + Sync,
    Fut: Future<Output = Result<serde_json::Value>> + Send,
{
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    async fn call(&self, arguments: serde_json::Value) -> Result<serde_json::Value> {
        (self.function)(arguments).await
    }
}

fn tool_fn<F, Fut>(
    name: impl Into<String>,
    description: impl Into<String>,
    parameters: serde_json::Value,
    function: F,
) -> FunctionTool<F>
where
    F: Fn(serde_json::Value) -> Fut + Send + Sync,
    Fut: Future<Output = Result<serde_json::Value>> + Send,
{
    FunctionTool::new(ToolSpec::new(name, description, parameters), function)
}

/// A minimal registry owned by the example runtime.
#[derive(Default, Clone)]
struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    fn insert<T>(&mut self, tool: T)
    where
        T: Tool + 'static,
    {
        let spec = tool.spec();
        self.tools.insert(spec.name.clone(), Arc::new(tool));
    }

    fn specs(&self) -> Vec<ToolSpec> {
        let mut specs: Vec<_> = self.tools.values().map(|tool| tool.spec()).collect();
        specs.sort_by(|a, b| a.name.cmp(&b.name));
        specs
    }

    async fn call(&self, call: &ToolCall) -> Result<ToolResult> {
        let Some(tool) = self.tools.get(&call.name) else {
            return Ok(ToolResult::error(
                call.id.clone(),
                call.name.clone(),
                format!("tool `{}` is not registered", call.name),
            ));
        };

        match tool.call(call.arguments.clone()).await {
            Ok(value) => Ok(ToolResult::json(call.id.clone(), call.name.clone(), value)),
            Err(error) => Ok(ToolResult::error(
                call.id.clone(),
                call.name.clone(),
                error.to_string(),
            )),
        }
    }
}

struct AgentBuilder {
    model: Option<Arc<dyn ModelAdapter>>,
    system: Option<String>,
    tools: ToolRegistry,
    max_steps: usize,
    request_options: RequestOptions,
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self {
            model: None,
            system: None,
            tools: ToolRegistry::default(),
            max_steps: 8,
            request_options: RequestOptions::default(),
        }
    }
}

impl AgentBuilder {
    fn new() -> Self {
        Self::default()
    }

    fn model<M>(mut self, model: M) -> Self
    where
        M: ModelAdapter + 'static,
    {
        self.model = Some(Arc::new(model));
        self
    }

    fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    fn tool<T>(mut self, tool: T) -> Self
    where
        T: Tool + 'static,
    {
        self.tools.insert(tool);
        self
    }

    fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps.max(1);
        self
    }

    fn build(self) -> Result<Agent> {
        let model = self.model.ok_or(Error::MissingModel("agent"))?;
        Ok(Agent {
            model,
            system: self.system,
            tools: self.tools,
            max_steps: self.max_steps,
            request_options: self.request_options,
            history: Vec::new(),
        })
    }
}

/// The minimal runtime loop. It deliberately stays outside `mutil-ai`.
struct Agent {
    model: Arc<dyn ModelAdapter>,
    system: Option<String>,
    tools: ToolRegistry,
    max_steps: usize,
    request_options: RequestOptions,
    history: Vec<Message>,
}

impl Agent {
    fn builder() -> AgentBuilder {
        AgentBuilder::new()
    }

    async fn ask_detailed(&mut self, input: impl Into<String>) -> Result<AgentReply> {
        self.history.push(Message::user(input));

        let mut steps = Vec::new();
        let mut all_tool_results = Vec::new();

        for _step in 0..self.max_steps {
            let request = self.build_request();
            let response = self
                .model
                .complete_with(&request, &self.request_options)
                .await?;
            self.history.push(response.message.clone());
            steps.push(response.message.clone());

            let calls: Vec<_> = response.tool_calls().cloned().collect();
            if calls.is_empty() {
                return Ok(AgentReply {
                    text: response.text(),
                    steps,
                    tool_results: all_tool_results,
                });
            }

            let mut tool_results = Vec::new();
            for call in calls {
                let result = self.tools.call(&call).await?;
                all_tool_results.push(result.clone());
                tool_results.push(result);
            }
            self.history.push(Message::tool_results(tool_results));
        }

        Err(Error::MaxSteps(self.max_steps))
    }

    fn build_request(&self) -> ChatRequest {
        let mut messages = Vec::with_capacity(self.history.len() + 1);
        if let Some(system) = &self.system {
            messages.push(Message::system(system));
        }
        messages.extend(self.history.iter().cloned());

        ChatRequest {
            messages,
            tools: self.tools.specs(),
            ..ChatRequest::default()
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
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

            Ok::<_, Error>(json!({
                "city": city,
                "weather": "sunny",
                "temperature_c": 26
            }))
        },
    );

    let mut agent = Agent::builder()
        .model(mutil_ai::OpenAI::responses("gpt-5.2"))
        .system("需要天气信息时调用工具，不要猜。")
        .tool(weather)
        .max_steps(4)
        .build()?;

    let reply = agent.ask_detailed("广州现在天气怎么样？").await?;
    println!("{}", reply.text);
    println!(
        "model_steps={}, tool_results={}",
        reply.steps.len(),
        reply.tool_results.len()
    );

    Ok(())
}
