use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;
use crate::types::{ToolCall, ToolResult, ToolSpec};

/// A tool the agent can execute.
#[async_trait]
pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;

    async fn call(&self, arguments: serde_json::Value) -> Result<serde_json::Value>;
}

/// A closure-backed tool. This is the easiest way to create a small tool.
pub struct FunctionTool<F> {
    spec: ToolSpec,
    function: F,
}

impl<F> FunctionTool<F> {
    pub fn new(spec: ToolSpec, function: F) -> Self {
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

/// Create a tool from a normal Rust closure.
///
/// The closure receives JSON arguments and returns JSON. For beginners, a
/// closure is usually easier than defining a full `Tool` implementation.
pub fn tool_fn<F, Fut>(
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

/// A small registry of tools.
#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert<T>(&mut self, tool: T)
    where
        T: Tool + 'static,
    {
        let spec = tool.spec();
        self.tools.insert(spec.name.clone(), Arc::new(tool));
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        let mut specs: Vec<_> = self.tools.values().map(|tool| tool.spec()).collect();
        specs.sort_by(|a, b| a.name.cmp(&b.name));
        specs
    }

    pub async fn call(&self, call: &ToolCall) -> Result<ToolResult> {
        let Some(tool) = self.get(&call.name) else {
            return Ok(ToolResult::error(
                call.id.clone(),
                call.name.clone(),
                format!("tool `{}` is not registered", call.name),
            ));
        };

        match tool.call(call.arguments.clone()).await {
            Ok(value) => Ok(ToolResult::new(
                call.id.clone(),
                call.name.clone(),
                json_to_tool_content(&value),
            )),
            Err(error) => Ok(ToolResult::error(
                call.id.clone(),
                call.name.clone(),
                error.to_string(),
            )),
        }
    }
}

fn json_to_tool_content(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text.clone(),
        _ => serde_json::to_string(value).unwrap_or_else(|_| "null".to_string()),
    }
}
