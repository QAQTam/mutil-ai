use std::sync::Arc;

use crate::adapter::ModelAdapter;
use crate::error::{Error, Result};
use crate::headers::RequestOptions;
use crate::tool::{Tool, ToolRegistry};
use crate::types::{ChatRequest, ChatResponse, Message, ReasoningConfig, ToolResult};

/// The final result of an agent run.
#[derive(Debug, Clone)]
pub struct AgentReply {
    pub text: String,
    /// Every assistant message produced during the run, including tool-call
    /// turns. This is useful when learning how an agent loop works.
    pub steps: Vec<Message>,
    /// Tool results collected during the run.
    pub tool_results: Vec<ToolResult>,
}

/// Builder for an [`Agent`].
pub struct AgentBuilder {
    model: Option<Arc<dyn ModelAdapter>>,
    system: Option<String>,
    tools: ToolRegistry,
    max_steps: usize,
    temperature: Option<f32>,
    max_output_tokens: Option<u32>,
    reasoning: Option<ReasoningConfig>,
    request_options: RequestOptions,
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self {
            model: None,
            system: None,
            tools: ToolRegistry::new(),
            max_steps: 8,
            temperature: None,
            max_output_tokens: None,
            reasoning: None,
            request_options: RequestOptions::default(),
        }
    }
}

impl AgentBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the model adapter. The adapter owns the model name and provider
    /// credentials.
    pub fn model<M>(mut self, model: M) -> Self
    where
        M: ModelAdapter + 'static,
    {
        self.model = Some(Arc::new(model));
        self
    }

    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    pub fn tool<T>(mut self, tool: T) -> Self
    where
        T: Tool + 'static,
    {
        self.tools.insert(tool);
        self
    }

    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps.max(1);
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn max_output_tokens(mut self, max_output_tokens: u32) -> Self {
        self.max_output_tokens = Some(max_output_tokens);
        self
    }

    pub fn reasoning(mut self, reasoning: ReasoningConfig) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    pub fn request_options(mut self, request_options: RequestOptions) -> Self {
        self.request_options = request_options;
        self
    }

    pub fn build(self) -> Result<Agent> {
        let model = self.model.ok_or(Error::MissingModel("agent"))?;
        Ok(Agent {
            model,
            system: self.system,
            tools: self.tools,
            max_steps: self.max_steps,
            temperature: self.temperature,
            max_output_tokens: self.max_output_tokens,
            reasoning: self.reasoning,
            request_options: self.request_options,
            history: Vec::new(),
        })
    }
}

/// A small agent loop.
///
/// `Agent` keeps conversation history, sends tool definitions to the model,
/// executes requested tools, appends tool results, and repeats until the model
/// returns a normal answer or `max_steps` is reached.
pub struct Agent {
    model: Arc<dyn ModelAdapter>,
    system: Option<String>,
    tools: ToolRegistry,
    max_steps: usize,
    temperature: Option<f32>,
    max_output_tokens: Option<u32>,
    reasoning: Option<ReasoningConfig>,
    request_options: RequestOptions,
    history: Vec<Message>,
}

impl Agent {
    pub fn builder() -> AgentBuilder {
        AgentBuilder::new()
    }

    pub fn provider_name(&self) -> &'static str {
        self.model.provider_name()
    }

    pub fn model_name(&self) -> &str {
        self.model.model_name()
    }

    pub fn history(&self) -> &[Message] {
        &self.history
    }

    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Ask the agent and return only the final text.
    pub async fn ask(&mut self, input: impl Into<String>) -> Result<String> {
        Ok(self.ask_detailed(input).await?.text)
    }

    /// Ask the agent and return the full trace of the run.
    pub async fn ask_detailed(&mut self, input: impl Into<String>) -> Result<AgentReply> {
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
            temperature: self.temperature,
            max_output_tokens: self.max_output_tokens,
            reasoning: self.reasoning.clone(),
            ..ChatRequest::default()
        }
    }
}

/// Convenience for code that wants the raw model response without the agent
/// tool loop.
pub async fn complete_once(
    model: &dyn ModelAdapter,
    request: &ChatRequest,
) -> Result<ChatResponse> {
    model.complete(request).await
}
