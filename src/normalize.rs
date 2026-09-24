use crate::error::{Error, Result};
use crate::types::{ChatRequest, Message, Part, ProviderStateFormat, Role, ToolResult, ToolSpec};

/// The provider-facing protocol we are preparing a request for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    OpenAiChat,
    OpenAiResponses,
    AnthropicMessages,
    GeminiGenerateContent,
}

impl Protocol {
    pub fn provider_state_format(self) -> ProviderStateFormat {
        match self {
            Self::OpenAiChat => ProviderStateFormat::OpenAiChat,
            Self::OpenAiResponses => ProviderStateFormat::OpenAiResponses,
            Self::AnthropicMessages => ProviderStateFormat::AnthropicMessages,
            Self::GeminiGenerateContent => ProviderStateFormat::GeminiGenerateContent,
        }
    }
}

/// Roles that are safe to send to a provider adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExternalRole {
    System,
    User,
    Assistant,
    Tool,
}

/// What to do with a tool result that has no matching assistant tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrphanToolResultPolicy {
    /// Remove it. This is the safest default for a strict provider API.
    Drop,
    /// Turn it into a user message. Useful when debugging messy histories.
    DowngradeToUser,
    /// Stop normalization and return an error.
    Error,
}

/// What to do when an assistant tool call has no result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingToolResultPolicy {
    /// Add an error result so that tool calls are always paired.
    Synthesize,
    /// Remove the tool call from the assistant message.
    DropToolCall,
    /// Stop normalization and return an error.
    Error,
}

/// Rules used by the normalization layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizeOptions {
    /// The provider protocol that will receive the normalized request.
    pub protocol: Protocol,
    /// `Developer` is not understood by every provider, so downgrade it to
    /// `System` by default.
    pub developer_to_system: bool,
    /// Unknown/custom roles are downgraded to `User` by default.
    pub custom_to_user: bool,
    pub orphan_tool_result: OrphanToolResultPolicy,
    pub missing_tool_result: MissingToolResultPolicy,
    /// Anthropic and Gemini require user/model turns to alternate. OpenAI does
    /// not require this, so its adapter keeps the original message boundaries.
    pub merge_adjacent_same_role: bool,
    /// Keep reasoning state produced by a different provider protocol.
    ///
    /// This is disabled by default because signatures and encrypted reasoning
    /// tokens are provider-specific and cannot be safely translated.
    pub preserve_foreign_reasoning: bool,
}

impl NormalizeOptions {
    pub fn for_protocol(protocol: Protocol) -> Self {
        let merge_adjacent_same_role = matches!(
            protocol,
            Protocol::AnthropicMessages | Protocol::GeminiGenerateContent
        );

        Self {
            protocol,
            developer_to_system: true,
            custom_to_user: true,
            orphan_tool_result: OrphanToolResultPolicy::Drop,
            missing_tool_result: MissingToolResultPolicy::Synthesize,
            merge_adjacent_same_role,
            preserve_foreign_reasoning: false,
        }
    }
}

/// A provider-facing message after role downgrade and tool-pair repair.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedMessage {
    pub role: ExternalRole,
    pub parts: Vec<Part>,
}

/// A provider-facing conversation.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedChat {
    /// Anthropic, Gemini and OpenAI Responses accept system instructions
    /// outside the normal message list. OpenAI Chat can prepend this as a
    /// system message.
    pub system: Option<String>,
    pub messages: Vec<NormalizedMessage>,
    pub tools: Vec<ToolSpec>,
}

/// A human-readable trace of every repair made at the boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalizeAction {
    DowngradedDeveloper,
    DowngradedCustomRole {
        role: String,
    },
    AssignedToolCallId {
        name: String,
        id: String,
    },
    FilledToolResultId {
        name: String,
        id: String,
    },
    DroppedOrphanToolResult {
        name: String,
    },
    SynthesizedMissingToolResult {
        call_id: String,
        name: String,
    },
    DroppedUnpairedToolCall {
        call_id: String,
        name: String,
    },
    DroppedForeignReasoning {
        format: ProviderStateFormat,
        expected: ProviderStateFormat,
    },
    DroppedForeignToolState {
        format: ProviderStateFormat,
        expected: ProviderStateFormat,
    },
    DroppedForeignTextState {
        format: ProviderStateFormat,
        expected: ProviderStateFormat,
    },
    DroppedReasoningFromNonAssistant {
        role: ExternalRole,
    },
    MergedAdjacent {
        role: ExternalRole,
    },
}

/// Details about normalization. Callers can log this while learning.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NormalizeReport {
    pub actions: Vec<NormalizeAction>,
}

impl NormalizeReport {
    pub fn is_clean(&self) -> bool {
        self.actions.is_empty()
    }

    /// Aggregate repair counts without role names, tool names, call ids, or
    /// message content.
    ///
    /// This is the form used by [`crate::AuditEvent`] so enabling operational
    /// diagnostics never leaks conversation context.
    pub fn stats(&self) -> NormalizeStats {
        let mut stats = NormalizeStats::default();
        for action in &self.actions {
            stats.total += 1;
            match action {
                NormalizeAction::DowngradedDeveloper => stats.downgraded_developer += 1,
                NormalizeAction::DowngradedCustomRole { .. } => {
                    stats.downgraded_custom_role += 1;
                }
                NormalizeAction::AssignedToolCallId { .. } => stats.assigned_tool_call_id += 1,
                NormalizeAction::FilledToolResultId { .. } => stats.filled_tool_result_id += 1,
                NormalizeAction::DroppedOrphanToolResult { .. } => {
                    stats.dropped_orphan_tool_result += 1;
                }
                NormalizeAction::SynthesizedMissingToolResult { .. } => {
                    stats.synthesized_missing_tool_result += 1;
                }
                NormalizeAction::DroppedUnpairedToolCall { .. } => {
                    stats.dropped_unpaired_tool_call += 1;
                }
                NormalizeAction::DroppedForeignReasoning { .. } => {
                    stats.dropped_foreign_reasoning += 1;
                }
                NormalizeAction::DroppedForeignToolState { .. } => {
                    stats.dropped_foreign_tool_state += 1;
                }
                NormalizeAction::DroppedForeignTextState { .. } => {
                    stats.dropped_foreign_text_state += 1;
                }
                NormalizeAction::DroppedReasoningFromNonAssistant { .. } => {
                    stats.dropped_reasoning_from_non_assistant += 1;
                }
                NormalizeAction::MergedAdjacent { .. } => stats.merged_adjacent += 1,
            }
        }
        stats
    }
}

/// Context-free aggregate of normalization repairs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NormalizeStats {
    pub total: u32,
    pub downgraded_developer: u32,
    pub downgraded_custom_role: u32,
    pub assigned_tool_call_id: u32,
    pub filled_tool_result_id: u32,
    pub dropped_orphan_tool_result: u32,
    pub synthesized_missing_tool_result: u32,
    pub dropped_unpaired_tool_call: u32,
    pub dropped_foreign_reasoning: u32,
    pub dropped_foreign_tool_state: u32,
    pub dropped_foreign_text_state: u32,
    pub dropped_reasoning_from_non_assistant: u32,
    pub merged_adjacent: u32,
}

impl NormalizeStats {
    pub const fn is_clean(self) -> bool {
        self.total == 0
    }
}

#[derive(Debug, Clone)]
struct PendingCall {
    id: String,
    name: String,
    message_index: usize,
    fulfilled: bool,
}

/// Normalize a request using the default rules for a protocol.
pub fn normalize(
    request: &ChatRequest,
    protocol: Protocol,
) -> Result<(NormalizedChat, NormalizeReport)> {
    normalize_with_options(request, NormalizeOptions::for_protocol(protocol))
}

/// Normalize a request with explicit rules.
///
/// The output is intentionally boring: only `System`, `User`, `Assistant` and
/// `Tool` can reach an adapter, and every `ToolCall` is paired with a
/// `ToolResult`.
pub fn normalize_with_options(
    request: &ChatRequest,
    options: NormalizeOptions,
) -> Result<(NormalizedChat, NormalizeReport)> {
    let expected_reasoning_format = options.protocol.provider_state_format();
    let mut normalizer = Normalizer {
        options,
        report: NormalizeReport::default(),
        system: Vec::new(),
        messages: Vec::new(),
        pending_calls: Vec::new(),
        next_call_id: 1,
        expected_reasoning_format,
    };

    for message in &request.messages {
        normalizer.push(message)?;
    }
    normalizer.finish_pending_calls()?;
    normalizer.merge_adjacent_messages();

    Ok((
        NormalizedChat {
            system: normalizer.join_system(),
            messages: normalizer.messages,
            tools: request.tools.clone(),
        },
        normalizer.report,
    ))
}

struct Normalizer {
    options: NormalizeOptions,
    report: NormalizeReport,
    system: Vec<String>,
    messages: Vec<NormalizedMessage>,
    pending_calls: Vec<PendingCall>,
    next_call_id: u64,
    expected_reasoning_format: ProviderStateFormat,
}

impl Normalizer {
    fn push(&mut self, message: &Message) -> Result<()> {
        match &message.role {
            Role::System => {
                self.push_system(message);
                Ok(())
            }
            Role::Developer => {
                self.report
                    .actions
                    .push(NormalizeAction::DowngradedDeveloper);
                if self.options.developer_to_system {
                    self.push_system(message);
                } else {
                    self.push_user(message);
                }
                Ok(())
            }
            Role::Custom(role) => {
                self.report
                    .actions
                    .push(NormalizeAction::DowngradedCustomRole { role: role.clone() });
                if self.options.custom_to_user {
                    self.push_user(message);
                } else {
                    self.push_system(message);
                }
                Ok(())
            }
            Role::User => {
                self.finish_pending_calls()?;
                self.push_user(message);
                Ok(())
            }
            Role::Assistant => {
                self.finish_pending_calls()?;
                self.push_assistant(message)
            }
            Role::Tool => self.push_tool(message),
        }
    }

    fn push_system(&mut self, message: &Message) {
        self.report_non_assistant_reasoning(ExternalRole::System, &message.parts);
        let text = text_parts_joined(&message.parts);
        if !text.is_empty() {
            self.system.push(text);
        }
    }

    fn push_user(&mut self, message: &Message) {
        self.report_non_assistant_reasoning(ExternalRole::User, &message.parts);
        let parts = normalize_non_tool_parts(&message.parts);
        if !parts.is_empty() {
            self.messages.push(NormalizedMessage {
                role: ExternalRole::User,
                parts,
            });
        }
    }

    fn push_assistant(&mut self, message: &Message) -> Result<()> {
        let message_index = self.messages.len();
        let mut parts = Vec::new();

        for part in &message.parts {
            match part {
                Part::Text {
                    text,
                    provider_state,
                } => {
                    let text = text.clone();
                    let mut provider_state = provider_state.clone();
                    if !self.options.preserve_foreign_reasoning
                        && let Some(state) = &provider_state
                        && state.format != self.expected_reasoning_format
                    {
                        self.report
                            .actions
                            .push(NormalizeAction::DroppedForeignTextState {
                                format: state.format.clone(),
                                expected: self.expected_reasoning_format.clone(),
                            });
                        provider_state = None;
                    }
                    parts.push(Part::Text {
                        text,
                        provider_state,
                    });
                }
                Part::ImageUrl { .. } => parts.push(part.clone()),
                Part::Reasoning(reasoning) => {
                    if !self.options.preserve_foreign_reasoning
                        && let Some(state) = &reasoning.state
                        && state.format != self.expected_reasoning_format
                    {
                        self.report
                            .actions
                            .push(NormalizeAction::DroppedForeignReasoning {
                                format: state.format.clone(),
                                expected: self.expected_reasoning_format.clone(),
                            });
                        continue;
                    }
                    parts.push(Part::Reasoning(reasoning.clone()));
                }
                Part::ToolCall(call) => {
                    let mut call = call.clone();
                    if !self.options.preserve_foreign_reasoning
                        && let Some(state) = &call.provider_state
                        && state.format != self.expected_reasoning_format
                    {
                        self.report
                            .actions
                            .push(NormalizeAction::DroppedForeignToolState {
                                format: state.format.clone(),
                                expected: self.expected_reasoning_format.clone(),
                            });
                        call.provider_state = None;
                    }
                    if call.id.is_none() {
                        let id = format!("call_{}", self.next_call_id);
                        self.next_call_id += 1;
                        call.id = Some(id.clone());
                        self.report
                            .actions
                            .push(NormalizeAction::AssignedToolCallId {
                                name: call.name.clone(),
                                id: id.clone(),
                            });
                    }

                    let id = call.id.clone().expect("id was assigned above");
                    self.pending_calls.push(PendingCall {
                        id: id.clone(),
                        name: call.name.clone(),
                        message_index,
                        fulfilled: false,
                    });
                    parts.push(Part::ToolCall(call));
                }
                Part::ToolResult(result) => {
                    // A result inside an assistant message is malformed.
                    self.handle_orphan_tool_result(result)?;
                }
            }
        }

        if !parts.is_empty() {
            self.messages.push(NormalizedMessage {
                role: ExternalRole::Assistant,
                parts,
            });
        }

        Ok(())
    }

    fn push_tool(&mut self, message: &Message) -> Result<()> {
        self.report_non_assistant_reasoning(ExternalRole::Tool, &message.parts);
        let mut accepted = Vec::new();

        for part in &message.parts {
            let Part::ToolResult(result) = part else {
                continue;
            };

            let Some(index) = self.find_pending_call(result) else {
                self.handle_orphan_tool_result(result)?;
                continue;
            };

            let pending = &mut self.pending_calls[index];
            pending.fulfilled = true;
            let call_id = pending.id.clone();
            let name = pending.name.clone();

            let mut result = result.clone();
            if result.call_id.is_none() {
                result.call_id = Some(call_id.clone());
                self.report
                    .actions
                    .push(NormalizeAction::FilledToolResultId {
                        name: result.name.clone(),
                        id: call_id.clone(),
                    });
            }
            result.name = name;
            accepted.push(Part::ToolResult(result));
        }

        if !accepted.is_empty() {
            self.messages.push(NormalizedMessage {
                role: ExternalRole::Tool,
                parts: accepted,
            });
        }

        Ok(())
    }

    fn report_non_assistant_reasoning(&mut self, role: ExternalRole, parts: &[Part]) {
        if parts.iter().any(|part| matches!(part, Part::Reasoning(_))) {
            self.report
                .actions
                .push(NormalizeAction::DroppedReasoningFromNonAssistant { role });
        }
    }

    fn find_pending_call(&self, result: &ToolResult) -> Option<usize> {
        if let Some(call_id) = &result.call_id
            && let Some(index) = self
                .pending_calls
                .iter()
                .position(|call| call.id == *call_id)
        {
            return Some(index);
        }

        self.pending_calls
            .iter()
            .position(|call| call.name == result.name && !call.fulfilled)
    }

    fn handle_orphan_tool_result(&mut self, result: &ToolResult) -> Result<()> {
        match self.options.orphan_tool_result {
            OrphanToolResultPolicy::Drop => {
                self.report
                    .actions
                    .push(NormalizeAction::DroppedOrphanToolResult {
                        name: result.name.clone(),
                    });
                Ok(())
            }
            OrphanToolResultPolicy::DowngradeToUser => {
                self.messages.push(NormalizedMessage {
                    role: ExternalRole::User,
                    parts: vec![Part::text(format!(
                        "Orphan tool result from `{}`: {}",
                        result.name, result.content
                    ))],
                });
                Ok(())
            }
            OrphanToolResultPolicy::Error => Err(Error::Normalize(format!(
                "tool result `{}` has no matching assistant tool call",
                result.name
            ))),
        }
    }

    fn finish_pending_calls(&mut self) -> Result<()> {
        let missing: Vec<(String, String, usize)> = self
            .pending_calls
            .iter()
            .filter(|call| !call.fulfilled)
            .map(|call| (call.id.clone(), call.name.clone(), call.message_index))
            .collect();

        if missing.is_empty() {
            self.pending_calls.clear();
            return Ok(());
        }

        match self.options.missing_tool_result {
            MissingToolResultPolicy::Synthesize => {
                let mut results = Vec::new();
                for (call_id, name, _) in missing {
                    self.report
                        .actions
                        .push(NormalizeAction::SynthesizedMissingToolResult {
                            call_id: call_id.clone(),
                            name: name.clone(),
                        });
                    results.push(Part::ToolResult(ToolResult::error(
                        Some(call_id),
                        name,
                        "missing tool result",
                    )));
                }
                self.messages.push(NormalizedMessage {
                    role: ExternalRole::Tool,
                    parts: results,
                });
                self.pending_calls.clear();
                Ok(())
            }
            MissingToolResultPolicy::DropToolCall => {
                for (call_id, name, message_index) in missing {
                    if let Some(message) = self.messages.get_mut(message_index) {
                        message.parts.retain(|part| {
                            !matches!(
                                part,
                                Part::ToolCall(call)
                                    if call.id.as_deref() == Some(call_id.as_str())
                            )
                        });
                    }
                    self.report
                        .actions
                        .push(NormalizeAction::DroppedUnpairedToolCall { call_id, name });
                }
                self.pending_calls.clear();
                Ok(())
            }
            MissingToolResultPolicy::Error => Err(Error::Normalize(format!(
                "{} assistant tool call(s) have no matching tool result",
                self.pending_calls.iter().filter(|c| !c.fulfilled).count()
            ))),
        }
    }

    fn merge_adjacent_messages(&mut self) {
        if !self.options.merge_adjacent_same_role {
            return;
        }

        let mut merged: Vec<NormalizedMessage> = Vec::with_capacity(self.messages.len());
        for message in self.messages.drain(..) {
            if let Some(last) = merged.last_mut()
                && last.role == message.role
            {
                last.parts.extend(message.parts);
                self.report.actions.push(NormalizeAction::MergedAdjacent {
                    role: last.role.clone(),
                });
            } else {
                merged.push(message);
            }
        }
        self.messages = merged;
    }

    fn join_system(&self) -> Option<String> {
        if self.system.is_empty() {
            None
        } else {
            Some(self.system.join("\n\n"))
        }
    }
}

fn text_parts_joined(parts: &[Part]) -> String {
    parts
        .iter()
        .filter_map(|part| match part {
            Part::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn normalize_non_tool_parts(parts: &[Part]) -> Vec<Part> {
    parts
        .iter()
        .filter_map(|part| match part {
            Part::Text { text, .. } => Some(Part::text(text)),
            Part::ImageUrl { .. } => Some(part.clone()),
            Part::Reasoning(_) => None,
            Part::ToolCall(call) => Some(Part::text(format!(
                "Unpaired tool call `{}` with arguments {}",
                call.name, call.arguments
            ))),
            Part::ToolResult(result) => Some(Part::text(format!(
                "Orphan tool result from `{}`: {}",
                result.name, result.content
            ))),
        })
        .collect()
}
