use serde::{Deserialize, Serialize};

use crate::normalize::NormalizeStats;

/// A stable summary of transformations applied while preparing or decoding a
/// model request.
///
/// Reports never contain message, reasoning, tool argument, or tool result
/// content.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletionReport {
    /// Whether a [`crate::transform::RequestTransform`] was applied to the
    /// request.
    pub transform_applied: bool,
    /// Normalization repairs applied to the conversation model.
    pub normalization: NormalizeStats,
    /// Lossy provider-wire transformations.
    pub wire: WireReport,
}

impl CompletionReport {
    /// Creates a report carrying only normalization statistics.
    pub fn from_normalization(normalization: NormalizeStats) -> Self {
        Self {
            normalization,
            ..Self::default()
        }
    }

    /// Returns `true` when no transform, normalization repair, or wire
    /// action was recorded.
    pub fn is_clean(&self) -> bool {
        !self.transform_applied && self.normalization.is_clean() && self.wire.actions.is_empty()
    }
}

/// Provider-wire transformations that could not be represented losslessly.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireReport {
    /// Individual lossy transformations, in the order they occurred.
    pub actions: Vec<WireAction>,
}

impl WireReport {
    /// Records a wire action.
    pub fn push(&mut self, action: WireAction) {
        self.actions.push(action);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WireAction {
    /// A feature was not representable on the wire and was dropped.
    Unsupported {
        /// The dropped feature.
        feature: WireFeature,
        /// How many items were affected.
        count: u32,
    },
    /// A feature was sent in a weaker, lossy form.
    Downgraded {
        /// The original feature.
        feature: WireFeature,
        /// Short description of the fallback used.
        to: String,
        /// How many items were affected.
        count: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WireFeature {
    /// Structured tool-result parts.
    StructuredToolResult,
    /// Image references inside tool results or messages.
    ImageReference,
    /// Provider-hosted tools.
    ServerTool,
    /// Opaque provider-hosted tool items.
    ProviderItem,
    /// A request-level field.
    RequestField,
}
