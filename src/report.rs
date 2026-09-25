use serde::{Deserialize, Serialize};

use crate::normalize::NormalizeStats;

/// A stable summary of transformations applied while preparing or decoding a
/// model request.
///
/// Reports never contain message, reasoning, tool argument, or tool result
/// content.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletionReport {
    pub transform_applied: bool,
    pub normalization: NormalizeStats,
    pub wire: WireReport,
}

impl CompletionReport {
    pub fn from_normalization(normalization: NormalizeStats) -> Self {
        Self {
            normalization,
            ..Self::default()
        }
    }

    pub fn is_clean(&self) -> bool {
        !self.transform_applied && self.normalization.is_clean() && self.wire.actions.is_empty()
    }
}

/// Provider-wire transformations that could not be represented losslessly.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireReport {
    pub actions: Vec<WireAction>,
}

impl WireReport {
    pub fn push(&mut self, action: WireAction) {
        self.actions.push(action);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WireAction {
    Unsupported {
        feature: WireFeature,
        count: u32,
    },
    Downgraded {
        feature: WireFeature,
        to: String,
        count: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WireFeature {
    StructuredToolResult,
    ImageReference,
    ServerTool,
    ProviderItem,
    RequestField,
}
