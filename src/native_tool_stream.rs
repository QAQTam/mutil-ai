//! Protocol-level accumulation for one native custom tool call.
//!
//! This module is independent of the Responses adapter until it is integrated
//! with terminal response reconciliation and result pairing.

use crate::{
    Error, NativeToolCall, NativeToolCompletion, NativeToolKind, NativeToolPayload, Result,
};

/// Identity fields as they become available across streamed frames.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeToolIdentity {
    pub namespace: Option<String>,
    pub name: Option<String>,
    pub call_id: Option<String>,
    pub item_id: Option<String>,
}

/// Accumulates a single custom call; deltas are fragments and done inputs are
/// authoritative snapshots. Identical completion snapshots are idempotent.
///
/// The caller must route events by response and output index and deduplicate
/// repeated delta frames using transport event identity when available.
#[derive(Debug, Default)]
pub struct CustomToolAccumulator {
    identity: NativeToolIdentity,
    input: String,
    snapshot: Option<String>,
    item_completed: bool,
    interrupted: bool,
    invalid: bool,
}

impl CustomToolAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Merge identity from a frame without replacing an existing value.
    pub fn merge_identity(&mut self, identity: &NativeToolIdentity) -> Result<()> {
        fn merge(target: &mut Option<String>, incoming: &Option<String>) -> Result<()> {
            if let Some(value) = incoming {
                if value.is_empty() || target.as_ref().is_some_and(|old| old != value) {
                    return Err(Error::StreamProtocol("tool_identity_conflict".into()));
                }
                *target = Some(value.clone());
            }
            Ok(())
        }
        // Apply transactionally so a failed frame cannot partially mutate identity.
        let mut merged = self.identity.clone();
        let result = (|| {
            merge(&mut merged.namespace, &identity.namespace)?;
            merge(&mut merged.name, &identity.name)?;
            merge(&mut merged.call_id, &identity.call_id)?;
            merge(&mut merged.item_id, &identity.item_id)
        })();
        if result.is_err() {
            self.invalid = true;
        }
        result?;
        self.identity = merged;
        Ok(())
    }

    /// Append a delta fragment before a completion snapshot.
    pub fn push_delta(&mut self, delta: &str) -> Result<()> {
        if self.snapshot.is_some() || self.interrupted {
            self.invalid = true;
            return Err(Error::StreamProtocol("tool_payload_conflict".into()));
        }
        self.input.push_str(delta);
        Ok(())
    }

    /// Merge a full input snapshot. A partially delivered delta sequence may be
    /// a prefix of the snapshot; disagreement is always an error.
    pub fn finish_input(&mut self, snapshot: &str) -> Result<()> {
        if self.interrupted
            || self.snapshot.as_ref().is_some_and(|old| old != snapshot)
            || !snapshot.starts_with(&self.input)
        {
            self.invalid = true;
            return Err(Error::StreamProtocol("tool_payload_conflict".into()));
        }
        self.input = snapshot.to_owned();
        self.snapshot = Some(snapshot.to_owned());
        Ok(())
    }

    /// Record an authoritative completed item after validating its full input.
    /// An input-done frame alone does not establish that the item completed.
    pub fn finish_item(&mut self, snapshot: &str) -> Result<()> {
        self.finish_input(snapshot)?;
        self.item_completed = true;
        Ok(())
    }

    /// Mark an unfinished call interrupted; an already completed item remains
    /// a completed fact, leaving cancellation scheduling to the caller.
    pub fn interrupt(&mut self) {
        if !self.item_completed {
            self.interrupted = true;
        }
    }

    /// Snapshot of accumulated raw input, including incomplete input.
    pub fn input(&self) -> &str {
        &self.input
    }

    /// Whether a completed item snapshot was observed.
    pub fn is_completed(&self) -> bool {
        self.item_completed && !self.invalid
    }

    /// Obtain a completed call only with a full snapshot and real identity.
    pub fn completed_call(&self) -> Result<NativeToolCall> {
        if self.invalid {
            return Err(Error::StreamProtocol("tool_call_invalid".into()));
        }
        if self.interrupted || !self.item_completed {
            return Err(Error::StreamProtocol("tool_call_incomplete".into()));
        }
        let call = NativeToolCall {
            kind: NativeToolKind::Custom,
            namespace: self.identity.namespace.clone(),
            name: self.identity.name.clone().unwrap_or_default(),
            call_id: self.identity.call_id.clone().unwrap_or_default(),
            item_id: self.identity.item_id.clone(),
            payload: NativeToolPayload::RawText(self.input.clone()),
            completion: NativeToolCompletion::Completed,
            provider_state: None,
        };
        call.validate_completed()?;
        Ok(call)
    }
}
