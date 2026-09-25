use crate::error::Result;
use crate::types::ChatRequest;

/// A synchronous, provider-neutral request transform.
///
/// Transforms run once per logical request before capability gating and
/// normalization. Retries and stream reconnects reuse the transformed request.
pub trait RequestTransform: Send + Sync {
    fn transform(&self, request: &ChatRequest) -> Result<ChatRequest>;
}
