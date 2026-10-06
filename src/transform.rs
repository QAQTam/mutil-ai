use crate::error::Result;
use crate::types::ChatRequest;

/// A synchronous, provider-neutral request transform.
///
/// Transforms run once per logical request before capability gating and
/// normalization. Retries and stream reconnects reuse the transformed request.
pub trait RequestTransform: Send + Sync {
    /// Applies the transform to a request.
    ///
    /// Implementations should be pure and deterministic: the result is reused
    /// across retries and stream reconnects.
    ///
    /// # Errors
    /// Returns an error when the request cannot be transformed; the request is
    /// then not sent at all.
    fn transform(&self, request: &ChatRequest) -> Result<ChatRequest>;
}
