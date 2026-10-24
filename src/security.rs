use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{Error, RequestOptions, Result};

/// Bounds decoded provider data across all attempts of a logical request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ResponseLimits {
    /// Maximum decoded HTTP/SSE bytes, including failed attempts and reconnects.
    pub max_body_bytes: usize,
    /// Maximum SSE messages and retry fields, including replayed events.
    pub max_stream_events: usize,
    /// Maximum bytes in a single SSE line, independent of the total body cap.
    pub max_sse_line_bytes: usize,
    /// Maximum data bytes in a single SSE event.
    pub max_sse_event_bytes: usize,
    /// Retain raw provider JSON in the response. Disable to reduce memory use.
    pub retain_raw: bool,
    /// Maximum accumulated arguments for one OpenAI Chat tool call.
    pub max_tool_argument_bytes: usize,
    /// Minimum argument growth between OpenAI Chat progress snapshots.
    /// Deltas remain lossless; a final snapshot is emitted before Done.
    pub tool_progress_interval_bytes: usize,
}

impl Default for ResponseLimits {
    fn default() -> Self {
        Self {
            max_body_bytes: 16 * 1024 * 1024,
            max_stream_events: 16_384,
            max_sse_line_bytes: crate::sse::DEFAULT_MAX_LINE_BYTES,
            max_sse_event_bytes: crate::sse::DEFAULT_MAX_EVENT_BYTES,
            retain_raw: true,
            max_tool_argument_bytes: 1024 * 1024,
            tool_progress_interval_bytes: 4096,
        }
    }
}

#[derive(Clone)]
pub(crate) struct RequestBudget {
    pub deadline: Option<tokio::time::Instant>,
    pub limits: ResponseLimits,
    bytes: Arc<AtomicUsize>,
}

impl RequestBudget {
    pub fn new(options: &RequestOptions) -> Result<Self> {
        let limits = options.response_limits;
        if limits.max_body_bytes == 0
            || limits.max_stream_events == 0
            || limits.max_sse_line_bytes == 0
            || limits.max_sse_event_bytes == 0
            || limits.max_tool_argument_bytes == 0
            || limits.tool_progress_interval_bytes == 0
        {
            return Err(Error::InvalidRequest(
                "response limits must be non-zero".into(),
            ));
        }
        let deadline = options
            .timeout
            .map(|timeout| {
                tokio::time::Instant::now()
                    .checked_add(timeout)
                    .ok_or_else(|| Error::InvalidRequest("request timeout is too large".into()))
            })
            .transpose()?;
        Ok(Self {
            deadline,
            limits,
            bytes: Arc::new(AtomicUsize::new(0)),
        })
    }

    pub fn add_bytes(&self, bytes: usize) -> Result<()> {
        let mut used = self.bytes.load(Ordering::Relaxed);
        loop {
            let Some(total) = used
                .checked_add(bytes)
                .filter(|total| *total <= self.limits.max_body_bytes)
            else {
                return Err(Error::ResponseLimit {
                    resource: "decoded body bytes",
                    limit: self.limits.max_body_bytes,
                });
            };
            match self.bytes.compare_exchange_weak(
                used,
                total,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(()),
                Err(current) => used = current,
            }
        }
    }

    pub fn check_deadline(&self) -> Result<()> {
        if self
            .deadline
            .is_some_and(|deadline| tokio::time::Instant::now() >= deadline)
        {
            return Err(Error::Timeout("total request deadline".into()));
        }
        Ok(())
    }

    pub async fn run<T>(&self, future: impl Future<Output = Result<T>>) -> Result<T> {
        match self.deadline {
            Some(deadline) => {
                self.check_deadline()?;
                tokio::select! {
                    biased;
                    _ = tokio::time::sleep_until(deadline) => Err(Error::Timeout("total request deadline".into())),
                    result = future => { self.check_deadline()?; result },
                }
            }
            None => future.await,
        }
    }
}
