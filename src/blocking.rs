use std::sync::Arc;

use tokio::runtime::{Handle, Runtime};

use crate::adapter::ModelAdapter;
use crate::error::{Error, Result};
use crate::headers::RequestOptions;
use crate::stream::{ModelStream, StreamEvent, next_event};
use crate::types::{ChatRequest, ChatResponse};

/// Blocking wrapper for an async [`ModelAdapter`].
///
/// Construction and calls fail when invoked from inside a Tokio runtime. Use
/// this wrapper only from synchronous runtimes or dedicated worker threads.
pub struct BlockingAdapter<A> {
    inner: A,
    runtime: Arc<Runtime>,
}

impl<A> BlockingAdapter<A>
where
    A: ModelAdapter,
{
    /// Creates a blocking adapter around `inner`, backed by a dedicated
    /// single-threaded Tokio runtime.
    ///
    /// # Errors
    ///
    /// Returns an error when called from inside a Tokio runtime, or when the
    /// blocking runtime cannot be created.
    pub fn new(inner: A) -> Result<Self> {
        ensure_sync_context()?;
        let runtime = Runtime::new().map_err(|error| {
            Error::InvalidRequest(format!("failed to create blocking runtime: {error}"))
        })?;
        Ok(Self {
            inner,
            runtime: Arc::new(runtime),
        })
    }

    /// Borrows the wrapped async adapter.
    pub fn inner(&self) -> &A {
        &self.inner
    }

    /// Consumes the wrapper and returns the wrapped async adapter.
    pub fn into_inner(self) -> A {
        self.inner
    }

    /// Blocking version of [`ModelAdapter::complete`].
    ///
    /// # Errors
    ///
    /// Returns an error when called from inside a Tokio runtime, or the
    /// underlying adapter error.
    pub fn complete(&self, request: &ChatRequest) -> Result<ChatResponse> {
        ensure_sync_context()?;
        self.runtime.block_on(self.inner.complete(request))
    }

    /// Blocking version of [`ModelAdapter::complete_with`].
    ///
    /// # Errors
    ///
    /// Returns an error when called from inside a Tokio runtime, or the
    /// underlying adapter error.
    pub fn complete_with(
        &self,
        request: &ChatRequest,
        options: &RequestOptions,
    ) -> Result<ChatResponse> {
        ensure_sync_context()?;
        self.runtime
            .block_on(self.inner.complete_with(request, options))
    }

    /// Blocking version of [`ModelAdapter::stream`].
    ///
    /// # Errors
    ///
    /// Returns an error when called from inside a Tokio runtime, or the
    /// underlying adapter error.
    pub fn stream(&self, request: &ChatRequest) -> Result<BlockingStream> {
        ensure_sync_context()?;
        let stream = self.runtime.block_on(self.inner.stream(request))?;
        Ok(BlockingStream {
            runtime: self.runtime.clone(),
            stream,
        })
    }

    /// Blocking version of [`ModelAdapter::stream_with`].
    ///
    /// # Errors
    ///
    /// Returns an error when called from inside a Tokio runtime, or the
    /// underlying adapter error.
    pub fn stream_with(
        &self,
        request: &ChatRequest,
        options: &RequestOptions,
    ) -> Result<BlockingStream> {
        ensure_sync_context()?;
        let stream = self
            .runtime
            .block_on(self.inner.stream_with(request, options))?;
        Ok(BlockingStream {
            runtime: self.runtime.clone(),
            stream,
        })
    }
}

/// Blocking iterator over a model stream.
pub struct BlockingStream {
    runtime: Arc<Runtime>,
    stream: ModelStream,
}

impl BlockingStream {
    /// Blocks until the next stream event is available, returning `None`
    /// when the stream has ended.
    ///
    /// Calling this from inside a Tokio runtime yields a single
    /// `Err(Error::InvalidRequest)` item instead of panicking.
    pub fn next_event(&mut self) -> Option<Result<StreamEvent>> {
        if let Err(error) = ensure_sync_context() {
            return Some(Err(error));
        }
        self.runtime.block_on(next_event(&mut self.stream))
    }
}

impl Iterator for BlockingStream {
    type Item = Result<StreamEvent>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_event()
    }
}

fn ensure_sync_context() -> Result<()> {
    if Handle::try_current().is_ok() {
        Err(Error::InvalidRequest(
            "blocking adapter cannot be called from an async Tokio runtime".to_string(),
        ))
    } else {
        Ok(())
    }
}
