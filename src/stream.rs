use std::collections::{HashSet, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use bytes::Bytes;
use futures_core::Stream;

use crate::audit::{AuditContext, AuditOutcome, FirstTokenKind};
use crate::cancel::CancellationToken;
use crate::error::{Error, ErrorKind, Result};
use crate::sse::{SseError, SseEvent, SseMessage, SseParser};
use crate::types::{ChatResponse, ReasoningKind, ResponseMetadata, ServerToolState, Usage};

/// A boxed provider-neutral stream of model events.
pub type ModelStream = Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send + 'static>>;

/// Opt-in transport reconnect policy for an already-started SSE stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamReconnectPolicy {
    pub max_attempts: u32,
    pub delay: Duration,
    /// Refuse reconnect when no SSE event id has been observed.
    pub require_event_id: bool,
}

impl Default for StreamReconnectPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 2,
            delay: Duration::from_millis(250),
            require_event_id: true,
        }
    }
}

impl StreamReconnectPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn max_attempts(mut self, max_attempts: u32) -> Self {
        self.max_attempts = max_attempts;
        self
    }

    pub fn delay(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    pub fn require_event_id(mut self, require_event_id: bool) -> Self {
        self.require_event_id = require_event_id;
        self
    }
}

/// A provider-neutral streaming event.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// The HTTP response was accepted and the stream is starting.
    Start {
        provider: &'static str,
        model: String,
        metadata: ResponseMetadata,
    },
    /// Incremental assistant-visible text.
    TextDelta { text: String },
    /// Incremental reasoning summary or readable reasoning text.
    ReasoningDelta { kind: ReasoningKind, text: String },
    /// Incremental tool-call data.
    ///
    /// `index` identifies the tool call within the assistant message.
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments_delta: String,
    },
    /// The complete tool arguments accumulated so far for a call.
    ToolCallProgress {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments_so_far: String,
    },
    /// A provider-hosted or built-in tool changed state.
    ServerToolStatus {
        tool: String,
        call_id: Option<String>,
        state: ServerToolState,
    },
    /// Usage reported during the stream. Providers may send this only at the end.
    Usage { usage: Usage },
    /// A server-directed SSE retry delay.
    Retry { delay: Duration },
    /// The SDK scheduled a retry or reconnect with stable metadata.
    Retrying {
        attempt: u32,
        max_attempts: u32,
        delay: Duration,
        reason: String,
    },
    /// A recoverable stream error. Terminal errors are returned as `Err`.
    Error { error: StreamError },
    /// The stream completed and the SDK assembled a normal response.
    ///
    /// Consumers that persist conversation history should append
    /// `response.message`, not reconstruct it from deltas.
    Done {
        response: Box<ChatResponse>,
        finish_reason: Option<String>,
    },
}

/// Stable, content-free metadata for a recoverable stream error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamError {
    pub kind: ErrorKind,
    pub message: String,
    pub recoverable: bool,
}

/// Provider-specific mapper from one SSE message to zero or more events.
pub(crate) trait SseMapper: Send + Unpin + 'static {
    fn map(&mut self, message: SseMessage) -> Result<Vec<StreamEvent>>;

    /// Whether a clean EOF is a valid terminal condition for this protocol.
    fn eof_is_terminal(&self) -> bool {
        true
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>> {
        Ok(Vec::new())
    }
}

pub(crate) fn sse_stream_with_audit<M>(
    response: reqwest::Response,
    mapper: M,
    cancellation: Option<CancellationToken>,
    idle_timeout: Option<Duration>,
    audit: Option<AuditContext>,
) -> ModelStream
where
    M: SseMapper,
{
    let metadata = ResponseMetadata::from_response(&response);
    let cancellation_future = cancellation_future(cancellation.clone());
    Box::pin(SseStream {
        inner: Box::pin(response.bytes_stream()),
        parser: SseParser::new(),
        mapper,
        queue: VecDeque::new(),
        finished: false,
        terminal: false,
        last_event_id: None,
        seen_events: HashSet::new(),
        seen_order: VecDeque::new(),
        reconnect: None,
        reconnect_future: None,
        cancellation,
        cancellation_future,
        idle_timeout,
        idle_deadline: idle_deadline(idle_timeout),
        audit,
        audit_finished: false,
        audit_status: Some(metadata.status),
        audit_provider_request_id: metadata.request_id,
    })
}

type CancellationFuture = Pin<Box<dyn Future<Output = ()> + Send>>;

fn cancellation_future(cancellation: Option<CancellationToken>) -> Option<CancellationFuture> {
    cancellation.map(|cancellation| {
        Box::pin(async move {
            cancellation.cancelled().await;
        }) as CancellationFuture
    })
}

type IdleDeadline = Pin<Box<tokio::time::Sleep>>;

fn idle_deadline(timeout: Option<Duration>) -> Option<IdleDeadline> {
    timeout.map(|timeout| Box::pin(tokio::time::sleep(timeout)))
}

type ReconnectFuture = Pin<Box<dyn Future<Output = Result<reqwest::Response>> + Send>>;

trait ReconnectFactory: Send {
    fn connect(&mut self, last_event_id: Option<String>) -> ReconnectFuture;
}

impl<F> ReconnectFactory for F
where
    F: FnMut(Option<String>) -> ReconnectFuture + Send + 'static,
{
    fn connect(&mut self, last_event_id: Option<String>) -> ReconnectFuture {
        self(last_event_id)
    }
}

struct ReconnectState {
    factory: Box<dyn ReconnectFactory>,
    policy: StreamReconnectPolicy,
    attempts: u32,
}

pub(crate) fn sse_stream_with_reconnect<M, F>(
    response: reqwest::Response,
    mapper: M,
    policy: StreamReconnectPolicy,
    factory: F,
    cancellation: Option<CancellationToken>,
    idle_timeout: Option<Duration>,
    audit: Option<AuditContext>,
) -> ModelStream
where
    M: SseMapper,
    F: FnMut(Option<String>) -> ReconnectFuture + Send + 'static,
{
    let metadata = ResponseMetadata::from_response(&response);
    let cancellation_future = cancellation_future(cancellation.clone());
    Box::pin(SseStream {
        inner: Box::pin(response.bytes_stream()),
        parser: SseParser::new(),
        mapper,
        queue: VecDeque::new(),
        finished: false,
        terminal: false,
        last_event_id: None,
        seen_events: HashSet::new(),
        seen_order: VecDeque::new(),
        reconnect: Some(ReconnectState {
            factory: Box::new(factory),
            policy,
            attempts: 0,
        }),
        reconnect_future: None,
        cancellation,
        cancellation_future,
        idle_timeout,
        idle_deadline: idle_deadline(idle_timeout),
        audit,
        audit_finished: false,
        audit_status: Some(metadata.status),
        audit_provider_request_id: metadata.request_id,
    })
}

struct SseStream<M> {
    inner: Pin<Box<dyn Stream<Item = std::result::Result<Bytes, reqwest::Error>> + Send>>,
    parser: SseParser,
    mapper: M,
    queue: VecDeque<Result<StreamEvent>>,
    finished: bool,
    terminal: bool,
    last_event_id: Option<String>,
    seen_events: HashSet<(String, String)>,
    seen_order: VecDeque<(String, String)>,
    reconnect: Option<ReconnectState>,
    reconnect_future: Option<ReconnectFuture>,
    cancellation: Option<CancellationToken>,
    cancellation_future: Option<CancellationFuture>,
    idle_timeout: Option<Duration>,
    idle_deadline: Option<IdleDeadline>,
    audit: Option<AuditContext>,
    audit_finished: bool,
    audit_status: Option<u16>,
    audit_provider_request_id: Option<String>,
}

const MAX_SEEN_EVENTS: usize = 4096;

impl<M> SseStream<M>
where
    M: SseMapper,
{
    fn finish_audit(&mut self, error: Option<&Error>, usage: Option<Usage>) {
        if self.audit_finished {
            return;
        }
        let Some(audit) = &self.audit else {
            return;
        };
        self.audit_finished = true;
        let outcome = match error {
            Some(Error::Cancelled) => AuditOutcome::Cancelled,
            Some(_) => AuditOutcome::Failure,
            None => AuditOutcome::Success,
        };
        audit.request_finished_with_request_id(
            outcome,
            self.audit_status,
            self.audit_provider_request_id.clone(),
            error,
            usage,
        );
    }

    fn enqueue_events(&mut self, events: Vec<StreamEvent>) {
        for event in events {
            if let Some(audit) = &self.audit {
                match &event {
                    StreamEvent::TextDelta { .. } => audit.first_token(FirstTokenKind::Text),
                    StreamEvent::ReasoningDelta { .. } => {
                        audit.first_token(FirstTokenKind::Reasoning);
                    }
                    StreamEvent::ToolCallDelta { .. } | StreamEvent::ToolCallProgress { .. } => {
                        audit.first_token(FirstTokenKind::ToolCall);
                    }
                    _ => {}
                }
            }
            if let StreamEvent::Done { response, .. } = &event {
                self.terminal = true;
                self.finish_audit(None, response.usage.clone());
            }
            self.queue.push_back(Ok(event));
        }
    }

    fn process_sse(&mut self, event: SseEvent) {
        match event {
            SseEvent::Retry(delay) => self.queue.push_back(Ok(StreamEvent::Retry { delay })),
            SseEvent::Message(message) => {
                if let Some(id) = &message.id {
                    self.last_event_id = Some(id.clone());
                    let key = (id.clone(), message.data.clone());
                    if self.seen_events.contains(&key) {
                        return;
                    }
                    self.seen_events.insert(key.clone());
                    self.seen_order.push_back(key);
                    while self.seen_order.len() > MAX_SEEN_EVENTS {
                        if let Some(oldest) = self.seen_order.pop_front() {
                            self.seen_events.remove(&oldest);
                        }
                    }
                }

                match self.mapper.map(message) {
                    Ok(events) => self.enqueue_events(events),
                    Err(error) => {
                        self.finish_audit(Some(&error), None);
                        self.queue.push_back(Err(error));
                        self.finished = true;
                    }
                }
            }
        }
    }

    fn process_sse_result(&mut self, result: std::result::Result<Vec<SseEvent>, SseError>) {
        match result {
            Ok(events) => {
                for event in events {
                    self.process_sse(event);
                }
            }
            Err(error) => {
                let error = Error::StreamProtocol(error.to_string());
                self.finish_audit(Some(&error), None);
                self.queue.push_back(Err(error));
                self.finished = true;
            }
        }
    }

    fn finish_stream(&mut self) {
        self.finished = true;

        match self.parser.finish() {
            Ok(events) => {
                for event in events {
                    self.process_sse(event);
                }
            }
            Err(error) => {
                let error = Error::StreamProtocol(error.to_string());
                self.finish_audit(Some(&error), None);
                self.queue.push_back(Err(error));
                return;
            }
        }

        if self.terminal || self.queue.iter().any(Result::is_err) {
            return;
        }

        match self.mapper.finish() {
            Ok(events) => self.enqueue_events(events),
            Err(error) => {
                self.finish_audit(Some(&error), None);
                self.queue.push_back(Err(error));
            }
        }

        if !self.terminal && !self.queue.iter().any(Result::is_err) {
            let error =
                Error::StreamProtocol("stream ended without a provider terminal event".to_string());
            self.finish_audit(Some(&error), None);
            self.queue.push_back(Err(error));
        }
    }

    fn schedule_reconnect(&mut self, delay_hint: Option<Duration>) -> bool {
        if self.terminal || self.finished {
            return false;
        }
        let Some(reconnect) = self.reconnect.as_mut() else {
            return false;
        };
        if reconnect.attempts >= reconnect.policy.max_attempts {
            return false;
        }
        if reconnect.policy.require_event_id && self.last_event_id.is_none() {
            return false;
        }

        let last_event_id = self.last_event_id.clone();
        let delay = delay_hint
            .or_else(|| self.parser.retry())
            .unwrap_or(reconnect.policy.delay);
        reconnect.attempts += 1;
        if let Some(audit) = &self.audit {
            audit.stream_reconnect_scheduled(reconnect.attempts, delay, last_event_id.is_some());
        }
        let future = reconnect.factory.connect(last_event_id);
        self.queue.push_back(Ok(StreamEvent::Retry { delay }));
        self.queue.push_back(Ok(StreamEvent::Retrying {
            attempt: reconnect.attempts,
            max_attempts: reconnect.policy.max_attempts,
            delay,
            reason: "stream reconnect".to_string(),
        }));
        self.idle_deadline = None;
        let cancellation = self.cancellation.clone();
        self.reconnect_future = Some(Box::pin(async move {
            if let Some(cancellation) = cancellation {
                tokio::select! {
                    biased;
                    _ = cancellation.cancelled() => Err(Error::Cancelled),
                    _ = tokio::time::sleep(delay) => future.await,
                }
            } else {
                tokio::time::sleep(delay).await;
                future.await
            }
        }));
        true
    }

    fn cancel(&mut self) {
        let error = Error::Cancelled;
        self.finish_audit(Some(&error), None);
        self.queue.push_back(Err(error));
        self.finished = true;
    }

    fn reset_idle_deadline(&mut self) {
        self.idle_deadline = idle_deadline(self.idle_timeout);
    }

    fn handle_idle_timeout(&mut self) {
        if self.schedule_reconnect(None) {
            self.idle_deadline = None;
            return;
        }
        let error = Error::Timeout("stream idle timeout".to_string());
        self.finish_audit(Some(&error), None);
        self.queue.push_back(Err(error));
        self.finished = true;
    }

    fn handle_transport_error(&mut self, error: reqwest::Error) {
        if self.schedule_reconnect(None) {
            return;
        }
        let error = Error::Http(error);
        self.finish_audit(Some(&error), None);
        self.queue.push_back(Err(error));
        self.finished = true;
    }
}

impl<M> Drop for SseStream<M> {
    fn drop(&mut self) {
        if self.audit_finished {
            return;
        }
        let Some(audit) = &self.audit else {
            return;
        };
        audit.request_finished_with_request_id(
            AuditOutcome::Cancelled,
            self.audit_status,
            self.audit_provider_request_id.clone(),
            None,
            None,
        );
        self.audit_finished = true;
    }
}

impl<M> Stream for SseStream<M>
where
    M: SseMapper,
{
    type Item = Result<StreamEvent>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(event) = this.queue.pop_front() {
                return Poll::Ready(Some(event));
            }

            if this.terminal {
                this.finished = true;
                return Poll::Ready(None);
            }

            if this.finished {
                return Poll::Ready(None);
            }

            if let Some(mut future) = this.cancellation_future.take() {
                match future.as_mut().poll(cx) {
                    Poll::Pending => {
                        this.cancellation_future = Some(future);
                    }
                    Poll::Ready(()) => {
                        this.cancel();
                        continue;
                    }
                }
            }

            if this.reconnect_future.is_none()
                && let Some(mut deadline) = this.idle_deadline.take()
            {
                match deadline.as_mut().poll(cx) {
                    Poll::Pending => {
                        this.idle_deadline = Some(deadline);
                    }
                    Poll::Ready(()) => {
                        this.handle_idle_timeout();
                        continue;
                    }
                }
            }

            if let Some(mut future) = this.reconnect_future.take() {
                match future.as_mut().poll(cx) {
                    Poll::Pending => {
                        this.reconnect_future = Some(future);
                        return Poll::Pending;
                    }
                    Poll::Ready(Ok(response)) => {
                        this.inner = Box::pin(response.bytes_stream());
                        this.parser.reset_for_reconnect();
                        this.reset_idle_deadline();
                        continue;
                    }
                    Poll::Ready(Err(error)) => {
                        if this.schedule_reconnect(error.retry_after()) {
                            continue;
                        }
                        this.finish_audit(Some(&error), None);
                        this.queue.push_back(Err(error));
                        this.finished = true;
                        continue;
                    }
                }
            }

            match this.inner.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Ok(bytes))) => {
                    if let Some(audit) = &this.audit {
                        audit.add_response_bytes(bytes.len());
                    }
                    this.reset_idle_deadline();
                    let result = this.parser.push(&bytes);
                    this.process_sse_result(result);
                }
                Poll::Ready(Some(Err(error))) => this.handle_transport_error(error),
                Poll::Ready(None) => {
                    if !this.terminal
                        && !this.mapper.eof_is_terminal()
                        && this.schedule_reconnect(None)
                    {
                        continue;
                    }
                    this.finish_stream();
                }
            }
        }
    }
}

/// Await the next stream event without requiring the caller to import
/// `futures_util::StreamExt`.
pub async fn next_event(stream: &mut ModelStream) -> Option<Result<StreamEvent>> {
    use std::future::poll_fn;

    poll_fn(|cx| stream.as_mut().poll_next(cx)).await
}

/// Collect a stream into the final response while preserving all events.
///
/// This is mainly useful for tests and for applications that intentionally do
/// not render tokens incrementally.
pub async fn collect_stream(mut stream: ModelStream) -> Result<ChatResponse> {
    while let Some(event) = next_event(&mut stream).await {
        if let StreamEvent::Done { response, .. } = event? {
            return Ok(*response);
        }
    }

    Err(Error::StreamProtocol(
        "stream ended without a Done event".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    struct TestMapper {
        text: String,
    }

    impl SseMapper for TestMapper {
        fn map(&mut self, message: SseMessage) -> Result<Vec<StreamEvent>> {
            self.text.push_str(&message.data);
            Ok(vec![StreamEvent::TextDelta { text: message.data }])
        }

        fn finish(&mut self) -> Result<Vec<StreamEvent>> {
            Ok(vec![StreamEvent::Done {
                response: Box::new(ChatResponse::new(Message::assistant(&self.text))),
                finish_reason: Some("stop".to_string()),
            }])
        }
    }

    #[test]
    fn sse_stream_handles_split_cjk_and_emits_done() {
        let payload = "retry: 10\ndata: 你好\n\n";
        let split = payload.find("你").expect("CJK marker exists") + "你".len() - 1;
        let chunks = vec![
            Ok::<_, reqwest::Error>(Bytes::copy_from_slice(&payload.as_bytes()[..split])),
            Ok::<_, reqwest::Error>(Bytes::copy_from_slice(&payload.as_bytes()[split..])),
        ];
        let inner: Pin<Box<dyn Stream<Item = std::result::Result<Bytes, reqwest::Error>> + Send>> =
            Box::pin(futures_util::stream::iter(chunks));
        let stream: ModelStream = Box::pin(SseStream {
            inner,
            parser: SseParser::new(),
            mapper: TestMapper {
                text: String::new(),
            },
            queue: VecDeque::new(),
            finished: false,
            terminal: false,
            last_event_id: None,
            seen_events: HashSet::new(),
            seen_order: VecDeque::new(),
            reconnect: None,
            reconnect_future: None,
            cancellation: None,
            cancellation_future: None,
            idle_timeout: None,
            idle_deadline: None,
            audit: None,
            audit_finished: false,
            audit_status: None,
            audit_provider_request_id: None,
        });

        let response = futures_executor::block_on(collect_stream(stream)).unwrap();
        assert_eq!(response.text(), "你好");
    }
}
