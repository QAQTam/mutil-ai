use std::borrow::Cow;
use std::time::Duration;

use thiserror::Error;

const DEFAULT_MAX_LINE_BYTES: usize = 64 * 1024;
const DEFAULT_MAX_EVENT_BYTES: usize = 1024 * 1024;
const UTF8_BOM: [u8; 3] = [0xEF, 0xBB, 0xBF];

/// A parsed SSE message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseMessage {
    /// The `event` field, or `None` when the event used the default type.
    pub event: Option<String>,
    /// The `data` field, with multiple data lines joined by `\n`.
    pub data: String,
    /// The most recent `id` field seen before this message.
    pub id: Option<String>,
}

/// An event emitted by [`SseParser`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SseEvent {
    /// A complete SSE message, dispatched on a blank line.
    Message(SseMessage),
    /// A server-directed `retry` reconnection delay.
    Retry(Duration),
}

/// How invalid UTF-8 in field values is handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SseUtf8Policy {
    /// WHATWG EventSource behavior: invalid UTF-8 becomes U+FFFD.
    Replace,
    /// Stricter provider-client behavior: invalid UTF-8 is an error.
    Strict,
}

/// Errors that can occur while parsing an SSE stream.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SseError {
    /// Invalid UTF-8 was encountered under [`SseUtf8Policy::Strict`].
    #[error("invalid UTF-8 in an SSE field (strict mode)")]
    Utf8,

    /// A single line exceeded the configured byte limit.
    #[error("SSE line exceeded {limit} bytes")]
    LineTooLong { limit: usize },

    /// A single event exceeded the configured byte limit.
    #[error("SSE event exceeded {limit} bytes")]
    EventTooLarge { limit: usize },
}

/// Incremental, zero-I/O SSE parser.
///
/// Feed arbitrary byte chunks with [`SseParser::push`]. It does not assume
/// chunks end on lines or events. A final incomplete event is discarded by
/// [`SseParser::finish`], matching the usual EventSource behavior.
pub struct SseParser {
    line: Vec<u8>,
    pending_cr: bool,
    bom_checked: bool,
    bom_prefix: [u8; UTF8_BOM.len()],
    bom_prefix_len: u8,

    event_type: Option<String>,
    data: Vec<u8>,
    last_event_id: Option<String>,
    retry: Option<Duration>,

    max_line_bytes: usize,
    max_event_bytes: usize,
    utf8_policy: SseUtf8Policy,
}

impl Default for SseParser {
    fn default() -> Self {
        Self::new()
    }
}

impl SseParser {
    /// Creates a parser with default limits: 64 KiB per line and 1 MiB per
    /// event, using [`SseUtf8Policy::Replace`].
    pub fn new() -> Self {
        Self::with_limits(DEFAULT_MAX_LINE_BYTES, DEFAULT_MAX_EVENT_BYTES)
    }

    /// Strict UTF-8 behavior matching `sse-stream` rather than EventSource.
    pub fn strict() -> Self {
        Self::new().with_utf8_policy(SseUtf8Policy::Strict)
    }

    /// Creates a parser with the given per-line and per-event byte limits.
    ///
    /// A limit of `0` rejects any non-empty line or data field.
    pub fn with_limits(max_line_bytes: usize, max_event_bytes: usize) -> Self {
        Self {
            line: Vec::with_capacity(256),
            pending_cr: false,
            bom_checked: false,
            bom_prefix: [0; UTF8_BOM.len()],
            bom_prefix_len: 0,
            event_type: None,
            data: Vec::with_capacity(1024),
            last_event_id: None,
            retry: None,
            max_line_bytes,
            max_event_bytes,
            utf8_policy: SseUtf8Policy::Replace,
        }
    }

    /// Sets how invalid UTF-8 in field values is handled.
    pub fn with_utf8_policy(mut self, utf8_policy: SseUtf8Policy) -> Self {
        self.utf8_policy = utf8_policy;
        self
    }

    /// Start a reconnected parser from the last event id seen on the previous
    /// connection.
    pub fn with_last_event_id(mut self, last_event_id: impl Into<String>) -> Self {
        self.last_event_id = Some(last_event_id.into());
        self
    }

    /// Returns the last event id seen on this connection, if any.
    pub fn last_event_id(&self) -> Option<&str> {
        self.last_event_id.as_deref()
    }

    /// Returns the reconnect delay most recently requested by a `retry` field.
    pub fn retry(&self) -> Option<Duration> {
        self.retry
    }

    /// Clears all state, including [`last_event_id`](Self::last_event_id) and
    /// [`retry`](Self::retry).
    pub fn reset(&mut self) {
        self.clear_pending();
        self.last_event_id = None;
        self.retry = None;
    }

    /// Clear partial connection state while preserving Last-Event-ID and the
    /// server retry hint.
    pub fn reset_for_reconnect(&mut self) {
        self.clear_pending();
    }

    fn clear_pending(&mut self) {
        self.line.clear();
        self.pending_cr = false;
        self.bom_checked = false;
        self.bom_prefix_len = 0;
        self.event_type = None;
        self.data.clear();
    }

    /// Feed one arbitrary byte chunk and return all complete events.
    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>, SseError> {
        let mut events = Vec::new();
        self.push_into(chunk, &mut events)?;
        Ok(events)
    }

    /// Feed one arbitrary byte chunk into an existing event buffer.
    ///
    /// Reusing the same buffer across chunks avoids one allocation per chunk.
    pub fn push_into(&mut self, chunk: &[u8], events: &mut Vec<SseEvent>) -> Result<(), SseError> {
        let mut input = chunk;

        if !self.bom_checked {
            let prefix_len = usize::from(self.bom_prefix_len);
            let needed = UTF8_BOM.len().saturating_sub(prefix_len);
            let take = needed.min(input.len());
            self.bom_prefix[prefix_len..prefix_len + take].copy_from_slice(&input[..take]);
            self.bom_prefix_len += take as u8;
            input = &input[take..];

            let prefix_len = usize::from(self.bom_prefix_len);
            if prefix_len < UTF8_BOM.len() && UTF8_BOM.starts_with(&self.bom_prefix[..prefix_len]) {
                return Ok(());
            }

            self.bom_checked = true;
            if self.bom_prefix[..prefix_len] != UTF8_BOM {
                let mut prefix = [0_u8; UTF8_BOM.len()];
                prefix[..prefix_len].copy_from_slice(&self.bom_prefix[..prefix_len]);
                self.feed_bytes(&prefix[..prefix_len], events)?;
            }
        }

        self.feed_bytes(input, events)
    }

    /// Finish the stream. An incomplete final event is discarded.
    pub fn finish(&mut self) -> Result<Vec<SseEvent>, SseError> {
        let mut events = Vec::new();
        self.finish_into(&mut events)?;
        Ok(events)
    }

    /// Finish the stream into an existing event buffer.
    pub fn finish_into(&mut self, events: &mut Vec<SseEvent>) -> Result<(), SseError> {
        if !self.bom_checked {
            self.bom_checked = true;
            let prefix_len = usize::from(self.bom_prefix_len);
            if prefix_len != 0 && self.bom_prefix[..prefix_len] != UTF8_BOM {
                let mut prefix = [0_u8; UTF8_BOM.len()];
                prefix[..prefix_len].copy_from_slice(&self.bom_prefix[..prefix_len]);
                self.feed_bytes(&prefix[..prefix_len], events)?;
            }
        }

        if self.pending_cr {
            self.pending_cr = false;
        }

        // A non-empty line without a following blank line is an incomplete
        // event. EventSource implementations discard it at EOF.
        self.line.clear();
        self.event_type = None;
        self.data.clear();
        self.pending_cr = false;

        Ok(())
    }

    fn feed_bytes(&mut self, bytes: &[u8], events: &mut Vec<SseEvent>) -> Result<(), SseError> {
        let mut input = bytes;

        loop {
            if self.pending_cr {
                self.pending_cr = false;
                if input.first() == Some(&b'\n') {
                    input = &input[1..];
                }
            }

            let Some(delimiter) = input.iter().position(|byte| matches!(*byte, b'\r' | b'\n'))
            else {
                self.push_line_bytes(input)?;
                return Ok(());
            };

            let delimiter_byte = input[delimiter];
            let segment = &input[..delimiter];
            input = &input[delimiter + 1..];
            self.finish_line_segment(segment, events)?;

            if delimiter_byte == b'\r' {
                if input.first() == Some(&b'\n') {
                    input = &input[1..];
                } else if input.is_empty() {
                    self.pending_cr = true;
                    return Ok(());
                }
            }

            if input.is_empty() {
                return Ok(());
            }
        }
    }

    fn finish_line_segment(
        &mut self,
        segment: &[u8],
        events: &mut Vec<SseEvent>,
    ) -> Result<(), SseError> {
        if self.line.is_empty() {
            if segment.is_empty() {
                self.dispatch(events);
                return Ok(());
            }
            return self.process_line(segment, events);
        }

        self.push_line_bytes(segment)?;
        self.end_line(events)
    }

    fn push_line_bytes(&mut self, value: &[u8]) -> Result<(), SseError> {
        let next_len = self
            .line
            .len()
            .checked_add(value.len())
            .ok_or(SseError::LineTooLong {
                limit: self.max_line_bytes,
            })?;
        if next_len > self.max_line_bytes {
            return Err(SseError::LineTooLong {
                limit: self.max_line_bytes,
            });
        }
        self.line.extend_from_slice(value);
        Ok(())
    }

    fn end_line(&mut self, events: &mut Vec<SseEvent>) -> Result<(), SseError> {
        if self.line.is_empty() {
            self.dispatch(events);
            return Ok(());
        }

        let mut line = std::mem::take(&mut self.line);
        let result = self.process_line(&line, events);
        line.clear();
        self.line = line;
        result
    }

    fn process_line(&mut self, line: &[u8], events: &mut Vec<SseEvent>) -> Result<(), SseError> {
        if line.first() == Some(&b':') {
            return Ok(());
        }

        let (field, value) = split_field(line);

        match field {
            b"data" => self.append_data(value)?,
            b"event" => {
                self.event_type = Some(self.decode(value)?.into_owned());
            }
            b"id" => {
                if !value.contains(&0) {
                    self.last_event_id = Some(self.decode(value)?.into_owned());
                }
            }
            b"retry" => {
                if let Some(millis) = parse_retry(value) {
                    let retry = Duration::from_millis(millis);
                    self.retry = Some(retry);
                    events.push(SseEvent::Retry(retry));
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn append_data(&mut self, value: &[u8]) -> Result<(), SseError> {
        let value = self.decode(value)?;
        let next_len = self
            .data
            .len()
            .checked_add(value.len())
            .and_then(|len| len.checked_add(1))
            .ok_or(SseError::EventTooLarge {
                limit: self.max_event_bytes,
            })?;

        if next_len > self.max_event_bytes {
            return Err(SseError::EventTooLarge {
                limit: self.max_event_bytes,
            });
        }

        self.data.extend_from_slice(value.as_bytes());
        self.data.push(b'\n');
        Ok(())
    }

    fn decode<'a>(&self, value: &'a [u8]) -> Result<Cow<'a, str>, SseError> {
        match self.utf8_policy {
            SseUtf8Policy::Replace => Ok(String::from_utf8_lossy(value)),
            SseUtf8Policy::Strict => std::str::from_utf8(value)
                .map(Cow::Borrowed)
                .map_err(|_| SseError::Utf8),
        }
    }

    fn dispatch(&mut self, events: &mut Vec<SseEvent>) {
        if self.data.is_empty() {
            self.event_type = None;
            return;
        }

        self.data.pop(); // trailing LF added by append_data
        let Ok(data) = String::from_utf8(std::mem::take(&mut self.data)) else {
            // append_data validated every line, so this is unreachable.
            self.event_type = None;
            return;
        };

        events.push(SseEvent::Message(SseMessage {
            event: self.event_type.take(),
            data,
            id: self.last_event_id.clone(),
        }));
    }
}

fn split_field(line: &[u8]) -> (&[u8], &[u8]) {
    let Some(colon) = line.iter().position(|byte| *byte == b':') else {
        return (line, &[]);
    };

    let mut value = &line[colon + 1..];
    if value.first() == Some(&b' ') {
        value = &value[1..];
    }
    (&line[..colon], value)
}

fn parse_retry(value: &[u8]) -> Option<u64> {
    if value.is_empty() || !value.iter().all(u8::is_ascii_digit) {
        return None;
    }

    let mut result = 0_u64;
    for byte in value {
        result = result
            .checked_mul(10)?
            .checked_add(u64::from(byte - b'0'))?;
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(data: &str) -> SseEvent {
        SseEvent::Message(SseMessage {
            event: None,
            data: data.to_string(),
            id: None,
        })
    }

    #[test]
    fn reset_for_reconnect_preserves_last_event_id() {
        let mut parser = SseParser::new();
        parser.push(b"id: 42\ndata: first\n\n").unwrap();
        parser.reset_for_reconnect();

        assert_eq!(parser.last_event_id(), Some("42"));
        let events = parser.push(b"data: second\n\n").unwrap();
        assert_eq!(
            events,
            vec![SseEvent::Message(SseMessage {
                event: None,
                data: "second".to_string(),
                id: Some("42".to_string()),
            })]
        );
    }

    #[test]
    fn parses_basic_event() {
        let mut parser = SseParser::new();
        let events = parser.push(b"data: hello\n\n").unwrap();
        assert_eq!(events, vec![message("hello")]);
    }

    #[test]
    fn push_into_reuses_event_buffer() {
        let mut parser = SseParser::new();
        let mut events = Vec::new();

        parser.push_into(b"data: one\n\n", &mut events).unwrap();
        assert_eq!(events, vec![message("one")]);

        events.clear();
        parser.push_into(b"data: two\n\n", &mut events).unwrap();
        assert_eq!(events, vec![message("two")]);
    }

    #[test]
    fn parses_multiline_data() {
        let mut parser = SseParser::new();
        let events = parser.push(b"data: hello\ndata: world\n\n").unwrap();
        assert_eq!(events, vec![message("hello\nworld")]);
    }

    #[test]
    fn handles_every_byte_boundary() {
        let input = b"event: update\nid: 42\ndata: hello\ndata: world\n\n";
        for split in 0..input.len() {
            let mut parser = SseParser::new();
            let mut events = parser.push(&input[..split]).unwrap();
            events.extend(parser.push(&input[split..]).unwrap());
            assert_eq!(events.len(), 1, "split={split}");
            assert_eq!(
                events[0],
                SseEvent::Message(SseMessage {
                    event: Some("update".to_string()),
                    data: "hello\nworld".to_string(),
                    id: Some("42".to_string()),
                })
            );
        }
    }

    #[test]
    fn handles_crlf_split_across_chunks() {
        let mut parser = SseParser::new();
        assert!(parser.push(b"data: hello\r").unwrap().is_empty());
        let events = parser.push(b"\n\r\n").unwrap();
        assert_eq!(events, vec![message("hello")]);
    }

    #[test]
    fn ignores_bom_split_across_chunks() {
        let mut parser = SseParser::new();
        assert!(parser.push(&UTF8_BOM[..1]).unwrap().is_empty());
        assert!(parser.push(&UTF8_BOM[1..]).unwrap().is_empty());
        let events = parser.push(b"data: hello\n\n").unwrap();
        assert_eq!(events, vec![message("hello")]);
    }

    #[test]
    fn emits_retry_and_keeps_last_id() {
        let mut parser = SseParser::new();
        let events = parser.push(b"retry: 1500\nid: 7\ndata: hi\n\n").unwrap();
        assert_eq!(
            events,
            vec![
                SseEvent::Retry(Duration::from_millis(1500)),
                SseEvent::Message(SseMessage {
                    event: None,
                    data: "hi".to_string(),
                    id: Some("7".to_string()),
                }),
            ]
        );
        assert_eq!(parser.last_event_id(), Some("7"));
        assert_eq!(parser.retry(), Some(Duration::from_millis(1500)));
    }

    #[test]
    fn ignores_comments_and_unknown_fields() {
        let mut parser = SseParser::new();
        let events = parser
            .push(b": keepalive\nunknown: value\ndata: hi\n\n")
            .unwrap();
        assert_eq!(events, vec![message("hi")]);
    }

    #[test]
    fn empty_data_field_dispatches_empty_message() {
        let mut parser = SseParser::new();
        let events = parser.push(b"data:\n\n").unwrap();
        assert_eq!(events, vec![message("")]);
    }

    #[test]
    fn no_data_field_does_not_dispatch() {
        let mut parser = SseParser::new();
        let events = parser.push(b"event: ping\n\n").unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn discards_incomplete_final_event() {
        let mut parser = SseParser::new();
        let events = parser.push(b"data: incomplete").unwrap();
        assert!(events.is_empty());
        assert!(parser.finish().unwrap().is_empty());
    }

    #[test]
    fn enforces_line_limit() {
        let mut parser = SseParser::with_limits(4, 1024);
        assert_eq!(
            parser.push(b"data: hello\n\n"),
            Err(SseError::LineTooLong { limit: 4 })
        );
    }

    #[test]
    fn enforces_event_limit() {
        let mut parser = SseParser::with_limits(1024, 4);
        assert_eq!(
            parser.push(b"data: hello\n\n"),
            Err(SseError::EventTooLarge { limit: 4 })
        );
    }

    #[test]
    fn replaces_invalid_utf8_by_default() {
        let mut parser = SseParser::new();
        let events = parser.push(b"data: \xFF\n\n").unwrap();
        assert_eq!(events, vec![message("\u{FFFD}")]);
    }

    #[test]
    fn strict_mode_rejects_invalid_utf8() {
        let mut parser = SseParser::strict();
        assert_eq!(parser.push(b"data: \xFF\n\n"), Err(SseError::Utf8));
    }

    #[test]
    fn last_event_id_persists_without_new_id() {
        let mut parser = SseParser::new();
        let events = parser.push(b"id: 7\ndata: one\n\ndata: two\n\n").unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(
            events[0],
            SseEvent::Message(SseMessage {
                event: None,
                data: "one".to_string(),
                id: Some("7".to_string()),
            })
        );
        assert_eq!(
            events[1],
            SseEvent::Message(SseMessage {
                event: None,
                data: "two".to_string(),
                id: Some("7".to_string()),
            })
        );
    }

    #[test]
    fn event_type_resets_after_dispatch() {
        let mut parser = SseParser::new();
        let events = parser
            .push(b"event: update\ndata: one\n\ndata: two\n\n")
            .unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(
            events[0],
            SseEvent::Message(SseMessage {
                event: Some("update".to_string()),
                data: "one".to_string(),
                id: None,
            })
        );
        assert_eq!(events[1], message("two"));
    }

    #[test]
    fn invalid_and_overflowing_retry_values_are_ignored() {
        let mut parser = SseParser::new();
        let events = parser
            .push(b"retry: nope\nretry: 18446744073709551616\ndata: hi\n\n")
            .unwrap();
        assert_eq!(events, vec![message("hi")]);
        assert_eq!(parser.retry(), None);
    }

    #[test]
    fn handles_cr_only_line_endings() {
        let mut parser = SseParser::new();
        let mut events = parser.push(b"data: hello\r\r").unwrap();
        events.extend(parser.finish().unwrap());
        assert_eq!(events, vec![message("hello")]);
    }

    #[test]
    fn removes_only_one_leading_space() {
        let mut parser = SseParser::new();
        let events = parser.push(b"data:  hello\n\n").unwrap();
        assert_eq!(events, vec![message(" hello")]);
    }

    #[test]
    fn ignores_id_containing_nul() {
        let mut parser = SseParser::new();
        let events = parser.push(b"id: a\x00b\ndata: hi\n\n").unwrap();
        assert_eq!(events, vec![message("hi")]);
        assert_eq!(parser.last_event_id(), None);
    }

    #[test]
    fn colonless_data_is_an_empty_data_field() {
        let mut parser = SseParser::new();
        let events = parser.push(b"data\n\n").unwrap();
        assert_eq!(events, vec![message("")]);
    }

    #[test]
    fn parses_cjk_data() {
        let mut parser = SseParser::new();
        let events = parser.push("data: 你好，世界\n\n".as_bytes()).unwrap();
        assert_eq!(events, vec![message("你好，世界")]);
    }

    #[test]
    fn parses_cjk_split_inside_codepoints() {
        let input = "data: 中文\n\n".as_bytes();
        for split in 0..input.len() {
            let mut parser = SseParser::new();
            let mut events = parser.push(&input[..split]).unwrap();
            events.extend(parser.push(&input[split..]).unwrap());
            assert_eq!(events, vec![message("中文")], "split={split}");
        }
    }

    #[test]
    fn parses_cjk_event_and_id() {
        let mut parser = SseParser::new();
        let events = parser
            .push("event: 更新\nid: 会话-7\ndata: 内容\n\n".as_bytes())
            .unwrap();
        assert_eq!(
            events,
            vec![SseEvent::Message(SseMessage {
                event: Some("更新".to_string()),
                data: "内容".to_string(),
                id: Some("会话-7".to_string()),
            })]
        );
    }

    #[test]
    fn parses_emoji_split_inside_codepoint() {
        let input = "data: 🦀🚀\n\n".as_bytes();
        for split in 0..input.len() {
            let mut parser = SseParser::new();
            let mut events = parser.push(&input[..split]).unwrap();
            events.extend(parser.push(&input[split..]).unwrap());
            assert_eq!(events, vec![message("🦀🚀")], "split={split}");
        }
    }

    #[test]
    fn line_limit_counts_utf8_bytes() {
        let mut parser = SseParser::with_limits(8, 1024);
        assert_eq!(
            parser.push("data: 中文\n\n".as_bytes()),
            Err(SseError::LineTooLong { limit: 8 })
        );
    }

    #[test]
    fn bom_before_cjk_is_ignored() {
        let mut parser = SseParser::new();
        let mut input = UTF8_BOM.to_vec();
        input.extend_from_slice("data: 中文\n\n".as_bytes());
        let events = parser.push(&input).unwrap();
        assert_eq!(events, vec![message("中文")]);
    }

    #[test]
    fn partial_bom_at_eof_is_discarded() {
        let mut parser = SseParser::new();
        assert!(parser.push(&UTF8_BOM[..2]).unwrap().is_empty());
        assert!(parser.finish().unwrap().is_empty());
    }
}
