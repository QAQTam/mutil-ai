English | [简体中文](README.zh-CN.md)

# mutil-ai (provider-neutral AI SDK)

A provider-neutral model-invocation SDK for Rust beginners.

[![CI](https://github.com/QAQTam/mutil-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/QAQTam/mutil-ai/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/mutil-ai.svg)](https://crates.io/crates/mutil-ai)
[![docs.rs](https://docs.rs/mutil-ai/badge.svg)](https://docs.rs/mutil-ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Installation

```bash
cargo add mutil-ai
```

- Minimum supported Rust version (MSRV): **1.97.1** (see `rust-version` in `Cargo.toml`)
- Optional feature: `blocking` (provides the blocking `BlockingAdapter` / `BlockingStream`)
- License: MIT


Its goal is not to support every advanced feature at once, but to answer one key question first:

> Internal agent history can be messy — but before it is sent to a provider, how does it become clean, strictly paired roles and tool calls?

This library does not provide an Agent runtime. It defines its own very small neutral message model, and then lets each provider adapter be responsible only for:

1. Invoking the normalization layer;
2. Serializing the clean messages into the provider's HTTP request;
3. Converting the provider response back into the neutral model.

Session history, tool execution, permission approval, cancellation policy, and loop step limits are decided by downstream projects. [`examples/minimal_agent.rs`](examples/minimal_agent.rs) in this repository shows how to build a minimal Agent runtime using only the public API; it is not a public capability of the SDK.

Adapters must still follow each provider's HTTP protocol — otherwise the calls simply wouldn't work — but the application layer never has to face those differences directly.

For a summary of protocol surfaces and field differences across major domestic (China) models, see:

- [`docs/provider-compatibility.md`](docs/provider-compatibility.md)

For the SDK's long-term contract, current implementation, and handoff checklist, see:

- [`docs/sdk-spec.md`](docs/sdk-spec.md)

For frozen candidate API semantics intended to help teams evaluate adoption, see:

- [`docs/api-freeze-0.2.md`](docs/api-freeze-0.2.md)
- [`docs/api-freeze-0.1.md`](docs/api-freeze-0.1.md) (historical version)

For the implementation handoff of the current version, see:

- [`docs/handoff-0.2.md`](docs/handoff-0.2.md)

For a registry of generic extension capabilities required by downstream gateways / agents, see:

- [`docs/consumer-capability-requirements.md`](docs/consumer-capability-requirements.md)

## 1. Core idea

### Internal roles

Applications may internally use these roles:

```text
System / Developer / User / Assistant / Tool / Custom(String)
```

This lets legacy agents, plugins, and workflows keep their own semantics without having to rewrite their history for some specific provider.

### Egress roles

Before anything reaches a provider, only safe roles remain:

```text
System / User / Assistant / Tool
```

Default downgrade rules:

| Internal role | Egress handling |
|---|---|
| `System` | Extracted as system instructions |
| `Developer` | Downgraded to `System` |
| `Custom("...")` | Downgraded to `User` |
| `User` | `User` |
| `Assistant` | `Assistant` / Gemini's `model` |
| `Tool` | The provider's corresponding tool result |

## 2. Tool-pairing cleanup

What strict providers fear most is "a tool call without a tool result" or "a tool result without a matching tool call".

This library's default rules:

1. An assistant `ToolCall` without an id: automatically assigned `call_1`, `call_2`, etc.
2. A tool result without an id: matched by tool name and given the corresponding id.
3. A tool call with no following result: an error result is synthesized automatically, guaranteeing complete pairing.
4. Orphan tool results: dropped by default, to avoid polluting upstream.
5. Anthropic / Gemini: adjacent user/assistant messages are merged, and tool results are placed into the user turn.

You can run the example directly to see the cleanup process:

```bash
cargo run --example normalize
```

The output will show:

```text
DowngradedDeveloper
DowngradedCustomRole { role: "reviewer" }
AssignedToolCallId { name: "get_weather", id: "call_1" }
SynthesizedMissingToolResult { call_id: "call_1", name: "get_weather" }
```

## 3. Minimal usage: calling a model directly

```rust
use mutil_ai::{ChatRequest, Message, ModelAdapter, OpenAI};

#[tokio::main]
async fn main() -> mutil_ai::Result<()> {
    let model = OpenAI::responses("gpt-5.2");

    let request = ChatRequest::new([
        Message::system("你是一个简洁、耐心的 Rust 老师。"),
        Message::user("用三句话解释什么是 agent。"),
    ]);

    let response = model.complete(&request).await?;
    println!("{}", response.text());

    Ok(())
}
```

`OPENAI_API_KEY` is read from the environment. `ModelAdapter` here is the stable boundary downstream projects should call directly: pass in a `ChatRequest`, get back a `ChatResponse`; use `complete_with` when you need request-level headers, cancellation, or transforms.

Switching providers requires no changes to business request code:

```rust
let model = OpenAI::chat("gpt-4.1");
// let model = OpenAI::responses("gpt-5.2");
// let model = mutil_ai::Anthropic::messages("claude-sonnet-4-5");
// let model = mutil_ai::Gemini::generate_content("gemini-3-flash");
```

## 4. Minimal Agent runtime: an example, not a library capability

The SDK does not export `Agent`, `AgentBuilder`, `ToolRegistry`, or `tool_fn`. Those types would bake history storage, tool registration, permission models, and termination policy into the library — not a good fit as a stable API for a general-purpose provider core.

The complete minimal runtime lives at:

[`examples/minimal_agent.rs`](examples/minimal_agent.rs)

It depends only on this crate's public types, and implements:

1. Keeping the `Vec<Message>` history locally;
2. Putting the system prompt and `ToolSpec`s into a `ChatRequest`;
3. Calling `ModelAdapter::complete_with`;
4. Executing local tools and appending `Message::tool_results`;
5. Looping at most `max_steps` times.

Run it:

```bash
cargo run --example minimal_agent
```

Other projects can copy this file as a starting point and then substitute their own permission approval, cancellation, auditing, and tool dispatch implementations — no need to modify or extend the SDK's public API.

## 5. Unified chain of thought: message vs. reasoning

Each provider returns thinking content with different field names and containers:

| Protocol | Raw fields |
|---|---|
| OpenAI Chat-compatible API | `message.reasoning_content`, `message.reasoning`, `message.reasoning_details` |
| OpenAI Responses | output item `type = "reasoning"`, which may contain `summary`, `content`, `encrypted_content` |
| Anthropic Messages | content blocks `thinking` / `redacted_thinking`, where `signature` / `data` must be passed back verbatim |
| Gemini generateContent | part `thought: true`, `thoughtSignature`; the signature may also be attached to a `functionCall` part |

The application layer never needs to inspect these fields. Responses are uniformly turned into:

```rust
Part::Text { text, provider_state }        // final message shown to the user; state may carry a Gemini signature
Part::Reasoning(Reasoning {                 // thinking context or summary
    kind: ReasoningKind,
    summary: Option<String>,
    text: Option<String>,
    state: Option<ProviderState>,
})
Part::ToolCall(call)                        // tool call
Part::ToolResult(result)                    // tool result
```

`ReasoningKind` has exactly four variants:

```text
Summary    // a summary, e.g. OpenAI Responses summary
Text       // human-readable thinking text, e.g. Anthropic thinking / Gemini thought
Encrypted  // encrypted reasoning state, unreadable to the user
Redacted   // blocks removed by safety policy
```

`state` is provider-specific opaque data; it may contain OpenAI reasoning items, Anthropic `signature`, Gemini `thoughtSignature`, etc. Applications should not parse or modify it.

Downstream code only needs to match like this:

```rust
for part in &response.message.parts {
    match part {
        Part::Text { text, .. } => println!("message: {text}"),
        Part::Reasoning(reasoning) if reasoning.kind == ReasoningKind::Summary => {
            println!("summary: {}", reasoning.summary.as_deref().unwrap_or(""));
        }
        Part::Reasoning(reasoning) => {
            println!("reasoning: {}", reasoning.text.as_deref().unwrap_or(""));
        }
        Part::ToolCall(call) => println!("tool: {}", call.name),
        Part::ToolResult(result) => println!("tool result: {}", result.content),
        Part::ImageUrl { .. } => {}
    }
}
```

`response.text()` and `message.text_content()` never include reasoning, preventing the thinking process from being mistaken for the final answer. To try it:

```bash
cargo run --example reasoning
```

### Request configuration is unified too

Per-provider toggles are likewise mapped by the adapter:

```rust
use mutil_ai::{
    ChatRequest, ReasoningConfig, ReasoningEffort, ReasoningSummary,
};

let request = ChatRequest::user("分析这个问题")
    .reasoning(
        ReasoningConfig::new()
            .effort(ReasoningEffort::High)
            .summary(ReasoningSummary::Detailed)
            .include_text(true)
            .include_encrypted(true),
    );
```

Mapping rules:

- OpenAI Chat: `effort` -> `reasoning_effort`
- OpenAI Responses: `effort/summary/include_encrypted` -> `reasoning` and `include`
- Anthropic: `mode/budget_tokens/include_text` -> `thinking`
- Gemini: `effort/budget_tokens/include_text` -> `generationConfig.thinkingConfig`

Unsupported target fields are ignored — the library never guesses an approximate field and stuffs the value in there.

Common request controls also have a neutral representation:

```rust
use mutil_ai::{ResponseFormat, ToolChoice};

let request = ChatRequest::user("返回结构化天气结果")
    .tools(tools)
    .tool_choice(ToolChoice::Required)
    .response_format(ResponseFormat::JsonSchema {
        name: "weather".to_string(),
        schema: serde_json::json!({
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "temperature": {"type": "number"}
            },
            "required": ["city", "temperature"]
        }),
        strict: Some(true),
    })
    .stop(["END"])
    .seed(42);
```

Each protocol maps only the fields it supports; for example, Anthropic does not accept the neutral `response_format` and will return `Unsupported` rather than silently ignoring it.

### Replay boundaries

- Reasoning `state` from the same protocol is replayed verbatim.
- Anthropic's `thinking` / `redacted_thinking` blocks are placed back into the assistant content in their original order.
- OpenAI Responses reasoning items are placed back into `input` as the original objects.
- Gemini's `thoughtSignature` is preserved on the corresponding reasoning part, tool call, or visible text part.
- Opaque state across protocols is dropped by default in the normalization layer — Claude/Gemini signatures are never sent to OpenAI.
- Reasoning inside non-assistant messages is dropped by default and recorded in `NormalizeReport`; it is never silently promoted into user text.

This way the egress layer still only knows one simple set of content types, and provider special cases are confined to adapters.

Reference links:

- OpenAI Reasoning: <https://developers.openai.com/api/docs/guides/reasoning>
- Anthropic Extended Thinking: <https://platform.claude.com/docs/en/build-with-claude/extended-thinking>
- Gemini Thinking: <https://ai.google.dev/gemini-api/docs/thinking>
- Gemini Thought Signatures: <https://ai.google.dev/gemini-api/docs/thought-signatures>
- OpenRouter Reasoning Tokens: <https://openrouter.ai/docs/use-cases/reasoning-tokens>

## 6. Unified streaming

All four built-in adapters implement:

```rust
async fn stream_with(
    &self,
    request: &ChatRequest,
    options: &RequestOptions,
) -> Result<ModelStream>;
```

Downstream code only sees the unified `StreamEvent`:

```rust
pub enum StreamEvent {
    Start {
        provider: &'static str,
        model: String,
        metadata: ResponseMetadata,
    },
    TextDelta {
        text: String,
    },
    ReasoningDelta {
        kind: ReasoningKind,
        text: String,
    },
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments_delta: String,
    },
    Usage {
        usage: Usage,
    },
    Retry {
        delay: Duration,
    },
    Done {
        response: Box<ChatResponse>,
        finish_reason: Option<String>,
    },
}
```

Usage example:

```rust
use mutil_ai::{ChatRequest, ModelAdapter, OpenAI, StreamEvent, next_event};

let model = OpenAI::chat("gpt-5.2");
let request = ChatRequest::user("你好");

let mut stream = model.stream(&request).await?;
while let Some(event) = next_event(&mut stream).await {
    match event? {
        StreamEvent::TextDelta { text } => print!("{text}"),
        StreamEvent::ReasoningDelta { text, .. } => print!("[thinking] {text}"),
        StreamEvent::ToolCallDelta { arguments_delta, .. } => {
            print!("[tool args] {arguments_delta}");
        }
        StreamEvent::Done { response, .. } => {
            // 持久化时使用组装好的完整 message，不要自己拼 delta。
            println!("\n{}", response.text());
        }
        _ => {}
    }
}
```

Key problems already handled:

- OpenAI Chat: `delta.content`, `reasoning_content`, `reasoning_details`
- OpenAI Responses: text / reasoning summary / reasoning text / function arguments events
- Anthropic: `content_block_delta`, `thinking_delta`, `signature_delta`, `input_json_delta`
- Gemini: `:streamGenerateContent?alt=sse`, thought, text, functionCall, thoughtSignature
- Tool arguments are accumulated by index and parsed into JSON in `Done.response`
- Reasoning text and the final signature/encrypted state remain intact in `Done.response`
- Usage is sent as its own event and also kept in the final response
- `Start.metadata` and `Done.response.metadata` include status, request id, and response headers
- The SSE `retry:` field is mapped to `StreamEvent::Retry`

Current boundaries:

- HTTP errors before the stream starts receiving are retried according to `RetryPolicy`.
- Transport disconnections after the stream starts can be handled with explicitly enabled reconnect.
- Reconnect requires seeing an SSE `id`, sends `Last-Event-ID`, and reuses the same mapper.
- Duplicate `(id, data)` pairs already received are deduplicated, avoiding repeated text or tool arguments after a reconnect.
- Only Chat/Gemini treat EOF as a normal end; Responses/Anthropic treat an EOF without a terminal event as a recoverable disconnect.
- `collect_stream` lets you skip incremental rendering and simply collect the final `ChatResponse`.

Enabling disconnect reconnect:

```rust
use std::time::Duration;
use mutil_ai::{RequestOptions, StreamReconnectPolicy};

let options = RequestOptions::new().stream_reconnect(
    StreamReconnectPolicy::new()
        .max_attempts(2)
        .delay(Duration::from_millis(250))
        .require_event_id(true),
);

let mut stream = model.stream_with(&request, &options).await?;
```

On reconnect the following is sent:

```http
Last-Event-ID: <last-seen-id>
```

Session, idempotency key, and the original request body remain unchanged.

Runnable example:

```bash
cargo run --example streaming
```

## 7. How retry headers are recognized

`Retry-After` is an HTTP response header, not a body field. It commonly appears with:

- `429 Too Many Requests`
- `503 Service Unavailable`

It has two formats:

```http
Retry-After: 120
```

meaning wait 120 seconds.

```http
Retry-After: Wed, 21 Oct 2026 07:28:00 GMT
```

meaning wait until this HTTP date. If the date is already in the past, the wait time is treated as 0.

This library will:

1. Check the response's `retry-after` header;
2. First try to parse it as a number of seconds;
3. Fall back to parsing it as an HTTP date;
4. Store it in `Error::Api.retry_after`;
5. Expose it to retry logic via `error.retry_after()` and `error.is_retryable()`.

```rust
use std::time::Duration;

match model.complete(&request).await {
    Ok(response) => println!("{}", response.text()),
    Err(error) if error.is_retryable() => {
        let delay = error
            .retry_after()
            .unwrap_or(Duration::from_secs(1));
        eprintln!("retry after {delay:?}: {error}");
    }
    Err(error) => return Err(error),
}
```

Errors carry a stable classification, so there is no need to match on error strings:

```rust
use mutil_ai::ErrorKind;

match model.complete(&request).await {
    Ok(response) => println!("{}", response.text()),
    Err(error) if error.kind() == ErrorKind::RateLimited => {
        eprintln!("rate limited: {:?}", error.request_id());
    }
    Err(error) if error.kind() == ErrorKind::Authentication => {
        eprintln!("check credentials");
    }
    Err(error) => return Err(error),
}
```

`Error` also provides:

- `kind()`: the stable `ErrorKind`;
- `status()`: the HTTP status;
- `provider()`: the provider name;
- `request_id()`: the request id from the response, if available;
- `retry_after()` / `retry_source()`;
- `is_retryable()`;
- `body_bytes()`: the byte length of the provider error body;
- `provider_error()`: a safe structured error summary;
- `raw_body()`: explicitly read the provider's raw error body.

`provider_error()` understands common OpenAI/Anthropic/Gemini structures as well as compatible gateways, and returns a length- and character-whitelist-sanitized `code`, `error_type`, `status`, message byte count, and details count; it does not return the provider message verbatim.

By default, `Error::Api`'s `Display` and `Debug` do not output the provider body, preventing `eprintln!` or logging frameworks from accidentally recording echoed prompts, account information, or tool content. Only an explicit call to `raw_body()` or direct field access retrieves the raw text.

### RetryPolicy design

Retry logic is layered in three tiers:

```text
Retry-After / status code
        ↓
RetryPolicy::delay_for()
        ↓
retry_async / send_json_retry
        ↓
adapter
```

The adapter never decides how long to sleep itself; it is only responsible for rebuilding the HTTP request.

```rust
use mutil_ai::{OpenAI, RetryPolicy};
use std::time::Duration;

let model = OpenAI::responses("gpt-5.2").retry_policy(
    RetryPolicy::default()
        .max_attempts(5)
        .base_delay(Duration::from_millis(300))
        .max_delay(Duration::from_secs(10))
        .jitter_ratio(0.2)
        .max_retry_after(Duration::from_secs(60)),
);
```

Rules:

1. Retry only when `error.is_retryable()` is true.
2. If `Retry-After` is present: respect the server-provided time first.
3. If `Retry-After` exceeds `max_retry_after`: stop, to avoid waiting forever.
4. If there is no `Retry-After`: exponential backoff with jitter.
5. `max_attempts` includes the first request.
6. By default `require_idempotency_key = true`: without an `Idempotency-Key`, the request is sent once and POSTs are not replayed automatically.

Explicitly accepting retries without an idempotency key:

```rust
let policy = RetryPolicy::default()
    .max_attempts(3)
    .require_idempotency_key(false);
```

Only turn off this protection when you know for certain the request can be safely replayed.

### Header priority

The parsing order is fixed:

```text
1. retry-after-ms
2. retry-after
3. provider-specific reset headers
```

`retry-after-ms` is not a standard header, but the official OpenAI, Anthropic, and Google SDKs all support it, and it has better precision than whole seconds.

### OpenAI

In the official OpenAPI spec, `429` and `503` responses return:

```http
Retry-After: 3
```

The official Python SDK also recognizes, with priority:

```http
retry-after-ms: 1500
```

As a fallback, this library also supports:

```http
x-ratelimit-reset-requests: 6m0s
x-ratelimit-reset-tokens: 1h
```

Supporting `ns / us / µs / ms / s / m / h` as well as combined forms, e.g. `1h30m`.

### Anthropic

The official SDK likewise prefers `retry-after-ms`, then `retry-after`.

Anthropic's reset headers are RFC3339 UTC timestamps:

```http
anthropic-ratelimit-requests-reset: 2026-02-25T20:02:32Z
anthropic-ratelimit-tokens-reset: 2026-02-25T20:02:36Z
anthropic-ratelimit-input-tokens-reset: 2026-02-25T20:02:37Z
anthropic-ratelimit-output-tokens-reset: 2026-02-25T20:02:36Z
```

### Google Gemini

The official `python-genai` SDK supports:

```http
retry-after-ms: 1
```

Google API error bodies may also contain `RetryInfo.retryDelay`, but that is not a header, so the current version does not mix it into the header-parsing layer.

Reference links:

- OpenAI Rate Limits: <https://developers.openai.com/api/docs/guides/rate-limits>
- Anthropic Rate Limits: <https://platform.claude.com/docs/en/api/rate-limits>
- Gemini Rate Limits: <https://ai.google.dev/gemini-api/docs/rate-limits>

## 8. Hand-written SSE parser

`src/sse.rs` is an incremental, zero-I/O SSE parser:

```rust
let mut parser = SseParser::new();
for chunk in response.bytes_stream() {
    for event in parser.push(&chunk?)? {
        match event {
            SseEvent::Message(message) => println!("{}", message.data),
            SseEvent::Retry(delay) => println!("retry={delay:?}"),
        }
    }
}
```

It does not depend on `sse-stream`, `tokio-util`, or `http-body`; it only handles byte state.

Supported:

- Arbitrary chunk splits
- `\n`, `\r\n`, and lone `\r`
- UTF-8 BOM
- Comments and unknown fields
- Multi-line `data`
- `event`, `id`, `retry`
- `id` inheritance across events
- `event` reset after dispatch
- Line/event size limits
- Incomplete events discarded at EOF

UTF-8 handling follows WHATWG EventSource by default: invalid bytes are replaced with U+FFFD. For `sse-stream`-style strict errors, use:

```rust
let parser = SseParser::strict();
```

Correctness tests cover:

- Splitting at every byte boundary
- CRLF across chunks
- BOM across chunks
- Multi-line data
- ID inheritance
- Event reset
- Invalid retry values and overflow
- NUL IDs
- Colonless data
- UTF-8 replace / strict
- CJK split in the middle of a codepoint
- Emoji split in the middle of a codepoint
- CJK event/id/data
- Line limits counted in UTF-8 bytes
- BOM followed by CJK
- Incomplete events at EOF

Comparison tests against `sse-stream 0.3.0`:

```bash
cargo test --test sse_compare -- --ignored --nocapture
cargo test --test sse_concurrency -- --ignored --nocapture
```

The first test simulates 200,000 tokens — a 10,000 tok/s stream sustained for 20 seconds — measured at 1 / 10 / 100 events per chunk.

The second test dynamically selects 1 / 4 / 8 / 12 concurrent clients based on the machine's logical CPU count; each client parses 300,000 tokens, measuring:

- Worker CPU time
- CPU time per token
- CPU coefficient of variation across rounds
- CPU coefficient of variation across clients
- CPU headroom relative to 10,000 tok/s

Results on the current machine:

```text
1-8 clients:
  手写 parser 的每 token CPU 更低，通常低 15%-35%
  两种实现的轮次波动都很小

12/12 logical CPUs, 1 event/chunk:
  手写 parser 仍然更低

12/12 logical CPUs, 10 events/chunk:
  sse-stream 的每 token CPU 略低，客户端间波动也更小

所有场景:
  两种 parser 相对 10,000 tok/s 都有约 85x-230x CPU 余量
```

This shows that 10,000 tok/s is nowhere near the SSE parser's saturation point. What actually drives CPU fluctuation is mainly JSON, networking, the model service, logging, and upper-level agent scheduling — not SSE byte parsing. Under concurrent saturation, the memory allocator, SMT, and the scheduler begin to matter more than the parser itself.

### Isolated comparison against qaqh-gate's `SseDecoder`

In a separate worktree, qaqh's data-only decoder was compared directly against this library's general-purpose parser:

```text
release, 500,000 events, 5 samples
```

After optimization:

```text
ASCII:
  qaqh 约 83-95 ns/event
  mutil-ai 约 121-125 ns/event
  差距约 1.3x-1.47x

CJK + emoji:
  qaqh 约 107-109 ns/event
  mutil-ai 约 144-154 ns/event
  差距约 1.34x-1.42x
```

Multiple clients:

```text
1-8 clients:
  每事件 CPU 差距约 1.14x-1.47x

12 logical CPUs:
  差距扩大，约 2.0x-2.4x
  主要来自分配器、SMT 和调度竞争
```

qaqh's decoder only handles `data:`, does not preserve event/id/retry, and does not enforce a strict UTF-8 policy, so it being faster is the expected outcome; this library keeps full SSE semantics and the state needed for reconnect.

### 10,000-session stress test

Added:

```bash
cargo test --test sse_sessions
cargo test --test sse_sessions -- --ignored --nocapture
```

Test model:

- 10,000 parser states alive simultaneously;
- 12 workers rotating through sessions;
- Each session parsed independently;
- Total event counts verified — any loss fails the test;
- The regular smoke test runs on every commit;
- The heavy version with 100 events/session is marked ignored.

Current release results on this machine:

```text
qaqh:    约 46M-60M events/s
mutil-ai: 约 17M-18M events/s
```

Based on a typical website workload of 10,000 concurrent sessions with an average of 10-100 events/s per session:

```text
总事件率约 100k-1M events/s
mutil-ai 仍有约 17x-170x 的解析余量
```

So the ten-thousand-session target is not bottlenecked by the SSE parser; the real constraints are:

- Per-session memory footprint;
- Async task count and scheduling;
- Network connections/TLS;
- JSON and business state machines;
- Logging and audit pipelines.


## 9. Headers and User-Agent

The SDK provides a default UA, but clients may override it:

```rust
let transport = TransportConfig::new()
    .user_agent("my-app/1.2.3")?
    .header("x-route", "prod")?
    .client_info(
        ClientInfo::new("my-app", "1.2.3")
            .instance_id("node-7"),
    );

let model = OpenAI::responses("gpt-5.2").transport(transport);
```

Request-level headers:

```rust
let options = RequestOptions::new()
    .header("x-sessionid", session_id)?
    .user_agent("my-app/1.2.3 (gateway)")?
    .context(RequestContext::new().session_id(session_id));
```

Business fields like `x-sessionid` and `x-tenant-id` are not hard-coded into adapters; clients pass them through.

Dynamic auth or session headers go through `HeaderInjector`:

```rust
use std::sync::Arc;
use async_trait::async_trait;
use mutil_ai::{HeaderInjector, RequestContext, Result, TransportConfig};

struct GatewayInjector;

#[async_trait]
impl HeaderInjector for GatewayInjector {
    async fn inject(&self, context: &RequestContext) -> Result<reqwest::header::HeaderMap> {
        let mut headers = reqwest::header::HeaderMap::new();
        // 这里可以读取缓存 token，或从安全 token provider 获取。
        headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_static("Bearer refreshed-token"),
        );
        if let Some(session_id) = &context.session_id {
            headers.insert(
                reqwest::header::HeaderName::from_static("x-sessionid"),
                reqwest::header::HeaderValue::from_str(session_id)?,
            );
        }
        Ok(headers)
    }
}

let transport = TransportConfig::new()
    .header_injector(Arc::new(GatewayInjector));
```

Header priority:

```text
SDK 默认 UA
    ↓
ClientConfig / TransportConfig
    ↓
RequestOptions
    ↓
HeaderInjector
```

`Authorization`, `Content-Type`, `Content-Length`, `Host`, and `Accept` are protected headers by default and cannot be silently overridden by ordinary configuration layers. `HeaderInjector` is part of the trusted transport configuration and may set `Authorization`, making it suitable for OAuth/token refresh scenarios.

For idempotent requests, pass:

```rust
let options = RequestOptions::new().idempotency_key("request-123");
```

The SDK sends `Idempotency-Key: request-123`, and retries of the same logical request reuse the same key.

Custom headers must be ASCII by default. Non-ASCII values such as CJK must be encoded by the caller — e.g. base64 or percent encoding; otherwise `NonAsciiHeaderValue` is returned.

`HeaderInjector` is re-executed before every HTTP attempt. Automatic retries can refresh tokens or transient headers, but session, idempotency key, static body, and static headers remain stable for the same logical request. Automatic replay is still not performed once stream deltas have been produced.

### Auditing and data points (no context retention)

Auditing is off by default; when enabled, it records only structured transport events — never prompts, messages, reasoning, tool arguments/results, request bodies, or response bodies:

```rust
use std::sync::Arc;
use mutil_ai::{AuditConfig, AuditEvent, AuditSink, TransportConfig};

struct Metrics;

impl AuditSink for Metrics {
    fn record(&self, event: AuditEvent) {
        // 发送到 metrics / audit pipeline。
        eprintln!("{event:?}");
    }
}

let transport = TransportConfig::new()
    .audit_sink(Arc::new(Metrics))
    .audit_config(AuditConfig::enabled());
```

Auditable data points include:

- `RequestStarted`
- `AttemptStarted`
- `ResponseHeaders`
- `AttemptFinished`
- `FirstToken`
- `Normalization`
- `RetryScheduled`
- `StreamReconnectScheduled`
- `RequestFinished` (`Success` / `Failure` / `Cancelled`)

Each event may include:

- provider, protocol, model;
- attempt number;
- status, duration, outcome;
- time to response headers and streaming TTFT;
- request body byte count and cumulative decoded response body byte count;
- `ErrorKind` and whether it is retryable;
- usage;
- provider request id;
- aggregate counts of normalization fix types;
- profile id / model profile hit status / capability snapshot, only when explicitly enabled.

`ResponseHeaders.elapsed` and `RequestFinished.timing.time_to_headers` are measured from the start of the logical request. They include the total time for DNS, connection, TLS, request sending, and server processing until the response headers arrive; reqwest does not expose the individual timestamps of these stages to this SDK, so finer-grained connect/TLS splits are not fabricated. `time_to_first_token` is produced only when a streaming response receives its first text, reasoning, or tool-call delta.

`Normalization` contains only counts per fix type, e.g. developer downgrades and tool call/result pairing repairs. It never records custom role names, tool names, call ids, or message content.

`RequestFinished.bytes.request_body` is the size of the serialized body of a single attempt; `bytes.response_body` is the cumulative decoded byte count received across retries/reconnects for this logical request, including provider error bodies but excluding body content.

When a stream is dropped by the consumer before a terminal event arrives, `RequestFinished { outcome: Cancelled }` is emitted. This can be disabled with `record_cancellation(false)`, or client-initiated cancellations can be distinguished from provider failures in your metrics.

Not included by default:

- session id;
- SDK request id;
- profile id and capability snapshot;
- prompt / message / reasoning / tool content;
- request/response bodies.

Correlation IDs must be explicitly opted into:

```rust
let config = AuditConfig::enabled()
    .include_provider_request_id(true)
    .include_sdk_request_id(false)
    .include_session_id(false)
    .include_usage(true)
    .include_profile(false)
    .record_timing(true)
    .record_normalization(true);
```

The session id is off by default because it may contain tenant or user context. Custom profile ids may also encode tenant information, so they are excluded from audits by default.

For production, a non-blocking bounded queue is recommended:

```rust
use mutil_ai::{AuditConfig, TransportConfig, bounded_audit_channel};

let (sink, receiver, stats) = bounded_audit_channel(4096);

let transport = TransportConfig::new()
    .audit_sink(sink)
    .audit_config(AuditConfig::enabled().sample_every(10));

// receiver 应交给后台线程/任务处理。
let snapshot = stats.snapshot();
println!(
    "accepted={}, dropped_full={}, disconnected={}",
    snapshot.accepted, snapshot.dropped_full, snapshot.dropped_disconnected
);
```

When the queue is full, events are dropped and counted immediately — model requests are never blocked. `sample_every(N)` samples per whole logical request, avoiding samples that capture attempts but miss the final result.

## 10. Explicit endpoints and extension fields

A single vendor may simultaneously offer Chat Completions, Responses, and Anthropic protocols. The base URL alone cannot determine behavior, so generic adapters require an explicit `EndpointSpec`:

```rust
use mutil_ai::{
    AuthStyle, EndpointAdapter, EndpointSpec, ProtocolSurface, ProviderProfile,
};

let profile = ProviderProfile::qwen_compatible();

let endpoint = EndpointSpec::new(
    ProtocolSurface::OpenAiChat,
    "https://gateway.example/v1",
    "/chat/completions",
    AuthStyle::Bearer,
)
.profile(profile);

let model = EndpointAdapter::new("qwen3", endpoint)
    .api_key_from_env("GATEWAY_API_KEY");
```

`OpenAICompatible` remains as a compatibility alias for `EndpointAdapter`; new code should prefer `EndpointAdapter`, since it dispatches to all of:

```text
OpenAiChat
OpenAiResponses
AnthropicMessages
GeminiGenerateContent
```

The protocol must still be chosen explicitly by the caller — never guessed from the URL. Gemini's normal and streaming paths are also described separately by `EndpointSpec`.

Built-in presets describe behavior only and are not bound to any URL:

```rust
ProviderProfile::deepseek_compatible();
ProviderProfile::qwen_compatible();
ProviderProfile::kimi_compatible();
ProviderProfile::glm_compatible();
ProviderProfile::doubao_compatible();
```

You can also look up presets by an explicit `ProfileId` — still no URL guessing:

```rust
use mutil_ai::{EndpointSpec, ProfileId, ProfileSelector};

let endpoint = EndpointSpec::openai_chat("https://gateway.example/v1")
    .profile_selector(ProfileSelector::Builtin(ProfileId::from("qwen")));
```

An unknown id returns `Unsupported`; it never falls back to URL guessing.

For example, the Qwen preset converts the neutral configuration into:

```text
ReasoningMode::Disabled       -> enable_thinking = false
ReasoningConfig::budget_tokens -> thinking_budget
```

The DeepSeek preset converts `thinking.type` and uses the effort mapping its documentation requires:

```text
minimal -> low
medium  -> high
xhigh   -> high
```

Model-level differences are overridden with `ModelProfile`. The SDK already provides two common Kimi presets:

```rust
use mutil_ai::{EndpointAdapter, ModelProfile};

let model = EndpointAdapter::new("kimi-k3", endpoint)
    .api_key_from_env("MOONSHOT_API_KEY")
    .model_profile(ModelProfile::kimi_k3());

let model = EndpointAdapter::new("kimi-k2.6", endpoint)
    .api_key_from_env("MOONSHOT_API_KEY")
    .model_profile(ModelProfile::kimi_k2_6());
```

`kimi_k2_6()` generates:

```json
{
  "thinking": {
    "type": "enabled",
    "keep": "all"
  }
}
```

Capabilities are checked before the request is sent:

```rust
let mut profile = ProviderProfile::qwen_compatible();
profile.capabilities.multimodal = false;
```

At that point, a request containing images returns `Unsupported` instead of first sending a request that is guaranteed to fail.


Provider extension fields for a single request:

```rust
let request = ChatRequest::user("你好")
    .extra_body("enable_thinking", true)
    .extra_body("reasoning_format", "parsed");

let options = RequestOptions::new()
    .header("x-sessionid", "session-1")?
    .extra_query("tenant", "team-a");
```

The merge rule is: profile defaults are written first, then request-level `extra_body`, so same-named extension fields are overridden by the single request. Canonical fields such as `model`, `messages`, `input`, and `stream` cannot be overridden; conflicts return `ReservedExtraBodyField` before the request is sent, rather than silently producing a broken request.

`EndpointAdapter` dispatches to Chat, Responses, Anthropic, or Gemini based on `EndpointSpec.protocol`; the normal path and Gemini's streaming path are both configured explicitly — never inferred from the URL.

## 11. Code structure

```text
src/
  types.rs             中立 message / reasoning / provider state
  normalize.rs         内部 role -> 出口 role，工具配对和跨协议清洗
  profile.rs           EndpointSpec / ProviderProfile / ModelProfile
  adapter/
    openai_chat.rs     OpenAI Chat Completions
    endpoint.rs        显式 EndpointSpec 的四协议 dispatch
    openai_responses.rs OpenAI Responses
    anthropic.rs       Anthropic Messages
    gemini.rs          Gemini generateContent
  retry.rs             Retry-After / reset header / backoff
  sse.rs               手写增量 SSE parser
  stream.rs            统一 StreamEvent 和流式事件组装
  headers.rs           UA / 请求头 / query / 注入接口

examples/
  hello.rs             直接调用 ModelAdapter 的最小示例
  minimal_agent.rs     只使用公开 API 的 Agent runtime 示例
```

Key point: **role conversion, tool repairs, and cross-provider reasoning cleanup happen exactly once, in `normalize.rs`** — adapters do not each maintain their own patch set.

## 12. What the current version deliberately does not do

This is v0.3.0; the current priority is getting roles, tool pairing, retry, SSE, reasoning, unified streaming, cancellation, and the generic provider extension boundary clearly defined. For now, the following are out of scope:

- WebSocket / Realtime
- OpenAI Responses' `previous_response_id` / `store`
- Anthropic `cache_control`
- Gemini Interactions API
- embedding / rerank / speech
- Automatically translating reasoning between different providers; the current choice is to safely drop it rather than guess

All of these can be added later — but they should be added in clearly defined places, not by stuffing provider special cases into the downstream runtime or the example Agent loop.
