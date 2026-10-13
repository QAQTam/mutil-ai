# Upgrading from 0.3.x to 0.4.0

0.4.0 is a security-hardening release with intentional source/behavior changes.
Use `mutil-ai = "0.4"` (or `cargo add mutil-ai@0.4`) when upgrading after publication.
The pre-1.0 incompatible release boundary is 0.3 -> 0.4, not 0.3.2 -> 0.3.3.

## Source compatibility

- `RequestOptions` adds `response_limits`; prefer `RequestOptions::new()` plus builders,
  or add `..Default::default()` to existing struct literals.
- `StreamReconnectPolicy` adds `max_delay` and `deduplicate_replay`; prefer its builders,
  or fill the new fields with `..Default::default()`.
- `Error` adds `ResponseLimit { resource, limit }`. Update exhaustive matches. Its
  stable `ErrorKind` is `Decode`, and it is not automatically retryable.
- Existing adapter/ModelAdapter entry points and message types keep their signatures.
- OpenAI Chat streams no longer invent repeated `call_0` IDs when the server provides
  no ID. Native IDs remain unchanged; absent IDs remain `None` for the normalizer to
  assign collision-free IDs. Tool results without an ID continue to match by name.

## Network and authentication

- Built-in clients do **not** follow redirects, including same-origin redirects.
  Configure the final canonical provider URL; 3xx responses are returned as API errors.
  There is no unsafe cross-origin redirect opt-in in this release.
- Static header layers protect `x-api-key`, `x-goog-api-key`, and the actual custom
  authentication header selected by EndpointAdapter. Explicit trusted overrides can
  still use `HeaderPolicy::allow_override`; trusted HeaderInjector refresh is unchanged.
- EndpointAdapter reserves its actual `AuthStyle::QueryKey` parameter, in addition to
  protocol-specific query names. Do not duplicate authentication via `extra_query`.
- An empty/whitespace-only idempotency key is rejected before sending. An idempotency
  key still requires server-side support; the SDK cannot guarantee a provider honors it.

## Errors and logging

- Transport errors no longer print URL/source internals. SDK-created reqwest errors
  have the attached URL removed; use `Error::kind()` to classify them.
- `ProviderStream` Display/Debug redact raw provider text. `raw_body()`, `body_bytes()`
  and `provider_error()` now also work for this variant; raw access remains explicit.
- RequestOptions Debug prints header/query names, not their values, and redacts context,
  idempotency keys and provider request options. TransportConfig Debug hides header values.
- Provider payloads must not be logged through raw accessors without caller-side policy.

## Tools and streaming

- Explicit tool-result IDs never fall back to names. Unknown, completed, or mismatched
  IDs use the existing orphan policy. Names are used only when the result has no ID.
- Empty/duplicate tool-call IDs within the request are rejected; generated IDs avoid
  all explicit IDs in the history. Fix imported histories instead of relying on repair
  to accept ambiguous identities.
- Normal SSE connections never deduplicate repeated events. Reconnect replay suppression
  is off by default. Enable it only for a server with documented replay/resume semantics:

```rust
use mutil_ai::{RequestOptions, StreamReconnectPolicy};
use std::time::Duration;

let options = RequestOptions::new().stream_reconnect(
    StreamReconnectPolicy::new()
        .max_delay(Duration::from_secs(10))
        .deduplicate_replay(true),
);
```

Replay suppression matches a sequential suffix, including event type and occurrence
order, only at reconnect. It stores at most 1 MiB / 4096 recent events using shared
entries. SSE IDs alone do not guarantee exactly-once delivery; the server's contract
is still required. Unmatched replay ends suppression rather than deleting normal events.

- Reconnect delays above `max_delay` (default 60 seconds), including server SSE retry
  hints, are refused rather than slept. The original stream failure becomes terminal.
- ToolCallDelta remains lossless. OpenAI Chat ToolCallProgress is now emitted initially,
  after at least 4 KiB of argument growth, and once more before Done if needed. Do not
  expect one progress snapshot per delta. The final ToolCall/Done remains authoritative.

## Response budgets and timeouts

Default limits per logical request, shared across failed attempts and reconnects:

| Setting | Default |
| --- | --- |
| Decoded body bytes (after gzip decompression) | 16 MiB |
| SSE messages and retry fields, including replays | 16,384 |
| Accumulated arguments per OpenAI Chat tool call | 1 MiB |
| OpenAI Chat progress snapshot growth interval | 4 KiB |
| Retain `ChatResponse.raw` | Yes |

These are input-data budgets, **not** a promise that total heap usage equals 16 MiB:
JSON, text, opaque state, mapper output and event queues can add overhead. Other
protocols' tool arguments are bounded by the overall decoded body budget.
Raw collection can be disabled; opaque reasoning/signatures and usage are not stripped.

```rust
use mutil_ai::{RequestOptions, ResponseLimits};
use std::time::Duration;

let mut limits = ResponseLimits::default();
limits.max_body_bytes = 32 * 1024 * 1024;
limits.max_stream_events = 32_768;
limits.retain_raw = false; // ChatResponse.raw becomes JSON null
let options = RequestOptions::new()
    .response_limits(limits)
    .timeout(Duration::from_secs(120));
```

All numeric response limits must be non-zero. Choose explicit larger limits for
large workloads rather than depending on unbounded response collection.

`RequestOptions.timeout` now means one monotonic absolute deadline covering async
header injection, HTTP attempts, response reads, retry delays, and subsequent stream
reads/reconnects. It does not reset per attempt. No total timeout is imposed when
`timeout` is `None`; byte/event budgets and reconnect-delay caps still apply.
Synchronous user transforms cannot be preempted; once they return, an expired
deadline is rejected before sending. CancellationToken and idle_timeout remain separate.

## Release validation

The baseline audit and historical unsafe-behavior probes are kept under
`docs/security-audit/` for reference and excluded from the published crate. Do not run
those baseline assertions as fixed-code tests. Current regression coverage is in
`tests/security_hardening.rs`.

Local validation: one all-features test run passed 197 tests (20 security regressions),
with 0 failures and 1 intentionally ignored stress test. Strict Clippy and rustdoc
checks passed. The withdrawn transitive `yoke-derive 0.8.3` lock entry was then updated
to `0.8.4` and revalidated with strict Clippy and Cargo package verification; the full
suite was not repeated for this lockfile-only refresh. Rust 1.97.1 was not installed
locally; the declared MSRV is unchanged and remains covered by the existing CI job.
