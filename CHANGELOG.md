# Changelog

## 0.6.1 — SSE tail flush (recover the final event on gateway-cut streams)

### Added

- `SseParser::with_tail_policy(TailPolicy)` with `TailPolicy::Discard` (default,
  unchanged spec behavior) and `TailPolicy::Flush`. Under `Flush`,
  `finish`/`finish_into` dispatch the trailing event instead of discarding it:
  a final `data:` line missing only the blank-line terminator, an aggregated
  multi-`data:` tail, a trailing bare CR as line terminator, and a buffered
  partial line (interpreted as a field; genuinely truncated payloads then
  surface as mapper/JSON errors instead of silent byte loss).
- `SseMapper::sse_parser` hook so adapters choose their parser configuration.
  `OpenAiChatStreamMapper` opts into `Flush`: lenient gateways routinely cut
  the connection right after the final tool-call argument delta, and the
  discard default turned that into a truncated `arguments` JSON downstream.

## 0.6.0 — `ultra` reasoning tier

### Breaking/source changes

- `ReasoningEffort` adds `Ultra` (wire value `ultra`), above `Max`. The enum is
  exhaustive, so downstream matches must add an arm.
- `EffortMapping` adds the `ultra` field. Struct literals built without
  `..Self::default()` must add it.

### Behavior

- OpenAI Responses and Chat send `reasoning_effort: "ultra"` verbatim.
- The built-in mappings treat `ultra` as "as high as this provider goes":
  `identity` → `ultra`; `deepseek` → `max` (DeepSeek has nothing above `max`,
  and dropping the request entirely would silently fall back to the provider
  default); Gemini collapses it into `HIGH` next to `xhigh`/`max`. Providers
  with no entry map it to unsupported (`None`), like any other level.
- The tier is gated upstream — not every account is offered it — so requesting
  it is only meaningful where the provider accepts it.

## 0.5.1 — `tool_call_content` no longer clobbers non-empty content

### Behavior fix

- `ToolCallContentMode::Null` / `Omit` / `Empty` now canonicalize only content
  that is absent, `null`, or an empty string. Non-empty assistant text next to
  `tool_calls` is real transcript data and is preserved verbatim in every mode
  (the documented contract; 0.2.1–0.5.0 silently dropped it, blinding consumers'
  models to their own previous narration). Consumers that genuinely need to
  strip non-empty content must use `RequestOptions::request_transform`.
- A tripwire logs `[mutil-ai][WIRE-LOSS]` and asserts if a non-empty content is
  ever altered by these modes again.
- No API changes; source-compatible with 0.5.0.

## 0.5.0 — Tool-pairing identity fixes

### Breaking/source changes

- `NormalizeAction` adds `ReconciledToolResultId { name, from, to }`,
  `FlattenedToolCallIntoText { name }` and `FlattenedToolResultIntoText { name }`.
  Update exhaustive matches.
- `NormalizeStats` adds `reconciled_tool_result_id`, `flattened_tool_call` and
  `flattened_tool_result`. Add them with `..Default::default()` in struct literals.
- `NormalizeReport::is_clean()` can now be `false` for histories it previously
  reported as clean, because tool parts stored on a non-tool message are recorded.

### Behavior fixes

- Pair a tool result by its call id alone. The 0.4.0 requirement that the
  result name also equal the call name silently dropped real tool output, so the
  model re-answered the same user turn on every tool round.
- A result whose id matches no pending call can pair with a call whose id this
  run generated, and the result id is rewritten to that call's id. Explicit
  call ids still never fall back to names.
- Tool parts flattened into literal text on a `User`/`System` message are now
  reported instead of being rewritten invisibly.
- `MissingToolResultPolicy::DropToolCall` no longer leaves an assistant message
  with no parts, which strict providers reject.
- Unpaired-call repair now targets the assistant message the call was actually
  written to, after an orphan insertion shifted the message indices.

## 0.4.1 — Dual licensing

- Add Apache-2.0 as an alternative to the existing MIT license (`MIT OR Apache-2.0`).
- Include both license texts and clarify commercial use and the separate model-provider terms.
- No Rust API or runtime behavior changes; the published 0.4.0 package and tag remain unchanged.

## 0.4.0 — Security hardening

### Breaking/source and behavior changes

- Add `RequestOptions.response_limits`, `StreamReconnectPolicy.max_delay` /
  `deduplicate_replay`, and `Error::ResponseLimit`.
- Disable HTTP redirects and protect provider/custom authentication fields.
- Enforce strict tool-result identity, unique call IDs and collision-free synthetic IDs.
- Preserve repeated SSE events by default; restrict opt-in deduplication to reconnect replay.
- Bound decoded response bytes/events, cache size and OpenAI Chat tool/progress accumulation.
- Enforce a logical-request deadline across header injection, retries and stream reconnects.
- Redact transport/stream errors and header/query/context values in configuration Debug.
- Leave absent OpenAI Chat stream call IDs for normalization rather than inventing `call_0`.
- Refresh the lockfile from withdrawn `yoke-derive 0.8.3` to compatible `0.8.4`.

See [the 0.4 migration guide](docs/migration-0.4.md) for defaults and upgrade examples.
