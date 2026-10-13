# Changelog

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
