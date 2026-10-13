# Changelog

## Unreleased

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
