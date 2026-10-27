# Native client tools: implementation status (0.7.0)

This is the 0.7.0 native-tool surface, not a provider capability claim. Real-provider/OAuth probes and full truncation/reconnect policy remain unverified.

`ToolDeclaration` models scoped function/custom declarations and text/grammar formats. `ChatRequest::tool_declarations` is accepted only on Responses and encoded into `to_responses_body`; other protocols still fail closed. The full Codex grammar is not bundled in this SDK repository; the caller supplies the grammar text. No real-provider capability claim exists yet.

`Part::NativeToolCall` preserves a Responses custom call and a namespaced function call without converting raw input to JSON. Completed native calls register pending by real `call_id` and require explicit matching results. Responses replay chooses output type by original kind. Native history is rejected on other wires. Legacy function constructors remain available. Persist the complete native call; a provider item is not a substitute.

The Responses stream mapper accumulates custom input and checks completion snapshots and identity. `NativeToolProgress` is not executable; `NativeToolCompleted` records item completion, not permission to dispatch. Cancellation and response failure policies remain the application scheduler's responsibility. Stream replay deduplication of fragments depends on transport event semantics; identical deltas cannot safely be presumed duplicates. This work does not yet meet the full T03–T04 acceptance or authorize production use.

An offline loopback HTTP test exercises decoding a synthetic custom response and sending its error result in the next request. Its first request does not carry a declaration, so it verifies only the replay half of the protocol, not a full declared-tool interaction or any real provider.

## Upgrade considerations

Adding `Part::NativeToolCall` and `StreamEvent::NativeToolProgress`/`NativeToolCompleted` requires downstream exhaustive Rust matches to add arms. Adding `ChatRequest::tool_declarations` requires downstream struct literals without `..Default::default()` to add the field. Existing serialized requests lacking it deserialize with an empty list. Since 0.7.0, native declarations are accepted on Responses; Chat, Anthropic Messages and Gemini still reject them.

## Outstanding

- Wire declarations are now encoded for Responses; remaining work is provider capability tests and real endpoint validation.
- Complete stream interleaving, reconnect, terminal reconciliation and malformed-frame tests, including namespaced function handling and strict call identity.
- Validate raw and typed replay coexistence, mixed history and all orphan/missing-result policies against actual HTTP request bodies.
- Integrate contract fixtures; source and attribute the full grammar; run compatibility and endpoint capability tests.
- Version/revision and migration notes are published as 0.7.0 (`docs/migration-0.7.md`); real OAuth probes and default enablement remain separate downstream gates.

## Additional replay safety review

Opaque Responses `function_call`, `custom_tool_call`, and their output items are rejected in assistant history: client calls and results must use typed parts so they cannot bypass pending registration or be encoded twice. Hosted provider items remain distinct. A targeted normalizer test covers opaque/typed coexistence and mixed legacy/custom results with identical names and distinct IDs. This does not establish the second-round HTTP body for the mixed case, or complete the stream/function reconciliation work.
