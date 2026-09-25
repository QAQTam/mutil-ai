# mutil-ai 0.2 Handoff

> 日期：2026-09-25
> 版本：`0.2.1`
> 状态：功能已实现并通过本地验收，尚未提交 commit
> API 契约：[`api-freeze-0.2.md`](api-freeze-0.2.md)
> 需求来源：[`consumer-capability-requirements.md`](consumer-capability-requirements.md)

## 1. 本次目标

把 SDK 从“可用的 provider-neutral core”补强为可作为通用 gate backend 的
0.2 版本，重点解决：

- 请求/流/retry/reconnect 的可协作取消；
- usage/cache/reasoning token 不丢失；
- stream tool progress、retry、server tool 状态；
- 结构化 ToolResult 与图片输入；
- provider request typed overrides；
- stateful/incremental history transform hook；
- OpenAI Responses server-side tool opaque item round-trip；
- idle timeout、error taxonomy 和同步调用 wrapper。

原则保持不变：

```text
入口宽松
+ 出口严格
+ 协议显式
+ provider quirks 留在 profile/adapter
+ 不为 QAQH 特判
```

## 2. 当前状态

已完成：

- P0：Cancellation、Usage、StreamEvent、ToolResult、Provider Request Profile、
  RequestTransform、ServerTool。
- P1：Blocking wrapper、idle timeout、动态 profile 能力确认、图片输入、
  Error taxonomy。
- P2 中已完成：usage/cache audit 摘要、ResponseMetadata helper、`ChatResponse.raw`
  边界说明。
- 公共 API 边界收缩：库不再导出 Agent runtime 和工具执行器，最小实现移到
  `examples/minimal_agent.rs`。

所有冻结验收命令通过：

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets --quiet
cargo test --all-targets --features blocking --quiet
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --quiet
```

当前工作区有大量未提交修改。接手后先执行 `git status` 和 `git diff`，确认是否需要
按功能拆成多个 commit。

## 3. 0.1 -> 0.2 迁移

0.2 是 breaking 版本，不保证与 0.1 的 struct literal 或 exhaustive enum match
源码兼容。

| 类型 | 0.2 变化 |
|---|---|
| `ChatRequest` | 新增 `server_tools` |
| `Part` | 新增 `Image`、`ProviderItem` |
| `ToolResult` | 新增 `parts`、`status`、`metadata`；`content` 保留 |
| `Usage` | 新增 total/cache/reasoning/raw |
| `StreamEvent` | 新增 `ToolCallProgress`、`ServerToolStatus`、`Retrying`、`Error` |
| `RequestOptions` | 新增 cancellation、transform、provider request、idle timeout |
| `ProviderProfile` | 新增 `request_options` |
| `ChatResponse` | 新增 `report` |
| `ErrorKind` | 新增 Cancelled、ContextLengthExceeded、ContentFiltered |
| `Error` | 新增 Cancelled、Timeout |
| `Agent` / `AgentBuilder` / `AgentReply` | 从库公共 API 移除；改用 `ModelAdapter`，参考 `examples/minimal_agent.rs` |
| `Tool` / `FunctionTool` / `ToolRegistry` / `tool_fn` | 从库公共 API 移除；由下游 runtime 自己实现 |

推荐迁移方式：

```rust
let request = ChatRequest {
    messages,
    ..ChatRequest::default()
};

let options = RequestOptions {
    timeout: Some(Duration::from_secs(30)),
    ..RequestOptions::default()
};
```

下游如果有 exhaustive match，需要处理新增 enum variant。

## 4. 公开 API 与语义

### 4.1 Cancellation

入口：

```rust
use mutil_ai::{CancellationToken, RequestOptions};

let token = CancellationToken::new();
let options = RequestOptions::new().cancellation(token.clone());
```

实现位置：

- `src/cancel.rs`
- `src/adapter/mod.rs`
- `src/stream.rs`
- `src/error.rs`

语义：

- token clone 共享状态；
- 连接前取消不发送请求；
- response body 读取、header injector、retry delay、reconnect delay 均响应取消；
- 流中取消返回 `Err(Error::Cancelled)`；
- `ErrorKind::Cancelled` 不可重试；
- audit 最终 outcome 为 `Cancelled`；
- 已产生 `Done` 后，取消不会覆盖已完成结果；
- 消费方停止 poll 后，只有 drop stream 能被 audit 观察到，不能主动返回错误。

### 4.2 Usage

`Usage` 现在包含：

```text
input_tokens
output_tokens
total_tokens
cache_read_tokens
cache_write_tokens
reasoning_tokens
raw
```

解析位置：

- OpenAI Chat：`prompt_tokens_details.cached_tokens`、
  `completion_tokens_details.reasoning_tokens`；
- OpenAI Responses：`input_tokens_details.cached_tokens`、
  `output_tokens_details.reasoning_tokens`；
- Anthropic：`cache_read_input_tokens`、`cache_creation_input_tokens`，多段事件
  按键覆盖合并；
- Gemini：`cachedContentTokenCount`、`thoughtsTokenCount`、
  `totalTokenCount`。

audit 会移除 `Usage.raw`，只保留 typed summary。不要修改这个隐私边界，除非先更新
`AuditConfig` 和文档。

### 4.3 StreamEvent

新增：

```rust
ToolCallProgress {
    index,
    id,
    name,
    arguments_so_far,
}
ServerToolStatus {
    tool,
    call_id,
    state,
}
Retrying {
    attempt,
    max_attempts,
    delay,
    reason,
}
Error {
    error: StreamError,
}
```

错误规则：

- 终止错误使用 `Result::Err`，不发送 `Done`；
- `StreamEvent::Error` 只保留给可恢复/诊断事件；
- 当前 mapper 尚未主动产生 `Error` 事件，如需使用先补 producer 和测试；
- `Retry` 是 provider SSE retry 指示；
- `Retrying` 是 SDK 已安排 reconnect 的稳定事件。

### 4.4 CompletionReport

入口：

```rust
response.report
```

流式最终报告在：

```rust
StreamEvent::Done { response, .. }
```

报告包含：

```text
transform_applied
normalization stats
wire downgrade / unsupported actions
```

实现位置：

- `src/report.rs`
- `src/adapter/mod.rs::completion_report`

禁止把 message、reasoning、tool 参数、tool result 内容或 provider body 放入报告。

### 4.5 ToolResult 与图片

新增：

```rust
Part::Image { image }
ToolResultPart::{Text, Json, Image}
ImageSource::{Url, Base64, FileRef}
```

adapter 行为：

- OpenAI Chat / Responses：图片输入转为 URL 或 data URL；结构化 tool result
  降级为文本并写 `WireAction::Downgraded`；
- Anthropic：URL/base64 映射为 image source；tool result 可保留 image block；
- Gemini：base64 转 `inlineData`，URL/ref 转 `fileData`；
- ToolResult status/metadata 不进入 provider wire，通过 report 标记未发送字段。

### 4.6 RequestTransform

入口：

```rust
let options = RequestOptions::new().request_transform(my_transform);
```

trait：

```rust
pub trait RequestTransform: Send + Sync {
    fn transform(&self, request: &ChatRequest) -> Result<ChatRequest>;
}
```

执行顺序：

```text
Endpoint profile validation
-> RequestTransform
-> capability gating
-> normalization
-> wire body
-> send
```

语义：

- 每个逻辑请求只执行一次；
- retry/reconnect 复用 transform 后的 body；
- transform 失败不发送请求；
- transform 结果仍经过 normalization 和 capability gate；
- audit 只记录 `TransformApplied`，不记录内容。

### 4.7 ProviderRequestOptions

入口：

```rust
ProviderProfile.request_options
RequestOptions.provider_request
```

字段：

```text
tool_call_content
require_provider_parameters
do_sample
include_stream_usage
prompt_cache_key
user
```

优先级：

```text
request typed value
> profile typed default
> adapter default
```

typed request options 在 `extra_body` 合并后应用，因此同名字段由 typed option 覆盖。
协议必需字段仍不能被覆盖。

注意：`do_sample`、`require_provider_parameters` 目前作为通用顶层字段应用。接入真实
provider 前应逐协议确认 wire 位置，必要时改为 profile 内协议专属映射。

### 4.8 ServerTool

请求：

```rust
ChatRequest::user("search").server_tools([ServerTool::WebSearch])
```

当前仅 OpenAI Responses 实现：

- `web_search`、`url_context`、`file_search`、custom server tool 声明；
- `web_search_call` 等 output item 转为 `Part::ProviderItem`；
- 下一轮原样回放 provider item；
- 流式 output item 产生 `ServerToolStatus`；
- 其他协议明确返回 `Unsupported`；
- `EndpointAdapter` 需要 profile 设置：

```rust
profile.capabilities.server_side_state = true;
```

`ProviderState` Debug 只输出 format 和字节数，不展开 opaque data。

### 4.9 Idle timeout

入口：

```rust
RequestOptions::new().idle_timeout(Duration::from_secs(30))
```

语义：

- 统计 SSE 连续无字节时间；
- 有 reconnect policy 时进入 reconnect；
- 无 reconnect 时返回 `ErrorKind::Timeout`；
- cancellation 优先于 idle timeout；
- 只有 stream 被 poll 时 deadline 才能被驱动。

### 4.10 Blocking wrapper

feature：

```toml
mutil-ai = { features = ["blocking"] }
```

入口：

```rust
BlockingAdapter<A>
BlockingStream
```

语义：

- 内部持有独立 runtime；
- 在 Tokio async context 中构造或调用会返回 `Error::InvalidRequest`；
- 默认 feature 不引入 runtime。

## 5. 关键文件

```text
src/cancel.rs                 CancellationToken
src/report.rs                 CompletionReport / WireAction
src/transform.rs              RequestTransform
src/blocking.rs               feature-gated sync wrapper
src/types.rs                  ChatRequest / Part / ToolResult / Usage / ServerTool
src/headers.rs                RequestOptions / ProviderRequestOptions 接入
src/profile.rs                ProviderProfile / ProviderRequestOptions
src/error.rs                  Cancelled / Timeout / taxonomy
src/stream.rs                 cancellation / idle timeout / StreamEvent
src/adapter/mod.rs            transport retry、取消、report、profile 合并
src/adapter/openai_chat.rs
src/adapter/openai_responses.rs
src/adapter/anthropic.rs
src/adapter/gemini.rs
src/adapter/endpoint.rs
examples/minimal_agent.rs
tests/cancellation.rs
tests/consumer_0_2.rs
tests/blocking.rs
docs/api-freeze-0.2.md
```

## 6. 测试覆盖

新增测试覆盖：

- 挂起 complete 取消；
- stream 中取消且不 reconnect；
- retry delay 取消且不二次发送；
- idle timeout；
- transform 执行、失败不发请求；
- typed request options 覆盖 profile；
- OpenAI Chat 扩展 usage；
- 结构化 tool result 降级报告；
- OpenAI Responses server tool opaque item round-trip；
- blocking async context 防护；
- context/content error taxonomy。

尚未覆盖：

- `StreamEvent::Error` producer；
- Responses streaming `ServerToolStatus` 的完整 fixture；
- 真实 provider usage/cache fixture；
- cancellation + reconnect + idle timeout 三方同时触发；
- blocking wrapper 的真实 HTTP complete/stream round-trip。

## 7. 已知限制与风险

1. 0.2 是 breaking 版本，下游 exhaustive match 会受影响。
2. `StreamEvent::Error` 目前是 API 预留，mapper 尚未主动发出。
3. `ProviderRequestOptions` 的 `do_sample` 和 `require_provider_parameters`
   只是通用顶层映射，真实 provider wire 需要逐协议复核。
4. Server tool 只实现 OpenAI Responses，其他协议保持 explicit unsupported。
5. OpenAI Chat/Responses 的结构化 ToolResult 图片当前降级为文本。
6. `CompletionReport` 只报告聚合，不暴露具体消息内容。
7. Blocking wrapper 每个 adapter 创建一个 runtime，不适合大量短生命周期对象。
8. `ChatResponse.raw` 可能很大，默认日志和 Debug 使用需谨慎。

## 8. 下一步建议

按优先级：

1. 为四个协议补带日期的 usage/cache/reasoning fixture。
2. 补 Responses streaming server tool fixture，并决定是否让 mapper 产生
   recoverable `StreamEvent::Error`。
3. 逐协议确认 `do_sample`、`require_provider_parameters`、`user` 的 wire 位置。
4. 补 cancellation + retry + reconnect + idle timeout 的组合测试。
5. 补 blocking wrapper 的本地 HTTP complete/stream 测试。
6. 评估是否按功能拆分当前工作区为多个 commit。
7. 更新 README 的使用示例，加入 cancellation、transform 和 server tool。
8. 在正式发布前检查是否需要 `#[non_exhaustive]` 或后续 1.0 兼容策略。

## 9. 接手检查清单

```text
[ ] git status / git diff 已复核
[ ] 确认 0.2 breaking changes 可接受
[ ] cargo test --all-targets --quiet
[ ] cargo test --all-targets --features blocking --quiet
[ ] cargo clippy --all-targets -- -D warnings
[ ] RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --quiet
[ ] 确认真实 provider fixture 是否必须进入本轮
[ ] 确认是否提交工作区
```
