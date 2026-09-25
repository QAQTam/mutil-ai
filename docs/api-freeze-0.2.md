# mutil-ai API Freeze 0.2

> 状态：0.2 能力冻结候选
> crate：`mutil-ai` / `mutil_ai`
> version：`0.2.1`
> edition：`2024`
> date：`2026-09-25`
> 实现交接：见 [`handoff-0.2.md`](handoff-0.2.md)

## 1. 定位

0.2 在 0.1 的中立消息、strict normalization、显式 endpoint/profile 基础上，
补齐作为通用 provider core 所需的取消、完整 usage、结构化 tool result、请求
transform、server-side tool 和同步 wrapper。

0.1 文档保留为历史冻结记录。0.2 是 breaking 版本，不承诺与 0.1 的 struct
literal 或 exhaustive enum match 源码兼容。

## 2. Breaking changes

- `ChatRequest` 新增 `server_tools`。
- `Part` 新增 `Image` 和 `ProviderItem`。
- `ToolResult` 新增 `parts`、`status`、`metadata`；`content` 保留为文本 fallback。
- `Usage` 新增 total/cache/reasoning/raw 字段。
- `StreamEvent` 新增 `ToolCallProgress`、`ServerToolStatus`、`Retrying`、`Error`。
- `RequestOptions` 新增 cancellation、request transform、typed provider request
  和 idle timeout 字段。
- `ProviderProfile` 新增 `request_options`。
- `ChatResponse` 新增 context-free `report`。
- `ErrorKind` 新增 `Cancelled`、`ContextLengthExceeded`、`ContentFiltered`。
- `Error` 新增 `Cancelled`、`Timeout`。
- 移除库级 `Agent`、`AgentBuilder`、`AgentReply`、`complete_once`。
- 移除库级 `Tool`、`FunctionTool`、`ToolRegistry`、`tool_fn`。

Agent runtime 和工具执行属于下游策略，不再作为 SDK 公共能力。库的稳定边界是
`ModelAdapter::complete/complete_with/stream/stream_with`；最小 runtime 参考实现
见 [`../examples/minimal_agent.rs`](../examples/minimal_agent.rs)。

调用方使用 `..Default::default()`、`..ChatRequest::default()` 或构造器时迁移
成本较小。所有公开 enum 的 exhaustive match 需要补新分支。

## 3. Cancellation

```rust
let token = CancellationToken::new();
let options = RequestOptions::new().cancellation(token.clone());
```

冻结语义：

- `CancellationToken` clone 共享状态。
- 请求建立连接前取消时不发送请求。
- response body 读取、retry delay、stream reconnect delay 均受取消约束。
- 流开始后取消返回 `Err(Error::Cancelled)`，不会重新发起请求。
- `ErrorKind::Cancelled` 稳定且不可重试。
- audit 最终 outcome 使用 `AuditOutcome::Cancelled`。
- 已生成 `Done` 后取消不覆盖已完成响应。

## 4. Usage

`Usage` 保留中立字段并增加：

```text
total_tokens
cache_read_tokens
cache_write_tokens
reasoning_tokens
raw
```

冻结语义：

- OpenAI Chat、Responses、Anthropic、Gemini 的已知 usage 字段进入 typed 字段。
- provider-native usage 原文进入 `raw`。
- Anthropic 多段 usage 按字段覆盖合并，raw object 按键覆盖合并。
- `StreamEvent::Usage` 与 `Done.response.usage` 使用同一合并结果。
- audit 只接收去掉 `raw` 的 usage summary。
- SDK 不根据 input/output 猜测缺失的 total。

## 5. StreamEvent

新增：

- `ToolCallProgress`：当前累积工具参数。
- `ServerToolStatus`：provider-hosted tool 状态。
- `Retrying`：attempt、max_attempts、delay、reason。
- `Error`：可恢复错误的稳定摘要。

错误契约：

- 终止错误返回 `Result::Err`，不发送 `Done`。
- `StreamEvent::Error` 只表达可恢复或附带诊断的流错误。
- 同一终止错误不同时作为 `Error` 事件和 `Err` 重复发送。
- `Retry` 保留为 provider 原始 SSE retry 指示；`Retrying` 表示 SDK 已安排重连。

## 6. CompletionReport

`ChatResponse.report` 和流式 `Done.response.report` 暴露：

```text
transform_applied
normalization stats
wire downgrade / unsupported actions
```

报告不包含 message、reasoning、tool 参数、tool result 内容或 provider body。
`WireAction` 用于表达结构化 tool result 降级、opaque provider item 丢弃和
server tool 不支持。

## 7. ToolResult 与图片

`ToolResult` 支持：

- 文本；
- 结构化 JSON；
- 图片 URL、base64、file ref；
- status 与 metadata。

`Part::Image` 使用 `ImageSource`：

```text
Url
Base64 { media_type, data }
FileRef { uri, media_type }
```

adapter 行为：

- OpenAI Chat / Responses：base64 转为 data URL；非文本 tool result 降级为文本并报告。
- Anthropic：URL、base64 映射为 image block；tool result 可保留 image block。
- Gemini：base64 映射为 `inlineData`，URL/ref 映射为 `fileData`。
- 不支持的结构必须产生 `WireAction`，不得静默丢字段。

## 8. Request Transform 与 Provider Request

`RequestTransform` 在 capability gating 和 normalization 之前执行一次：

```rust
let options = RequestOptions::new().request_transform(my_transform);
```

冻结语义：

- transform 失败时不发送请求。
- retry/reconnect 复用 transform 后的 body。
- transform 结果仍经过 normalization 和 capability gate。
- audit 只记录 `TransformApplied`，不记录内容。

`ProviderRequestOptions` 支持：

```text
tool_call_content
require_provider_parameters
do_sample
include_stream_usage
prompt_cache_key
user
```

优先级：request typed value > profile typed default > adapter default。
typed request options 在 `extra_body` 合并后应用，因此可覆盖同名字段。协议必需
字段仍不可覆盖。

## 9. Server-side Tool

`ServerTool` 支持 WebSearch、UrlContext、FileSearch 和 Custom。OpenAI Responses
支持：

- request 中声明 server tools；
- 输出 `web_search_call` 等 opaque item；
- 下一轮原样回放 item；
- 流式 `ServerToolStatus`。

其他协议默认返回 `Unsupported`。Endpoint profile 必须显式设置
`capabilities.server_side_state = true` 才允许 server tools。

`ProviderState` 的 Debug 输出只显示 format 和字节数，不展开 opaque data。

## 10. Idle timeout

```rust
let options = RequestOptions::new().idle_timeout(Duration::from_secs(30));
```

超时按“连续无字节”计算。存在 reconnect policy 时按该 policy 重连；否则返回
`ErrorKind::Timeout`。取消优先于 idle timeout。

## 11. Blocking wrapper

启用 `blocking` feature 后提供：

```text
BlockingAdapter<A>
BlockingStream
```

- 内部使用独立 runtime。
- 在 Tokio async context 内调用时返回 `Error::InvalidRequest`。
- 默认 feature 不引入 runtime。

## 12. Error taxonomy

新增稳定分类：

```text
Cancelled
ContextLengthExceeded
ContentFiltered
```

context/content 分类只使用清洗后的结构化 code/type/status 精确 token 匹配，
不使用 provider 名或自然语言 message 子串匹配。无法结构化分类时回退到 HTTP
status 分类。

## 13. 验收命令

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets --quiet
cargo test --all-targets --features blocking --quiet
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --quiet
```
