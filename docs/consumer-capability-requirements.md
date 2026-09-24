# 消费方集成能力需求（通用 SDK）

> 状态：需求登记，供 SDK 维护者评审。
> 日期：2026-09-25
> 提出方：QAQH backend / gate 适配评估
> 约束：**SDK 不得为 QAQH 做特判。** 本文只登记通用能力、扩展点和验收方式；
> 具体 provider/消费者策略留在各自 adapter 或 profile 配置中。

## 0. 背景

当前评估的接入形态是：

```text
QAQH runtime
  -> qaqh-gate facade
  -> generic bridge
  -> mutil-ai ModelAdapter / normalization
```

目标不是把 SDK 变成 QAQH 专用 gateway，而是让它具备作为通用 provider core 的
必要扩展点。QAQH 只是第一个高要求消费者；下面每项都应以通用语义设计，并可由
其他 agent/gateway 复用。

## 1. 优先级

- **P0**：没有它就无法完成等价替换，或会造成数据丢失/无法取消。
- **P1**：首轮替换可通过 bridge 临时绕开，但长期应进 SDK 或稳定文档化。
- **P2**：增强项，不阻塞 gate 替换。

---

## 2. P0：通用能力

### 2.1 请求取消（Cancellation）

**现状**

`RequestOptions` 有 audit 层的 cancellation 记录，但没有可传给 adapter 的取消句柄；
`ModelAdapter::stream_with` 也不能在流开始后主动中断。

**通用需求**

- 增加通用取消 token / handle，例如：

```rust
pub trait CancellationToken: Send + Sync {
    fn is_cancelled(&self) -> bool;
    fn cancel(&self);
}
```

或等价的 `Arc<AtomicBool>` 封装。
- 取消句柄应能进入：
  - `RequestOptions`
  - `ModelAdapter::complete_with`
  - `ModelAdapter::stream_with`
- 行为要求：
  - 连接建立前取消：不发送请求。
  - 流式读取中取消：尽快停止读取，返回结构化 `Cancelled`。
  - 重试/重连等待中取消：立即结束等待。
  - 取消与 retry/reconnect 组合时，不得重新发起请求。
- 取消结果应有稳定错误种类，而不是只靠字符串。

**验收**

- 本地 HTTP server 挂起响应时，取消在有限时间内返回。
- 流中取消不会触发重试。
- 取消与 `stream_reconnect` 组合时不会重连。
- audit 中 cancellation 与最终错误种类一致。

### 2.2 完整 Usage 与 raw passthrough

**现状**

`Usage` 只有：

```rust
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
}
```

**通用需求**

不同 provider 会返回 cache hit/miss、cache write、reasoning tokens、total 等字段。
SDK 不需要解释所有语义，但不能丢。

建议：

```rust
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub cache_read_tokens: Option<u64>,
    pub cache_write_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
    /// 未归一的 provider usage 原文，供消费者按需读取。
    pub raw: Option<serde_json::Value>,
}
```

或者保留 `Usage` 轻量，但提供稳定的 `raw_usage: Option<Value>`。

**验收**

- OpenAI Chat、Responses、Anthropic、Gemini 的 cache 字段至少能通过 raw 或 typed
  字段取回。
- `StreamEvent::Usage` 与 `Done.response.usage` 不丢失终值。
- 多段 usage 合并规则有文档和 fixture。

### 2.3 StreamEvent 完整性

**现状**

`StreamEvent` 有：

```text
Start / TextDelta / ReasoningDelta / ToolCallDelta / Usage / Retry / Done
```

消费方还需要工具进度、服务端工具状态和更完整的 retry 信息。

**通用需求**

建议补齐：

```rust
pub enum StreamEvent {
    Start { ... },
    TextDelta { ... },
    ReasoningDelta { ... },
    ToolCallDelta { ... },

    /// 已累积到当前时刻的工具参数，便于 UI 增量展示。
    ToolCallProgress {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments_so_far: String,
    },

    /// 通用 server-side tool / built-in tool 状态。
    ServerToolStatus {
        tool: String,
        call_id: Option<String>,
        state: ServerToolState,
    },

    Usage { ... },

    /// 重试开始；字段是通用语义，不是 provider 专属。
    Retrying {
        attempt: u32,
        max_attempts: u32,
        delay: Duration,
        reason: String,
    },

    /// 可恢复/不可恢复的流错误事件；与 Result::Err 的关系需明确。
    Error { ... },

    Done { ... },
}
```

如果不想扩展 enum，也可以提供原始 provider event hook，但必须能让消费者在
不 fork SDK 的情况下构造等价事件。

**验收**

- OpenAI Chat / Responses / Anthropic 的 tool-call 流都能产出稳定的 progress。
- retry 事件包含 attempt、上限、delay、reason。
- enum 新增 variant 的兼容策略写入 API freeze 文档。

### 2.4 结构化 ToolResult

**现状**

```rust
pub struct ToolResult {
    pub call_id: Option<String>,
    pub name: String,
    pub content: String,
    pub is_error: bool,
}
```

`content: String` 对多模态、结构化工具结果、内容引用和终态元数据不够。

**通用需求**

- ToolResult 内容应支持多 Part 或至少支持结构化 envelope：

```rust
pub struct ToolResult {
    pub call_id: Option<String>,
    pub name: String,
    pub parts: Vec<Part>,
    pub is_error: bool,
    pub status: Option<String>,
    pub metadata: Option<serde_json::Value>,
}
```

- 至少需要一种方式表达：
  - 文本结果
  - 图片/附件结果
  - 结构化 JSON 结果
  - provider 可接受的错误结果
  - 终态/展示元数据（SDK 不解释，只透传）
- adapter 仍需把不支持的结构降级为 provider 可接受的形式，并产生明确报告。

**验收**

- 文本、结构化 JSON、图片结果能跨 adapter round-trip 或明确降级。
- 终态 metadata 不进入 provider 不支持字段，但可被消费者取回。

### 2.5 通用 Provider Request Profile 扩展

**现状**

`ProviderProfile` 已覆盖 reasoning、tools、stream、usage、capabilities，但一些
跨厂商请求开关仍只能塞 `extra_body`。

**通用需求**

把下面这些做成 profile/request 字段，而不是 QAQH 特判：

- `tool_call_content_null`
- `require_provider_parameters`
- `do_sample`
- `include_stream_usage`
- `prompt_cache_key`
- `user` 字段的发送策略
- `max_tokens` / `max_output_tokens` 语义
- `thinking` 对象的发送模式
- `include` / encrypted reasoning 的发送策略

要求：

- reserved canonical field 冲突仍由 adapter 校验。
- profile 字段与 request-level extra 的优先级必须明确。
- 未知/不支持的字段应在能力门控下明确报错或忽略，不能静默变成 provider 非法 body。

**验收**

- profile 能表达至少两种不同 thinking 请求形态。
- `extra_body` 与 profile 默认值的覆盖关系有测试。
- reserved field 不能被 extra_body 覆盖。

### 2.6 Stateful / Incremental History

**现状**

SDK 的 `ChatRequest` 始终是完整 messages 列表；没有“只发送增量历史”的通用模式。

**通用需求**

不要加入 QAQH 专属字段。提供通用 history policy 或 request transform hook：

```rust
pub enum HistoryMode {
    Full,
    Incremental { since: Option<String> },
}
```

或：

```rust
pub trait RequestTransform: Send + Sync {
    fn transform(&self, request: ChatRequest) -> Result<ChatRequest>;
}
```

具体增量计算可由消费者实现，SDK 只负责：

- 在 normalization 前/后调用 transform 的明确时点；
- 保证 transform 结果仍经过 normalization；
- 在 audit/诊断中记录“发生了 transform”，不记录内容。

**验收**

- 同一 adapter 能接收 full 和 transform 后的增量请求。
- transform 后工具配对仍正确。
- transform 失败时不发送请求。

### 2.7 Server-side / Built-in Tool 通用支持

**现状**

SDK 没有 built-in `web_search` 的 typed 请求配置，也没有服务端工具状态事件和
opaque item 回传语义。

**通用需求**

提供通用 server tool 模型，而不是 `web_search` 特判：

```rust
pub enum ServerTool {
    WebSearch,
    UrlContext,
    FileSearch,
    Custom { name: String, config: Value },
}
```

并要求：

- request 中可声明 server tools；
- response/stream 中可回传 opaque server tool item；
- 下一轮可原样回放同一 provider 的 opaque item；
- 跨 provider 默认丢弃并产生 normalization report；
- 状态事件使用通用 `ServerToolStatus`。

**验收**

- OpenAI Responses 的 server tool item 能 round-trip。
- 不支持 server tool 的 provider 明确报 unsupported 或按 profile 降级。
- opaque item 不进入日志默认展开。

---

## 3. P1：可先由 bridge 绕开，但建议补

### 3.1 Blocking / sync adapter

**现状**

SDK 是 async-first，消费方可能是同步 gate/runtime。

**需求**

二选一：

1. 提供官方 blocking wrapper，内部使用独立 runtime，明确禁止在 async context 中调用；
2. 提供文档化的 async-to-sync bridge 模式，并保证不会在当前 runtime 上死锁。

**验收**

- 同步调用能跑通 `complete` 和 `stream`。
- 在 tokio runtime 内误用时有明确错误或文档警告。

### 3.2 Stream idle timeout

**现状**

有 request timeout、retry、reconnect，但没有“连续无字节”的空闲看门狗。

**需求**

通用 `idle_timeout` 配置，超时归入可重试 timeout，并受 cancellation 约束。

**验收**

- 半开连接在 idle timeout 后终止。
- 超时后可按 policy 重试或返回稳定错误。

### 3.3 动态 Profile 构造

**现状**

built-in `ProfileRegistry` 是静态集合。

**需求**

消费者应能安全地构造并注入自定义 `ProviderProfile` / `ModelProfile`，不必修改
SDK 内置 registry。自定义 profile 的序列化/诊断/审计格式应稳定。

**验收**

- 自定义 profile 不依赖全局可变注册表。
- profile 冲突和优先级有明确规则。

### 3.4 图片输入的通用形态

**现状**

`Part::ImageUrl` 主要面向 URL。

**需求**

如果消费方持有 base64、文件引用或内容寻址引用，应提供通用转换/输入形态，
避免所有消费者自己拼 data URL。SDK 可以只提供 `ImageData`/`ImageRef` 的中立
Part，由 adapter 转成 provider wire。

**验收**

- URL、base64、引用三类输入至少有一类能被 adapter 接受或明确降级。
- 大图片不因为 SDK 内部重复 clone 造成额外峰值。

### 3.5 Error taxonomy 细化

**需求**

继续使用 typed `ErrorKind`，并补充跨 provider 通用分类：

- context length exceeded
- content filter / safety
- rate limited / overloaded
- transient connection
- auth/permission
- invalid request

**验收**

- 分类不依赖 provider 名称或字符串包含匹配。
- 每个分类都有 retryable 语义。

---

## 4. P2：增强项

- `AuditEvent` 增加通用 usage/cache 汇总字段，但不记录正文。
- `ResponseMetadata` 暴露 response header 白名单和 request id。
- `ChatResponse.raw` 的跨 adapter 稳定性和大小边界写入文档。
- 提供 provider fixture 生成器；live probe 仍不进入正常请求路径。
- 提供 `normalize_report` 的稳定聚合指标，便于比较 bridge 前后行为。

---

## 5. 明确非目标

以下不应进入 SDK 核心：

- `qaqh` 名称、QAQH endpoint id、QAQH seed、QAQH timeline/turn 类型。
- 特定 provider 的 QAQH 业务字段特判。
- QAQH 的 DSML/XML tool parser。
- QAQH 的 skill envelope、todo、subagent 语义。
- QAQH 的 agent loop、timeline、权限审批、tool execution。
- 以 base URL 猜协议。
- 为兼容 QAQH 而破坏现有 normalization 默认语义。

这些应由 QAQH bridge 或 adapter profile 处理。

---

## 6. 推荐落地顺序

1. **Cancellation + Usage + StreamEvent 元数据**：P0，最先做。
2. **结构化 ToolResult + Provider Request Profile**：P0，决定能否无损替换。
3. **Stateful transform hook + ServerTool 模型**：P0/P1，覆盖 QAQH 特殊端点。
4. **Blocking wrapper + idle timeout + 动态 profile**：P1。
5. **Audit/metrics/fixture 增强**：P2。

每项都应遵循：

```text
通用 API
+ provider adapter 实现
+ fixture / integration test
+ api-freeze 兼容性说明
```

而不是：

```text
if consumer == qaqh { ... }
```

---

## 7. 最小替换验收集

当 SDK 作为 gate backend 时，至少要通过：

- 同一消息历史的请求 body golden test。
- 同一 SSE fixture 的事件序列 golden test。
- tool call / tool result 配对与终态保真。
- reasoning / provider state 原样回放。
- usage/cache 不丢失。
- cancellation、retry、idle timeout、reconnect 组合。
- full history 与 incremental history 两种模式。
- 不支持能力时的显式 unsupported/降级报告。
