# ChatGPT / Codex 认证集成设计与实现交接

> 官方资料核对日期：2026-10-13。代码基线：mutil-ai 0.4.1。
> 状态：**设计提案，尚未实现**。本文的接口名与模块划分是建议，不是现有 API。
> 范围：开源、本地运行客户端的 Sign in with ChatGPT（SIWC）、套餐推理，以及可选的 Codex app-server 集成。

## 1. 结论：客户端负责登录，SDK 负责推理与凭据接入

**不要在 mutil-ai 核心 SDK 中默认打开浏览器、启动登录回调服务器、选择用户账号或保存 refresh token。**
认证不是全部放客户端、也不是全部放 SDK；按责任拆分：

| 层 | 应负责 | 不应负责 |
| --- | --- | --- |
| 桌面/CLI 客户端 | 登录入口、系统浏览器、loopback listener、账号选择、用户同意、登出、凭据存储 | 复制各 provider 的请求序列化和 SSE 解析 |
| 客户端认证服务/可选 companion crate | OAuth/OIDC 校验、刷新、撤销、会话隔离、原子存储；向推理层提供短期 access token | 自动切换计费方式、执行 agent 工具 |
| mutil-ai 核心 | 可刷新的认证来源、发送 Bearer、ChatGPT 套餐请求能力校验、统一响应/错误、安全预算 | 持有应用用户数据库、默认读 Codex 凭据、默认存 refresh token |
| 可选 Codex bridge | app-server 子进程生命周期、RPC/事件、线程恢复、批准与工具执行边界 | 冒充普通无副作用的模型调用 |

优先交付一个本地客户端示例和最小 SDK 扩展；确实被多个客户端复用后，再考虑独立认证 crate。
浏览器授权流程可以写成库，但应由客户端显式调用，不能成为 SDK 初始化或请求的隐式副作用。

本节是**项目架构建议**。官方也将应用账号、会话与授权管理交给应用自身；SIWC 提供已验证身份与用户明确授予的权限。[官方 Quickstart](https://developers.openai.com/siwc/quickstart)

## 2. 三条路径必须区分

### A. 普通 API key 调用

保留现有 OpenAI Responses / Chat 等行为；使用 API Platform 凭据与计费路径。
OAuth 功能不能改变既有 `api_key()`、`OPENAI_API_KEY` 或其它 provider 的语义。

### B. SIWC 身份 + ChatGPT 套餐推理（本项目优先）

由用户授权本地应用自己的注册，获得 OAuth 凭据，向公开 Responses API 发送符合套餐能力约束的请求。
“登录成功”和“获准使用套餐”是不同状态；身份 scope 不提供会话历史、API key 或套餐推理权限。
官方目前提供开源/本地应用路径，合资格用户可使用自己的套餐；付费/远程托管产品需单独确认开放资格。
SDK 的 MIT / Apache-2.0 商业许可不能代替这项服务资格。[接入范围](https://developers.openai.com/siwc/token-sharing-open-source)

### C. Codex app-server（需要完整 coding agent 时）

客户端已取得 SIWC token 后，可以将它交给自己的 Codex app-server 子进程。
这是 agent runtime 集成，不只是换一个模型 endpoint。是否允许 shell/MCP、文件访问和工具执行仍须由客户端明确管理。
此配置不需要再做一遍独立 Codex 登录。[官方集成指南](https://developers.openai.com/siwc/token-sharing-open-source/codex-app-server)

不要混用上述三条路径，不把身份登录 token、套餐 access token、API key 当成等价凭据。
不要使用 ChatGPT `backend-api` 代替官方公开推理接口。[模型与推理指南](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference)

## 3. 0.4.1 代码现状与修改入口

下表是本仓库源码观察，**不是官方产品限制**：

| 文件 | 当前行为 | 实现者需要处理 |
| --- | --- | --- |
| [`src/headers.rs`](../src/headers.rs) | `HeaderInjector::inject` 异步返回 headers；可信注入器可设置 Authorization | 可复用接入点，但绑定的会话不能来自任意不可信 `RequestContext` 字符串 |
| [`src/adapter/mod.rs`](../src/adapter/mod.rs) | `resolve_api_key` 需要显式 key 或环境变量；`prepare_headers` 随后才运行注入器 | 必须显式支持 OAuth-only 认证来源，不能靠虚假占位 key 绕过检查 |
| [`src/adapter/openai_responses.rs`](../src/adapter/openai_responses.rs) | `complete_with` 普通 JSON；`stream_with` 设置 stream；请求可发 temperature/max_output_tokens；工具默认 flat function | 套餐路径需独立校验/序列化规则和仅流式线协议；不要修改普通 API 默认规则 |
| [`src/transform.rs`](../src/transform.rs) | `RequestTransform` 在 normalize 前处理中立请求 | 可做原型，但不能作为所有最终 wire 字段与扩展参数的唯一防线 |
| [`src/profile.rs`](../src/profile.rs) | provider/profile 扩展机制 | 可表达请求能力配置；不要把套餐差异分散到下游 agent 循环 |
| [`src/security.rs`](../src/security.rs)、[`src/retry.rs`](../src/retry.rs)、[`src/stream.rs`](../src/stream.rs) | 总 deadline、取消、响应预算、重试/重连 | 必须保留；新增刷新不能绕开总预算或无限延长一次请求 |
| [`src/error.rs`](../src/error.rs) | 现有传输/流错误体系 | 接入套餐错误时检查可结构化读取程度；不能逼 UI 解析 Display 文本 |

尤其不能声称“装一个 HeaderInjector 就已支持 OAuth”：Responses adapter 目前先解析 key。
`request.extra_body` 也不能覆盖 model/input/stream 等保留字段，不能拿它规避 SDK 校验。
当前 Responses mapper 已处理 completed/failed/incomplete，但仍须检查异常 EOF 是否能被客户端误判成功。

### 推荐扩展方式（未冻结）

- 显式区分 API-key 模式与 ChatGPT-plan 模式，默认维持 API-key 模式。
- 用 builder 或独立 wrapper/companion 接入认证来源；尽量避免给公开可构造结构体新增必填字段。
- 概念上需要 `access_token(session)`、`invalidate_session(session)` 和套餐 capability 校验；名称不代表现有 API。
- 认证服务保存 refresh token；核心推理 SDK 只获取短期 Bearer，以及必要的权限/会话绑定结果。
- 长期复用同一认证服务；刷新不能只发生在构造 adapter 时。也不要通过全局环境变量逐次切换账号。
- 若先复用 HeaderInjector：客户端固定绑定会话，SDK 必须提供显式 OAuth-only 入口并完成目的地址校验。
- `complete_with` 的套餐路径可内部消费 SSE 后返回最终响应，或明确返回 Unsupported；不能发普通非流式 HTTP。

实现前根据实际 public API 变动重新判断版本号；不要预先认定可以无破坏性地塞进某个 patch。

## 4. 客户端认证流程

以下是核对官方流程后的简要交接；参数细节以实施时官方文档为准，不复制 Codex 的固定 client ID。

### 4.1 新注册与重新登录

1. 客户端先创建/恢复本 host 的稳定、opaque `ext_agent_host_id`；它是标识符，不是认证凭据。
2. 绑定 `127.0.0.1` listener 后再打开系统浏览器；每次尝试生成新的 state、nonce 和 PKCE S256。
3. 新注册使用 `client_id=dynamic_agent_client` 及应用自己的 `agent_name_hint`。
4. 请求身份 scopes 与套餐 scopes：`openid profile email offline_access resource.invoke chatgpt.tokens.use.direct`。
5. 回调先验证 state，再处理 error/code；新注册取得 **issued client_id** 后才交换 code。
6. 验证 ID token 签名、issuer、audience、expiry、nonce；检查实际 granted scopes。
7. 验证完成后才更新所选账号记录；后续登录复用 issued client_id，不再使用注册入口 ID。

| 项目 | 当前官方值/约束 |
| --- | --- |
| Authorize | `https://auth.openai.com/api/accounts/authorize` |
| Token | `https://auth.openai.com/api/accounts/oauth/token` |
| Resource | `https://api.openai.com/v1` |
| Callback | HTTP `127.0.0.1`；后续端口可变，scheme/host/path 保持一致；同一次授权/交换 URI 必须完全一致 |

开源公共客户端流程不需要 client secret。issued client_id 和已验证账号绑定，不能把注册入口 ID 存为实际 client ID。
重新登录若回调返回不同的 client_id，不能直接覆盖原账号记录；客户端必须验证对应关系。
以上流程来源：[注册与登录](https://developers.openai.com/siwc/token-sharing-open-source/sign-in)。

### 4.2 会话状态设计（项目建议）

```text
SignedOut -> Authorizing -> IdentityOnly / PlanReady
PlanReady -> Refreshing -> PlanReady / ReauthRequired
任何状态 -> SigningOut -> SignedOut
网络/服务临时故障 -> 保留注册，报告临时不可用，不自动清空全部账号
```

`IdentityOnly` 表示身份已验证但套餐权限不足，不能偷偷调用推理。
账号选择器只获得显示信息和 opaque session ID，不把 tokens 暴露给 renderer/UI。
不要以 email 为唯一注册键；同一用户可能在不同 workspace/client 注册中有相同 email。
建议以 issuer、validated subject、issued client_id 区分注册，host ID 单独管理。

### 4.3 刷新、轮换与登出

认证服务刷新时使用 issued client_id、保存的 refresh_token、resource，以及 `grant_type=refresh_token`。
刷新成功后整体保存新的 token set；同一 session 的刷新应串行化，跨进程共享存储时也要协调。
登出先停止新推理，按 discovery 的 revocation endpoint 撤销 renewable session，然后清除本地 tokens。
远端撤销未确认时应如实提示，不把删除本地文件等同于远端撤销成功。
不要因临时网络错误清空凭据；确认 token 失效时才进入重新登录。[账号与会话](https://developers.openai.com/siwc/token-sharing-open-source/profiles-and-sessions)

expiry、轮换信息按实际 token response 保存，不把当前“一小时/三十天”永久硬编码成协议。
保留服务返回的 `earliest_refresh_at` 并按官方说明处理；不自行解析 opaque OpenAI auth metadata。[Token reference](https://developers.openai.com/siwc/token-sharing-open-source/token-reference)

## 5. ChatGPT 套餐推理配置

### 5.1 最小线协议

仅在已确认套餐 grant 的会话下向公开 `POST https://api.openai.com/v1/responses` 发送 Bearer access token。
每次 HTTP 请求使用 `stream: true`、`store: false`；上下文通过 input 数组传递。
模型选择取当前账号目录，使用返回的 slug，不写死模型或把列表出现当作最终授权证明。
首版建议 **纯文本、不带工具、显式单次请求**，先证明完整认证到推理闭环。[模型与推理](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference)

### 5.2 套餐配置不能等同于普通 Responses

当前预览有独立能力限制。实现中为套餐模式建立最终 wire 校验，不仅检查中立请求：

- system 放 instructions 或映射为 developer，不留下显式 system input item。
- 拒绝不支持的 sampling/token-limit 字段，包含 extra_body/profile 扩展带入的字段；不默默吞掉用户配置。
- HTTP 不发送 previous_response_id，不提供持久服务端 conversation/store 语义。
- 工具后续另做：按官方 namespace / additional_tools 约定适配，不能直接复用 flat function 序列化。
- 不把本地可执行工具与 hosted Responses tools 当成同一个能力集合。

完整 unsupported 字段/工具列表见 [Preview limitations](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations)。列表会变化，实现者必须复核，不能将本文当永久协议。
建议首版用保守允许集，扩展时逐项确认；不凭“同为 Responses”放行任意 server_tools/ProviderItem。

### 5.3 完成、失败和计费边界

只有明确收到 `response.completed` 才成功；delta、HTTP 200、EOF 都不充分。
failed/incomplete/截断分别处理，保留结构化 code、param、request ID，诊断内容须限长与脱敏。
套餐耗尽/未授权/暂不可用不能自动回退到 API key，否则会发生用户未同意的计费切换。
切换凭据来源由用户明确选择，UI 显示当前账号和推理来源。[错误与恢复](https://developers.openai.com/siwc/token-sharing-open-source/errors-and-recovery)

## 6. 并发、安全与 TOCTOU 设计要求（项目建议）

这些要求不是新增的官方协议字段，而是基于本项目多请求/重试场景的实现约束：

### 6.1 固定一次逻辑请求的身份

- 开始请求时捕获 session ID、issued client_id、validated identity 与 generation。
- 重试/重连可以取得同一 session 的新 access token，但**不能读取“当前 UI 账号”而切换用户**。
- scope、token、client ID 必须来自同一个经过验证的凭据快照，不能先检查 A 的 scope 后发送 B 的 token。
- UI 切换账号不影响已开始的请求；登出则明确取消相关请求，并拒绝后续重试。
- 若请求前重新授权到不同身份，旧请求停止，不拿新身份继续原会话。

### 6.2 刷新与登出竞态

- 同一 session 用 single-flight；跨进程需要等效锁。避免多个请求同时消费 rotating refresh token。
- 刷新结果提交必须检查 generation / 登出标记；过期任务不能在登出后写回凭据而“复活”会话。
- 原子提交 access/refresh/expiry/scopes，不能出现新 access + 旧 refresh 的半更新记录。
- 网络结果不确定时不能无界重试同一 rotation；参考服务错误与安全恢复策略，不能靠循环 OAuth 解困。
- 认证等待和刷新必须服从请求取消/deadline；共享刷新任务不能因一个等待者取消而破坏其它等待者。

### 6.3 保存与秘密边界

- 认证由 native/main process 持有；浏览器页面/localStorage、渲染进程 IPC、支持工单不得获得 tokens。
- 优先使用系统凭据存储；若文件存储，目录与文件均限制访问，使用安全临时文件 + 原子替换。
- Unix 权限和 Windows ACL 分开实现；Windows 上调用 chmod 并不等于 owner-only。
- 不先 `exists()` 再不安全写入；考虑 symlink/reparse point、可写父目录、替换文件的竞态。
- 将 token set 的 Debug/Display 设为脱敏；HTTP/OAuth 错误、授权 URL、headers、遥测不得带秘密。
- state/code/code_verifier、含 id_token_hint 的授权 URL 同样不能原样记录。
- 凭据存储与会话锁由同一可信组件管理，避免应用与 SDK 各自刷新同一份文件。

### 6.4 目的地址与回调

- 套餐模式仅发往核验过的官方 API origin；不能复用任意 base_url/profile 将 OAuth token 发往自定义网关。
- 继续禁用 HTTP redirects；OAuth token/token-revocation 客户端也要限制跳转和响应大小。
- discovery/JWKS 来源限可信 issuer；不能信任 JWT 中任意 URL 或未校验 issuer 发起网络请求。
- listener 只绑定 loopback，绑定完成后再打开浏览器；请求有时限/大小限额，回调 state 一次性消费。
- 拒绝重复/冲突回调参数、错误路径和未关联的授权结果；校验失败不得覆盖已有账号。
- 不读取或复制 Codex 官方客户端的 auth.json；本应用使用自己的授权记录。

## 7. 可选 Codex app-server 路径

若后续客户端需要 coding agent，而非单次模型调用，单独实现 bridge。
遵循 [官方 app-server 接入步骤](https://developers.openai.com/siwc/token-sharing-open-source/codex-app-server)：

1. OAuth 由客户端完成；只把 access token 注入子进程的 `ACCESS_TOKEN` 环境，不放命令行或日志。
2. 配置 Responses provider：公开 API base、`env_key=ACCESS_TOKEN`、`wire_api=responses`、`requires_openai_auth=false`、`supports_websockets=false`。
3. stdio RPC：initialize → initialized → thread/start → turn/start；clientInfo 使用本应用身份。
4. 显示 agent deltas；检查 turn/completed 的 status，不把 failed/interrupted 当成功。
5. token 更新后重启此配置的子进程、重新 initialize，再用保存的 thread ID resume。

app-server 的 model/list 可能是 bundled/cached catalog，不是用户 entitlement 校验。
线程/工具执行、sandbox、批准、工作目录与进程退出由 bridge/客户端管理，不作为普通 ModelAdapter 的默认能力。
只有短期 access token 进入子进程，refresh token 留在认证服务；子进程继承环境按需筛选。
重启、取消与工具执行状态必须明确协调，不能因刷新重启就自动重做已经有副作用的工具。

## 8. 分阶段落地与最小验收

### 阶段一：客户端认证 + 纯文本套餐推理

- 先实现用户显式登录、身份校验、权限状态、一个账号的刷新与登出。
- 补齐 SDK OAuth-only 入口和套餐 mode；保留既有 API-key 路径。
- 固定官方 origin，纯文本 SSE，只有 completed 成功；不做工具/多账号 UI/app-server。
- 示例不含真实凭据，不调用未公开 backend，不默认消耗套餐/API 费用。

### 阶段二：稳健会话与请求能力

- 多账号隔离、跨进程刷新协调、scope 变化、撤销恢复、结构化错误。
- 有明确需求后再支持 tools namespace / reasoning / image 等经过核对的能力。
- 不一次性将 OAuth 推进所有 adapters；其它 provider 的登录协议独立设计。

### 阶段三：可选 companion 与 Codex bridge

- 多客户端复用时提取认证 companion crate；不增加核心 SDK 的浏览器/OS keychain 默认依赖。
- 独立 app-server bridge，清晰标记执行副作用；用户不需要 agent runtime 时不启动子进程。

### 必要检查，不反复跑全套测试

docs-only 不需要运行 Rust 测试。后续认证实现应一次集中运行有意义的 mock/本地回归，至少覆盖：

| 场景 | 必须结果 |
| --- | --- |
| 错误 state/nonce/audience/issuer；回调重放 | 不交换/不接受无效身份，不污染现有账号 |
| 身份有效但未授予套餐 scope | 无推理请求、无隐式 API-key 回退 |
| 多请求同时刷新；刷新与登出并发 | 一次有效轮换；旧任务不能恢复已登出会话 |
| 账号切换同时发生重试/重连 | 旧请求身份保持固定或被明确取消 |
| 自定义 base_url / redirect / 恶意 JWKS URL | 不泄漏 OAuth token、不向不可信地址取认证材料 |
| 套餐不支持的参数通过 extra_body/profile 带入 | 本地明确拒绝，普通 API-key 模式保持原行为 |
| delta 后 failed/incomplete/EOF；限额错误 | 不误报成功、不重复副作用、不静默改计费 |
| 保存中断、秘密对象 Debug 和错误路径 | 无半更新凭据、无 secrets 泄漏 |

真实 OAuth/推理仅在开发者显式操作的 smoke check 中验证，不把登录/扣费接入普通 CI。
通过必要检查后再决定是否做一次全套兼容验证；不要为了“频率少”省掉认证边界检查。

## 9. 可直接交给后续模型的任务说明

> 阅读本文件、AGENTS.md 与相关源码，重新核对本文链接的当前官方文档。
> 以 0.4.1 为兼容基线，先实现阶段一：客户端负责登录/存储/刷新，SDK 提供明确 OAuth-only 认证接入和保守的 ChatGPT-plan Responses 配置。
> 不把提案中的接口名当成已存在，不用 placeholder API key，不复用 Codex 客户端凭据，不修改普通 API-key 默认行为。
> 先列出实际 public API 变化与会话/刷新/请求身份模型，再实现；发现官方流程变化时更新本文并说明差异。
> 集中完成必要安全回归，不反复跑完整测试。完成后报告剩余能力限制和是否破坏兼容；不要未经用户授权发布 crates.io、打 tag 或改变已发布版本。

## 10. 官方资料索引

以下是核对依据；项目建议不等同于 OpenAI 对本项目的认证或背书。

- [SIWC Quickstart](https://developers.openai.com/siwc/quickstart)：身份与套餐权限、接入类别。
- [OSS Overview](https://developers.openai.com/siwc/token-sharing-open-source)：开放范围、client 与 host。
- [Registration and sign-in](https://developers.openai.com/siwc/token-sharing-open-source/sign-in)：注册、回调、交换与验证。
- [Accounts and sessions](https://developers.openai.com/siwc/token-sharing-open-source/profiles-and-sessions)：账号、刷新、登出与秘密管理。
- [Token reference](https://developers.openai.com/siwc/token-sharing-open-source/token-reference)：响应字段、expiry 与 opaque metadata。
- [Models and inference](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference)：模型目录、公开 Responses、终态。
- [Errors and recovery](https://developers.openai.com/siwc/token-sharing-open-source/errors-and-recovery)：scope、限额、路由与刷新错误。
- [Preview limitations](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations)：当前请求/工具能力限制。
- [Codex app-server](https://developers.openai.com/siwc/token-sharing-open-source/codex-app-server)：子进程、RPC 与 token 更新。
