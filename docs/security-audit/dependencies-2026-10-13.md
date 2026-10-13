# 依赖安全与替代方案核查（2026-10-13）

## 实际扫描结果

对当前 0.4.0 工作区 Cargo.lock 使用官方 cargo-audit 0.22.2 执行扫描，而非只查最新版本的网络搜索：

```text
189 dependency entries
1307 RustSec advisories loaded
0 vulnerability matches
0 warnings (including yanked/unmaintained/unsound/notice)
ignore = []
target_os = []
target_arch = []
exit status = 0 with --deny warnings
```

扫描覆盖锁文件中的生产、构建、开发和多平台依赖，没有按当前 Windows 平台过滤。
数据来自新拉取的 [RustSec advisory-db](https://github.com/RustSec/advisory-db)，
commit `57dcc0f2192e025138907a495c1ac9a7e1e3e123`，提交时间 `2026-10-12T20:06:48+02:00`。
机器可读结果见 [dependencies-2026-10-13.json](dependencies-2026-10-13.json)。

工具从 RustSec 官方 GitHub release 下载并仅解压在忽略的 target/dependency-audit 目录，未全局安装。
本次只新增核查记录，没有修改 Cargo.toml、Cargo.lock 或源代码，也没有重新跑测试。
0 命中仅表示数据库已收录公告没有匹配当前版本，不等于不存在未知漏洞或未收录的原生 C 库问题。

## 值得关注的库

| 依赖 | 当前锁定版本 | 核查结论 | 建议 |
| --- | --- | --- | --- |
| yoke-derive | 0.8.4 | 旧的 0.8.3 已撤回，上一轮已更新；当前未撤回，无 RustSec 命中 | 保持 0.8.4，不需要替换 ICU/URL 链 |
| rustls | 0.23.45 | 已包含近期 TLS 1.3 握手加密层边界修复 | 不要降回 0.23.13–0.23.44；关注下游实际解析版本 |
| time | 0.3.55 | 已避开已知 RFC2822 栈耗尽漏洞；本项目只调用 RFC3339 解析 | 可以把直接要求从 0.3 提高为至少 0.3.47，无需换库 |
| aws-lc-rs / aws-lc-sys | 1.18.1 / 0.45.0 | 未撤回、无 RustSec 命中；上游已有 1.18.2 / 0.46.0，含安全加固与原生构建修复 | 本次最值得考虑的定点升级，不是更换 TLS/crypto 后端 |
| reqwest | 0.13.5 | 当前稳定版，未撤回、无 RustSec 命中；安全策略仍需 SDK 管理 | 保留；不要将上一轮 SDK 的重定向/日志缺陷归为它的 CVE |
| sse-stream | 0.3.0 | 未撤回、无 RustSec 命中，仅 dev-dependency 用于对照/压测 | 不进入 SDK 正常运行时，没有必要为发版替换 |
| async-trait | 0.1.92 | 无命中；项目需要 dyn ModelAdapter/HeaderInjector | 保留，不宜直接换成原生 async trait |
| bytes / futures-core / httpdate / serde / serde_json / thiserror / tokio | 1.12.1 / 0.3.34 / 1.0.3 / 1.0.229 / 1.0.151 / 2.0.21 / 1.53.1 | 扫描无匹配漏洞或维护/撤回告警 | 没有证据支持为“更安全”而整体替换 |

### yoke-derive：撤回不等于确认安全漏洞

crates.io 元数据确认 `0.8.3` yanked，`0.8.4` 未撤回；需要区分 `yoke-derive 0.8.3`
与当前仍使用的 `yoke 0.8.3`，它们不是同一个包。本次没有找到足以确认撤回原因的官方安全公告，
因此不将该撤回直接标为 CVE。ICU4X 另外曾修复 yoke 的 Send/Sync soundness，当前 yoke 已是该修复版本。

来源：[crates.io 版本元数据](https://crates.io/api/v1/crates/yoke-derive/0.8.3)、
[ICU4X changelog](https://github.com/unicode-org/icu4x/blob/main/CHANGELOG.md)。

### rustls：当前版本正好达到修复门槛

[RUSTSEC-2026-0285](https://rustsec.org/advisories/RUSTSEC-2026-0285.html)
影响 0.23.13–0.23.44，0.23.45 修复。问题是错误加密层的 TLS 1.3 握手消息未被拒绝，
不是已证明可让网络攻击者伪造或完成被认证的握手。
当前链为 reqwest -> hyper-rustls / tokio-rustls -> rustls 0.23.45，未命中。

但 reqwest 0.13.5 的 manifest 对 rustls 的要求仍是 `0.23.4`（caret），因此“升级 reqwest”
并不自动等价于“最低 rustls 版本已经被约束到 0.23.45”。下游旧锁文件仍值得单独审计。

### time：当前安全，且本项目没有使用公告中的格式

[RUSTSEC-2026-0009](https://rustsec.org/advisories/RUSTSEC-2026-0009.html)
影响 `>=0.3.6, <0.3.47` 的 RFC2822 解析，可被恶意输入耗尽栈；0.3.47 加入递归限制。
本项目在 src/retry.rs 中使用 Rfc3339，因此不能据此断言旧 time 会被当前 SDK 的这条路径利用。
不过把直接依赖下限收紧到 0.3.47 仍是低成本防御，避免许可范围接纳已知有问题的旧版本。

### AWS-LC：有比扫描结果更值得关注的上游更新

[aws-lc-rs 1.18.2 发布说明](https://github.com/aws/aws-lc-rs/releases/tag/v1.18.2)
将 aws-lc-sys 更新到 0.46.0、底层 AWS-LC 更新到 5.11.0，包含：

- C 实现的 Ed25519 解码拒绝非规范公钥；
- ChaCha20-Poly1305 密钥材料清零；
- ML-KEM CPU 特性判断修复，避免特定 x86-64 CPU 的非法指令失败；
- jitterentropy 原生构建优化参数处理等修复。

这些使定点升级有价值，但不能据此编造当前版本的 RustSec/CVE 命中。
若决定升级，应通过 aws-lc-rs 的兼容补丁版本带动 sys 更新，而不是直接对 0.x sys 依赖盲目加约束。
原生 crypto 构建可能需要重编译，Windows/Linux/macOS 构建应保留基本验证。

Cargo.lock 还包含 ring 0.17.14，但当前 Windows 正常依赖树未启用它。
不要用过时的“ring 全部失维护”公告判断：RUSTSEC-2025-0007 已撤回；
旧分支的维护与漏洞公告要分别按版本核对，当前锁定版本也没有扫描命中。

## 是否有更好的替代

### reqwest -> hyper / ureq

- hyper 更底层；需要自己承担请求、TLS 集成、重定向、连接池和错误策略，
  并不是删除 reqwest 就删除这些风险。当前 reqwest 本来就使用 hyper。
- [ureq](https://github.com/algesten/ureq) 面向阻塞 I/O，适合纯同步程序；
  本项目核心是 async + SSE，替换会改变调度与流式模型，不是快速安全升级。
- 更实际的减依赖方案是审视 reqwest 的 gzip/http2 feature 是否确实需要。
  关闭它们可减少对应解压/HTTP2 链，但会牺牲带宽效率/多路复用，且 Cargo feature
  可被其他依赖统一重新启用。没有证据表明当前这两个 feature 对应版本有已知漏洞。

### time -> jiff

[jiff](https://github.com/BurntSushi/jiff) 是可以评估的日期时间替代，适用于更完整的时区/日历需求。
本项目只解析 provider reset 的 RFC3339，换库没有明显功能收益，不能因为另一个库更新就声称更安全。
优先收紧 time 最低版本，而非改写日期解析。

### async-trait -> 原生 async trait

项目的 Arc<dyn ModelAdapter> 与 Arc<dyn HeaderInjector> 需要动态分派。
[async-trait 官方文档](https://docs.rs/async-trait/latest/async_trait/) 说明它提供 dyn 支持，
原生 async trait 不是这类用法的直接替代。手写 boxed future 可以移除宏，但增加接口维护成本，
当前没有安全命中支持这个替换。

## 发布库特有的注意事项与优先级

[Cargo 官方 FAQ](https://doc.rust-lang.org/cargo/faq.html#why-have-cargolock-in-version-control)
明确指出：库自身的 Cargo.lock 不约束消费者，Cargo.toml 才影响下游解析。
因此上一轮对 yoke-derive 的 lockfile 修复保障本仓库构建，不等于替所有下游应用更新了依赖。

建议按收益排序：

1. 评估 aws-lc-rs 1.18.2 / aws-lc-sys 0.46.0 的定点补丁升级。
2. 对 time 等已有明确公告的直接依赖设置合理的安全版本下限；TLS 链仍要求下游审计实际 lockfile。
3. CI 加依赖审计，区别漏洞与 yanked/维护告警；这种检查不执行测试，不必增加测试频率。
4. 只有明确不需要 gzip/http2 时才删 feature；不要为了替代而换掉成熟的 HTTP/TLS/序列化栈。

本次没有执行升级或添加 CI，以保持已准备的 0.4.0 发布包与锁文件一致。
