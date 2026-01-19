下面我把 **P1.2（PhaseExecutor 的 budget flush 阈值策略）**写成一个“工程可落地”的自适应方案：**按 phase 的历史平均 `opened_words/flush`（以及 `open_wire_bytes/flush`）动态调 `LazyLimits`**，并且加入**硬护栏**来避免“阈值过小 → budget flush 反而变多 → 总 flush 次数上升”的回退。

我会先说明你现在代码里阈值是怎么用的，然后给出一个 **phase-aware 自适应算法**（带推荐默认参数），最后给出 **改动 checklist（到函数/变量级别）**。

---

## 0) 你现在的“阈值入口”在哪里（为什么需要 P1.2）

你当前结构里，budget flush 的阈值来自：

* `runtime::PhaseExecutor::LazyLimits`

    * `open_pending_words`
    * `coeff_pending_jobs`
    * `trunc_pending_jobs`
    * `hatx_pending_words`

并且在 **PhaseExecutor::run / run_lazy** 里，每次 loop 都会判断：

* `budget_open`: `opens_.pending_words() >= open_pending_words`
* `budget_coeff`: `pfss_coeff_.pending_jobs() >= coeff_pending_jobs`
* `budget_hatx`: `pfss_coeff_.pending_hatx_words() >= hatx_pending_words`（trunc 类似）

触发时会调用 `do_flush_open("budget") / do_flush_coeff("budget") ...`

而你现在在 `src/nn/transformer_layer.cpp` 里把：

* `lazy_lim.open_pending_words = open_lim.max_pending_words;`
* `lazy_lim.hatx_pending_words = pfss_lim.max_pending_hatx_words;`
* `lazy_lim.coeff_pending_jobs = pfss_lim.max_pending_jobs;`
* `lazy_lim.trunc_pending_jobs = pfss_lim.max_pending_jobs;`

这基本等价于 **“budget flush 永远不触发（除非 buffer 快满）”**，导致 flush 主要落到：

* demand flush（task 显式 Need::Open / Need::Pfss*）
* fallback flush（deadlock 分支兜底）

所以你看到的现象就会是：

* `open_flushes_budget ≈ 0`，几乎全 demand
* `pfss_flushes_budget ≈ 0`，大量 demand/fallback
* overlap 很难吃到（尤其你想做 LUT_7 “open in flight + pfss compute”）

---

## 1) P1.2 的目标：把阈值变成“按 phase 自适应”的、带护栏的 budget 策略

你要的工程化目标可以写成两条：

1. **不要因为阈值过小导致 flush 次数增加**

    * 特别是 Softmax 这种 phase：round 本来就多，阈值小一点就可能提前 budget flush，然后 demand 又来一次，导致 “budget + demand” 双计数。
2. **在“值得 overlap 的 phase”里，让 budget flush 尽可能替代（或提前覆盖）demand**

    * 例如 MLP / QKV_Score：通常有较大的 PFSS 工作量（hatx_words 大），有机会把 open 通信隐藏在 PFSS eval 下。

因此阈值应该**按 phase 的“历史平均 flush 粒度”自适应**，并且**在 round 数高的 phase 更保守**（阈值更接近平均 flush 大小）。

---

## 2) 一个可落地的自适应方案：用历史 `opened_words/flush` + `open_wire_bytes/flush` 推导阈值

### 2.1 我建议用 “wire bytes” 做主尺度，words 做护栏

原因很现实：

* gpt2 你提到 `n_bits=50`，open packing 压缩比变差，**同样 opened_words 对应的 wire bytes 更大**。
* 只按 words 会把 gpt2 的阈值估得偏小；按 bytes 更稳。

因此：每个 phase p 维护/计算下面这些“运行中可得、且与秘密无关”的统计：

从 `PhaseExecutor::Stats::by_phase[p]` 里你已经有：

* `open_wire_bytes_sent`
* `opened_words`（或 mask+beaver 合计）
* `open_flushes`
* `pfss_coeff_hatx_words`
* `pfss_coeff_flushes` / `pfss_coeff_jobs`
  -（trunc 类同；你现在很多地方 trunc_jobs=0，但保持通用）

然后我们取 **running average**（按该 phase 的累计值 / 累计 flush 数）：

* `avg_open_bytes_per_flush(p) = open_wire_bytes_sent / max(1, open_flushes)`
* `avg_open_words_per_flush(p) = opened_words / max(1, open_flushes)`
* `bytes_per_word(p) = open_wire_bytes_sent / max(1, opened_words)`

### 2.2 “避免阈值过小”的关键：让阈值跟着 round complexity 变保守

你担心的“阈值太小 → flush 次数反而增加”，本质在于：

* phase 的 **flushes per layer 很高**（Softmax 常见 58/layer），
* 这时 budget flush 稍微提前一点，就很可能出现：

    * 预算 flush 发出去时 pending 还没攒到“本轮需求的全部”
    * 后面又需要一次 demand flush（导致 flush 次数 +1）

所以我建议引入一个 **beta(p)**（0~1）系数，让阈值 = beta * 平均 flush 粒度，并让 beta 随 “flushes per invocation” 增大而逼近 1：

* 需要一个“phase 出现次数”计数 `invocations[p]`（每次 `begin_phase(p)` 记一次）
* 计算 `flushes_per_inv(p) = open_flushes / max(1, invocations[p])`

然后用一个非常工程、可控的函数：

[
\beta(p) = \beta_{\min} + (\beta_{\max}-\beta_{\min}) \cdot \frac{flushes_per_inv(p)}{flushes_per_inv(p) + K}
]

推荐默认：

* `beta_min = 0.70`（低 round phase：更激进一点，方便 overlap）
* `beta_max = 0.93`（高 round phase：非常保守，避免 flush 次数膨胀）
* `K = 16`（软阈值：flushes_per_inv=16 时 beta 取中间）

直觉效果：

* Softmax（~58 flush/layer）：beta ≈ 0.9+ → 阈值接近平均 flush 粒度，不容易多出一轮
* MLP（~10 flush/layer）：beta ≈ 0.75 左右 → 可以更早开工通信

### 2.3 open_pending_words 的具体计算（带 bytes 主尺度 + words 护栏）

给一个明确的工程公式（每次进入某 phase 时计算一次）：

**Step A：先决定是否启用 open 的 budget flush**

* 如果这个 phase 的平均单次 flush 太小（无法 amortize），直接禁用 budget：

    * `if avg_open_bytes_per_flush(p) < MIN_ENABLE_OPEN_BYTES: open_pending_words = 0`
* 如果这个 phase 基本没 PFSS 工作可 overlap（例如 Softmax 通常 hatx 很小），也禁用：

    * `if avg_pfss_hatx_words_per_inv(p) < MIN_ENABLE_HATX_WORDS: open_pending_words = 0`
* 如果 overlap 条件本身没开（你在 LUT_7 想要的 `R.overlap_pfss_open` / `R.open_flush_async`），也禁用：

    * `if !R.overlap_pfss_open || !R.open_flush_async: open_pending_words = 0`

> 这一步非常关键：**Softmax/LN 这类“round 多但可 overlap 的 compute 少”的 phase，budget flush 往往只会增加 flush 次数，不会减少 wall-time。**

**Step B：如果启用，计算阈值 words**

* 先给一个“目标消息大小”：

    * `TARGET_OPEN_BYTES = 1MB`（GPU 上比较合理；你也可以 2MB）
* 用 bytes_per_word 把它换算成 words：

    * `target_words_from_bytes = TARGET_OPEN_BYTES / max(ε, bytes_per_word(p))`
* 再用历史平均 flush 粒度乘 beta 做“防过小护栏”：

    * `guard_words_from_avg = beta(p) * avg_open_words_per_flush(p)`
* 取最大并 clamp：

    * `open_pending_words = clamp(round_up(max(target_words_from_bytes, guard_words_from_avg), 1024), MIN_OPEN_WORDS, opens_.limits().max_pending_words)`

推荐默认参数：

* `TARGET_OPEN_BYTES = 1<<20`（1MB）
* `MIN_ENABLE_OPEN_BYTES = 1<<18`（256KB；小于这个直接禁用 budget open）
* `MIN_ENABLE_HATX_WORDS = 1<<18`（262k words；没有足够 PFSS 工作就别 budget open）
* `MIN_OPEN_WORDS = 16384`（或 32768，都行；避免极小阈值）

> 这样做的效果是：
>
> * 在 MLP/QKV_Score 这类 phase，阈值会稳定在“每次 flush 至少 ~1MB wire”且“不会低于平均 flush 的 ~0.75~0.85 倍”。
> * 在 Softmax/LN 这类 phase，往往会被 Step A 直接禁用 budget（从根上避免“阈值过小导致 flush 变多”）。

### 2.4 PFSS 的阈值（hatx_pending_words / coeff_pending_jobs）同理做一版

PFSS 部分你最关心的是：

* 不要把 PFSS flush 切得过碎（kernel launch + staging overhead）
* 同时不要让 executor 频繁走 fallback（deadlock 兜底）

用同样思想：

* `avg_hatx_words_per_flush(p) = (pfss_coeff_hatx_words + pfss_trunc_hatx_words) / max(1, pfss_coeff_flushes + pfss_trunc_flushes)`
* `flushes_per_inv_pfss(p) = (pfss_coeff_flushes + pfss_trunc_flushes) / max(1, invocations[p])`
* `beta_pfss(p)` 用同样公式（可以同一套 beta，也可单独一套）

阈值建议：

* `hatx_pending_words = clamp(round_up(beta_pfss(p) * avg_hatx_words_per_flush(p), 1024), MIN_HATX_WORDS, pfss_coeff_.limits().max_pending_hatx_words)`

    * 推荐 `MIN_HATX_WORDS = 1<<18` 或 `1<<19`（看你 GPU kernel 的 amortize 点）
* `coeff_pending_jobs`：

    * 如果你观察到 `jobs/flush` 基本恒为 1（你现在常见 num_jobs=num_flushes），那 job 阈值调大也没用；建议让它至少 2（能 batch 就 batch），否则保持 1：
    * `coeff_pending_jobs = clamp(round(avg_jobs_per_flush(p)), 1, MAX_JOBS_BUDGET)`
    * 推荐 `MAX_JOBS_BUDGET = 16`（不要无限大）

同样可以加禁用条件：

* 如果该 phase `avg_hatx_words_per_flush` 本来就很小（<MIN_HATX_WORDS），那 PFSS budget flush 没意义，直接 0（禁用）让它走 demand/fallback。

---

## 3) 关键护栏：如何检测“阈值太小导致 flush 次数上升”，并自动回退

上面 beta 的设计已经能大幅降低回退风险，但你要“更工程化”，我建议再加一个非常直接的 **runtime 反馈回退**：

### 3.1 维护每个 phase 的 “flushes_per_invocation EMA”

在 `PhaseExecutor` 里维护：

* `ema_open_flushes_per_inv[p]`
* `ema_open_bytes_per_flush[p]`

如果发现：

* `ema_open_flushes_per_inv` 上升（比如比过去高 20%+）
* 同时 `ema_open_bytes_per_flush` 下降（flush 变小了）

那几乎可以断定：**阈值过小触发了额外 budget flush**。

回退动作：

* 直接把 `open_pending_words` 乘 2（或把 beta_min/beta_max 提高 0.05），并且对该 phase 开一个 “cooldown”：

    * 在接下来 N 次 invocations 内禁用 budget open（N=2~4 就够）
    * 等 stats 稳住再重新启用

这套机制能让你很放心地把 adaptive 打开，而不会在 Softmax 这类 phase 上一脚踩雷。

---

## 4) Checklist：需要改动的具体位置（函数/变量级别）

下面是我建议你按顺序落地的 checklist（尽量小改动、可回滚、可验证）。

---

### ✅ A. PhaseExecutor 里加“phase 自适应阈值计算 + 护栏”

文件：`include/runtime/phase_executor.hpp`

1. **新增 phase invocation 计数**

* [ ] 成员变量（PhaseExecutor private）：

    * `std::array<uint32_t, static_cast<size_t>(Phase::kCount)> phase_invocations_{};`
* [ ] 在 `void begin_phase(Phase phase)` 里：

    * `phase_invocations_[(size_t)phase]++;`

2. **新增 adaptive 配置与开关**

* [ ] 新增：

    * `struct AdaptiveLazyConfig { ... }`
    * 成员：`bool adaptive_lazy_ = false;`
    * 成员：`AdaptiveLazyConfig adaptive_cfg_;`
* [ ] 新增 setter：

    * `void set_adaptive_lazy_limits(bool on) { adaptive_lazy_ = on; }`
    * `void set_adaptive_lazy_config(const AdaptiveLazyConfig& c) { adaptive_cfg_ = c; }`

3. **实现“按 phase 计算 tuned LazyLimits”的 helper**

* [ ] 增加一个 private helper：

    * `LazyLimits tuned_limits_for_phase(Phase p, const PhaseResources& R) const;`
* [ ] 这个函数里用：

    * `stats_.by_phase[(size_t)p]` 的累计统计
    * `phase_invocations_[(size_t)p]`
    * `opens_.limits().max_pending_words`
    * `pfss_coeff_.limits().max_pending_hatx_words` / `max_pending_jobs`
* [ ] 按上面 2.3/2.4 的公式算：

    * `open_pending_words`
    * `hatx_pending_words`
    * `coeff_pending_jobs` / `trunc_pending_jobs`
* [ ] 根据 `R.overlap_pfss_open` / `R.open_flush_async` 做 enable/disable。

4. **在 run() / run_lazy() 里用 tuned limits 替代 lazy_limits_**

* [ ] 在 `void run(PhaseResources& R)` 开头（进入 while loop 之前）：

    * `const LazyLimits lim = adaptive_lazy_ ? tuned_limits_for_phase(current_phase_, R) : lazy_limits_;`
* [ ] 把所有 `lazy_limits_.xxx` 的 budget 判断替换为 `lim.xxx`：

    * `budget_open`
    * `budget_coeff`
    * `budget_trunc`
    * `budget_hatx`

5. **（可选但强烈建议）加“回退护栏”**

* [ ] 新增 per-phase EMA 记录（private）：

    * `ema_open_flushes_per_inv[p]`
    * `ema_open_bytes_per_flush[p]`
    * `cooldown[p]`（剩余禁用 budget 的 invocations 次数）
* [ ] 在 `tuned_limits_for_phase` 里，如果 `cooldown[p] > 0`，直接返回 `open_pending_words=0`。
* [ ] 在每个 phase 的末尾（run 返回前）更新 EMA，并触发 cooldown/阈值翻倍逻辑。

---

### ✅ B. TransformerLayer 里不要再把 lazy_lim 全设成 max（否则 P1.2 根本不会起效）

文件：`src/nn/transformer_layer.cpp`

你当前这段（你自己也指出过）：

```cpp
lazy_lim.open_pending_words = open_lim.max_pending_words;
lazy_lim.coeff_pending_jobs = pfss_lim.max_pending_jobs;
lazy_lim.trunc_pending_jobs = pfss_lim.max_pending_jobs;
lazy_lim.hatx_pending_words = pfss_lim.max_pending_hatx_words;
pe->set_lazy_limits(lazy_lim);
```

会把 budget 永远关掉。

改法（两种选一种）：

* [ ] **方案 1（推荐）**：lazy_lim 设一个“保守默认”，真正的阈值由 adaptive 覆盖

    * `lazy_lim.open_pending_words = 0;`（默认禁用 budget open；adaptive 再按 phase 决定开不开）
    * `lazy_lim.hatx_pending_words = 0;`（同理）
    * `lazy_lim.coeff_pending_jobs = 0;`
    * 然后 `pe->set_adaptive_lazy_limits(true)`。

* [ ] **方案 2**：保留 static（便于先验证），但至少别设成 max：

    * `open_pending_words = 1<<18`（~256k words）
    * `hatx_pending_words = 1<<18`
    * `coeff_pending_jobs = 8`
    * 然后再上 adaptive 做 phase 细化。

---

### ✅ C. Bench/配置层把 adaptive 开关与参数“钉死可复现”

文件（推荐但不是必需）：

* `src/demo/bench_suf_transformer.cpp`（结果 JSON 序列化处）

* [ ] 在 bench config 加开关：

    * `SUF_PHASE_EXEC_ADAPTIVE_LIMITS=1`

* [ ] 把 adaptive_cfg 的关键参数写进 results JSON（否则之后你会遇到“同一 commit 不同跑法结果漂移”的复现实验问题）：

    * `target_open_bytes`
    * `min_enable_open_bytes`
    * `beta_min/beta_max/K`
    * `min_open_words/min_hatx_words`
    * 是否启用 cooldown

---

## 5) 你可以怎么验证这套 P1.2 是否“真的没踩雷”

跑一个 base/gpt2（哪怕复用缓存 SIGMA）前后对比，重点看这些指标：

1. **flush 次数有没有因为 budget 变多而上升**

* `stats.by_phase.*.open_flushes` 是否明显增加
* `open_flushes_budget` 是否出现，但 `open_flushes_demand` 是否等比例减少（理想：budget 在替代 demand，而不是叠加）

2. **平均 flush 粒度有没有变小（危险信号）**

* `open_wire_bytes_sent / open_flushes` 是否明显下降

    * 如果下降且 flushes 上升，说明阈值太小 → 回退护栏应该触发

3. **overlap 有没有从 0 变成正数**

* `pfss_flushes_overlapped_in_open` 是否开始 >0

    * 这是 LUT_7 路线的核心收益来源之一

---
