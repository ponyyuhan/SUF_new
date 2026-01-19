你现在的状态我理解成一句话：**离 SIGMA 的差距已经不是“某个 kernel 慢 10%”能补的，而是 Softmax/LN 里还残留着“必须 open/flush 的结构性轮次”，尤其是 Softmax 里那条“概率乘积后再 Trunc(Q2f→Qf)”**——它会把一个超大的矩阵（rows×cols）丢进 TruncTask，从而拉高 `open_flushes` 和 `pfss_flush_eval_eval_ns`，而且因为依赖链很串，`pfss.num_flushes≈pfss.num_jobs` 也很难靠 batching 再救。


---

## LUT_8：把 Softmax 的 **prob trunc(Q2f→Qf)** 也“像 nExp 一样消掉”

### 你现在 Softmax 里仍然有一个大 Trunc

在你给的代码里，Softmax 末尾仍然是：

* `SoftmaxBlockTask::MulRun`：`MulRowBroadcastTask(exp, inv) -> prod`（当前语义上是 Q2f）
* `SoftmaxBlockTask::TruncRun`：`TruncTask(prod Q2f -> Qf)`（这就是大头）

对应文件你本地就是：

* `include/nn/softmax_block_task.hpp` 的 `Stage::TruncRun`（你现在这份 zip 里还在）
* `src/nn/attention_block.cpp` 里 `prob_choice.shift_bits = fb;` 并且生成了 `prob_trunc` bundle（也是你日志里 Softmax open_flush 很高的根源之一）

你已经把 **nExp 内部那次“LUT×LUT 后 Trunc(Q2f→Qf)”**干掉了；下一步就是把 **softmax 的“exp×inv 后 Trunc(Q2f→Qf)”**也干掉。

---

## 核心想法：把 Qf 的缩放“拆到 exp 和 inv 上”，让乘积天然落在 Qf

目标是让：

* `exp` 输出不是 Qf，而是 **Qfe**
* `inv` 输出不是 Qf，而是 **Qfi**
* 让 `fe + fi = f`，于是 `exp(Qfe) * inv(Qfi)` 的乘积就是 **Qf**，**不需要 trunc**。

这完全沿用你 LUT_7 的“分摊缩放”理念，协议不变，只是**改变近似/缩放边界**。

### 推荐从一个“稳妥参数”开始

如果你担心精度，建议第一版用：

* `fe = 8`
* `fi = f - 8`（如果 f=16，则 inv 输出 Q8）

这样 exp 还保留 8bit 小数，精度不会太离谱；inv 也有 8bit 小数；最终概率仍是 Q16。

> 如果你后面想再更激进，可以试 `fe=6`（会让 inv 的“Trunc→Q6”更自然/甚至可去掉），但第一步我建议先 `fe=8`。

---

## 需要改哪里（按你代码结构给出落点）

下面按“最少侵入”的实现路径列改动点。

### 1）扩展 SoftmaxPlan：让 softmax 知道 exp/ inv 的 frac_bits

文件：`include/nn/softmax_block_task.hpp`

当前 `SoftmaxPlan` 只有一个 `frac_bits`（表示最终 Qf），建议加两个字段：

```cpp
struct SoftmaxPlan {
  int rows = 0;
  int cols = 0;
  int frac_bits = 0;        // 最终输出 prob 的 frac bits（仍是 f）
  int exp_frac_bits = 0;    // 新增：exp 输出 frac bits（fe）
  int inv_frac_bits = 0;    // 新增：inv 输出 frac bits（fi）
  bool skip_prob_trunc = false; // 新增：乘积已经是 Qf，则跳过 TruncRun
  ...
};
```

### 2）attention_block.cpp：构造 softmax plan 时设置 exp/inv frac bits，并**不再生成 prob_trunc bundle**

文件：`src/nn/attention_block.cpp`，你现在生成 `prob_choice` 的那块（大概在你提到的 1345 附近往后那段）

目前逻辑是：

* `plan.frac_bits = fb;`
* `prob_choice.shift_bits = fb;`
* `cache_trunc` 生成 `prob_trunc`

改成：

* `plan.frac_bits = fb;`
* `plan.exp_frac_bits = 8;`
* `plan.inv_frac_bits = fb - plan.exp_frac_bits;`
* `plan.skip_prob_trunc = true;`
* **删掉（或条件绕过）prob_choice/prob_trunc 这整段**

也就是：

```cpp
if (!plan.skip_prob_trunc) {
   // 原 prob_choice + cache_trunc 逻辑
} else {
   // 不生成 prob_trunc（也就没有那次大 Trunc 的 keys / pfss jobs / open）
}
```

> 这个点很关键：**你不仅省 online，还能省 offline key_bytes/keygen**（prob_trunc 的矩阵很大，省掉的 key 很可观）。

### 3）NExpSigmaGpuTaskMaterial：给 softmax 用一份“输出 Qfe 的 nExp material”

文件：`src/nn/attention_block.cpp` 里 `cache_nexp_sigma` 的那段（你现在 key 是 `{fb, batch_N}`）

现在你是按 `fb` 生成 nExp sigma material。你要改成按 `plan.exp_frac_bits` 生成：

* cache key 改成 `{exp_frac_bits, batch_N}`
* `dealer_make_nexp_sigma_gpu_task_material(ctx->trunc_backend(), exp_frac_bits, rng, batch_N)`

这样 Softmax 的 exp 输出就从 Qf 变成 Qfe。

> 你已经在 `include/gates/nexp_sigma_gpu.hpp` 里把 LUT0/LUT1 的分摊缩放做掉了，所以这里只是把“最终想要的 frac_bits”从 `f` 换成 `fe`。

### 4）InvSigmaGpu：让 inv 支持“输入 frac_bits != 输出 frac_bits”

你现在的 inv-sigma material把 `frac_bits` 同时当输入/输出的 frac bits 使用（table scale 是 `2^(frac_bits+6)`，并且 trunc 用 `frac_bits-6`）。

要实现 `sum(exp)` 的输入是 Qfe、输出是 Qfi，你需要把 inv 的 material 拆成两个参数：

* `in_frac_bits = fe`（sum 的 frac bits）
* `out_frac_bits = fi`（inv 的 frac bits）

#### 4.1 改 material 结构

文件：`include/gates/inv_sigma_gpu.hpp`

把：

```cpp
struct InvSigmaGpuTaskMaterial {
  ...
  int frac_bits;
  ...
  std::array<uint64_t, 4096> table;
};
```

改成：

```cpp
struct InvSigmaGpuTaskMaterial {
  ...
  int in_frac_bits;
  int out_frac_bits;
  ...
  std::array<uint64_t, 4096> table;
};
```

#### 4.2 改 table 生成公式

当前是：

* `scale = 2^(frac_bits + 6)`
* `table[i] = floor(scale / i)`

改为：

* `scale = 2^(out_frac_bits + 6)`
* `table[i] = floor(scale / i)`

因为 `i` 仍然是 Q6 的整数（代表 `real_z * 2^6`），你想要输出 Qout，则 `real(1/z) * 2^out = 2^(out+6)/i`。

#### 4.3 trunc_to_q6 的 shift 改成用 in_frac_bits

当前 trunc_to_q6 用 `frac_bits - 6`，改成 `in_frac_bits - 6`。

如果 `in_frac_bits == 6`，你甚至可以**完全跳过这个 trunc**（见下面“顺手收益”）。

### 5）SoftmaxBlockTask：Mul 后直接 finish，不进 TruncRun

文件：`include/nn/softmax_block_task.hpp`

现在逻辑是：

* `MulRun` 结束后进入 `TruncRun`
* `TruncRun` 才把 `out_qf_` 赋值

改成：

* 如果 `plan_.skip_prob_trunc`：

    * `MulRun` 完成后：`out_qf_ = mul_task_->out()`，直接 `st_=Done`（或进入 Done）
* 否则走原逻辑。

---

## 这一步你大概率会看到的指标变化

在你现在的 `bert-tiny GPU L128 B1` 场景下，我预期最直观的变化是：

1. **pfss/by_phase/Softmax 的 trunc/coeff jobs 会明显掉一大块**
   因为你删除了一个“对 rows×cols 大矩阵做 Trunc”的任务。

2. **pfss_flush_eval_eval_ns 会显著下降**
   这个大 Trunc 的 PFSS eval 是典型的“GPU eval kernel + staging + sync”重成本。

3. **Softmax 的 open_flushes 也会下降**
   TruncTask 自身会触发 open（mask open / beaver open / 或它内部的 open 组合），去掉它能减轮次。

4. **offline：key_bytes/party & keygen_time 还会更好**
   你删掉了一个超大矩阵的 trunc keys；即使你为了 `fe=8` 让 nExp 的 DReLU bits 增加了一点，整体也很可能仍是净下降。

---

## 选 fe=8 会让 nExp DReLU key 变大多少（给你一个心里预估）

你现在的 nExp DReLU key bytes per elem 用的是 `proto::dpf_point_key_bytes_for(in_bits)`，而
`in_bits = 66 - frac_bits`（因为代码是 `n - frac_bits + 2`，n=64）。

DPF point key bytes = `16 + 17*in_bits`，所以每减少 1bit frac，会增加 **17 bytes/elem**。

* 从 `f=16` 改到 `fe=8`：Δ=8 → **+136 bytes/elem**
* softmax 里 exp 元素数大概是 `rows*cols`（例如 1536×128=196,608）
  增量大概 `196,608 * 136 ≈ 26.7MB` / party

但你同时**删掉了 prob_trunc 的 keys**（这通常也在几十 MB 量级甚至更大，取决于 trunc lowering 选了什么）。
所以整体 key_bytes 可能不升反降。

---

## 一个“顺手的 P0 小补丁”：TruncTask shift_bits==0 直接 no-op

不管你做不做 LUT_8，这个补丁都值得加（能避免未来因为某些 plan 把 shift 算成 0 却还跑一套开销）。

文件：`include/runtime/phase_tasks.hpp` 的 `TruncTask::step()`

在最开头加：

```cpp
if (bundle_->suf.shift_bits == 0) {
  // out = in (按 span copy / 或直接 alias，如果语义允许)
  std::copy(in_.begin(), in_.end(), out_.begin());
  st_ = Stage::Done;
  return Need::None;
}
```

这在你后续尝试 `fe=6`（让 inv 的 trunc_to_q6 变成 shift=0）时尤其有用：能彻底消掉那次 inv trunc（虽然它本身很小，但轮次也能少一次）。

---

## 做完 LUT_8 之后，如果还差 SIGMA ~0.02–0.03s，下一步怎么选

我建议按“收益/风险”排序：

### A. 先做“证据化”：把 pfss_flush_eval 细分到 job type / 任务名

你现在只看到 `pfss_flush_eval_eval_ns` 总和。下一步应该在 `PfssSuperBatch::flush_eval` 或 PhaseExecutor 调用点加一个轻量 breakdown：

* 每个 job group 记录：类型（trunc / packcmp / interval LUT / …）、元素数、in_bits、kernel 路径（SoA/scalar）、耗时
* 输出到 `online_profile` 里（按 phase 聚合）

这样你能非常快确认：
**剩下的 150ms/200ms pfss_eval 到底是 LN 的 trunc 在烧，还是别的 gate。**

### B. 如果 LN trunc 仍是 top1：考虑“降低 norm_mul 后的截断位数”

LayerNorm 的大 trunc 往往来自 `diff(Qf) * inv_std(Qf) -> Q2f，再 Trunc f bits`。
一个可控方向是把 `inv_std` 的输出 frac_bits 降到 `s`（比如 8），让乘积变成 `Q(f+s)`，只需要 trunc `s` bits（比 trunc f bits 便宜很多）。

这属于你自己在“协议/数值边界”那一类更重的改动，但比改 maxdiff 轮数风险小，且非常贴近 SIGMA 的做法（它们通常也不会把所有中间量都维持在同一个高精度 Qf）。

---

## 总结一句：你现在最该打的点

**把 Softmax 末尾的 prob trunc 消掉**（就像你已经消掉 nExp trunc 一样），这是你现在描述的“量级问题”（open flush 次数 + pfss flush eval）里最有确定性、改动最贴 LUT_7、也最可能直接拉平 SIGMA 的一步。

如果你愿意，我也可以按你现在的代码风格，把上面这些改动点再进一步“落到函数/结构体具体签名怎么改、哪些 cache key 要更新、哪些 cost 估计要同步改”，避免你踩到“material cache key 不一致导致 key bytes 统计错/重复生成”的坑。
