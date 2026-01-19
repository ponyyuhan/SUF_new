下面我按你给的数字，把 **LN / Softmax 为啥这么多 flush**、以及 **下一步最划算的工程改动**（含具体代码点）写成一个可落地的行动清单。

---

## 1) LN1/LN2：flush 深度基本等价于 RsqrtTask 的迭代深度

你现在看到的：

* LN1：open_flushes=52，pfss_*_flushes=32
* LN2：open_flushes=52，pfss_*_flushes=32

（我按 bench 的聚合方式理解：通常是两方相加的总数；你自己也能用 bench_suf_transformer.cpp 的聚合逻辑对一下。）

LayerNormTask 的结构在 `include/runtime/phase_tasks.hpp` 里非常“公式化”，每一段会产生多少次 open/pfss flush 几乎是可数的：

### LayerNormTask 的固定交互部分（不含 rsqrt）

在 `LayerNormTask::step()` 里这些步骤会产生交互（按每方计数）：

* `mean_trunc_task_`：**1 次 open + 1 次 pfss(trunc)**
* `var_mul_`（diff*diff）：**1 次 open**
* `var_trunc_task_`：**1 次 open + 1 次 pfss(trunc)**
* `mul_norm_`（row-broadcast mul）：**1 次 open**
* `trunc_norm_`：**1 次 open + 1 次 pfss(trunc)**

所以 **不含 rsqrt** 的 LN 固定成本大概是：

* open flush：5 次/方
* pfss flush：3 次/方

### rsqrt 的交互轮数是线性放大的主因

`RsqrtTask` 在 `include/runtime/phase_tasks.hpp:1454` 起，典型模式是：

* `OpenXhat`：1 次 open
* `EnqueueInit`：1 次 pfss(coeff)
* `InitMul` + `InitTrunc`：1 次 open（mul） + 1 次 open + 1 次 pfss（trunc）
* 每一轮 Newton：`mul1 + trunc_y2 + mul2 + trunc_xy2 + mul3 + trunc_out`

    * open：**6 次/轮**
    * pfss(trunc)：**3 次/轮**

所以 **rsqrt 每多 1 次迭代**，会多出：

* open flush：+6 /方
* pfss flush：+3 /方

> 你现在 LN 的 open/pfss flush 数量，基本就能反推出 rsqrt 迭代数是不是偏高、以及 LN 的交互是不是被 rsqrt “吃满了”。

### 最划算的下一步：让 LN 不再用“多轮 rsqrt + 多次 trunc”堆精度

你有两个现实可落地的方向（一个偏“立刻减轮数”，一个偏“换协议”）：

#### 方向 A：**把 rsqrt_iters 变成可调，先把轮数压下来**

你现在材料生成在 `src/nn/transformer_layer.cpp` 里走的是：

* `make_layer_norm_material(..., rsqrt_iters, ...)`
* 你提到的工程里 rsqrt_iters 不一定是 1（你这份 flush 归因反而像更大）

建议你直接做两件事：

1. **把 rsqrt_iters 从常量改成 env/config 可控**（便于 A/B sweep）

    * 代码入口：`src/nn/transformer_layer.cpp` 里你现在设 `rsqrt_iters = ...` 的位置（你仓库里我看到一处是常量 1，但你实测 flush 很像 >1；无论如何把它改成可控总是对的）。
2. 用 bert-tiny 的 offline/online 对齐测试，扫 `rsqrt_iters = 1,2,3`

    * 看 online_time 和模型输出误差（通常 LN 的 rsqrt 精度对推理鲁棒性没那么脆，特别是 bert-tiny 这类小模型）。

这一步非常“值”：rsqrt_iters 每减 1，LN 这两段（LN1/LN2）就能直接少掉一大截 open/pfss 轮数。

#### 方向 B：**实现 sigma_gpu 风格的 rsqrt / invsqrt（大幅减少 trunc 轮数）**

你已经在 nExp / inv 上走了 sigma_gpu（`NExpSigmaGpuTask`、`InvSigmaGpuTask`），LN 反而还是“SUF 的 rsqrt gate + trunc 循环”，这在 flush 深度上天然吃亏。

如果你愿意做一个稍侵入但回报巨大的工程：

* 新增 `RsqrtSigmaGpuTask`（类似 `InvSigmaGpuTask` 的结构：open x_hat + LUT + 少量 beaver）
* 对 LayerNorm 的 var+eps 这种输入范围，做一个专用的近似：

    * LUT 近似 `1/sqrt(x)`（或者先归一化范围再 LUT）
    * 视精度再加 0~1 次 Newton 修正（**最好保证每轮只需要 1 次 trunc 或干脆不需要**）

这样 LN 的 pfss(trunc) flush 会从“每轮 3 次”直接掉到很少，open flush 同理。

> 你现在 LN 的 pfss_flushes（以及 pfss_flush_eval_eval_ns）很可能就是 rsqrt 的 trunc 链条在吃。把它换成 sigma 风格，是最直接的“把 LN 砍掉一半以上交互轮数”的路。

---

## 2) Softmax：open_flushes=124 这种量级，通常不是 PackCmp，而是“maxdiff + exp + inv/recip 的交互链条”

你给的 Softmax phase：

* open_flushes=124（最大）
* pfss_*_flushes=16（次大）

Softmax 这块在 SUF 里是串了两个任务：

1. `RowMaxDiffSigmaGpuTask`（attention_block.cpp 里你也贴了链路）
2. `SoftmaxBlockTask`（里面再跑 exp + inv + row-broadcast mul + prob trunc）

### RowMaxDiffSigmaGpuTask 本身的“最低交互轮数”是可数的

它每一轮 reduction（cols 每次减半）都需要：

* open diff_hat（1）
* B2A（mul：1 open）
* relu*diff（mul：1 open）

也就是**每轮 3 次 open flush（/方）**，轮数约 `ceil(log2(cols))`。

* cols=128 => 7 轮 => 21 次 open flush/方（两方合计 ~42）

这解释了 Softmax open_flushes 的一部分，但显然还不够到 124——剩下的通常来自 **exp / inv 的实现分支**：

* 如果 inv 还在走 `RecipTask`（Newton 迭代 + trunc），它会额外贡献很多 open/pfss。
* 如果 exp 还在走 SUF 的多次 truncate/多项式链，也会贡献很多轮。

你之前说“GeLU/nExp/inv 已经走 sigma_gpu”，那 Softmax 理论上应该已经避免了 RecipTask / CubicPoly 的高轮数链条；如果 Softmax 仍然这么多 open flush，我建议你重点检查两件事：

1. **Softmax 的 inv 分支到底是不是走了 `InvSigmaGpuTask`**

    * `SoftmaxBlockTask` 里明确是：

        * 有 `plan_.inv_sigma` 就走 `InvSigmaGpuTask`
        * 否则走 `RecipTask`
    * 关联 material 的生成在 `src/nn/attention_block.cpp` 里（你已经标出 inv_sigma_mat 的位置）
    * 关键门控：`gates::inv_sigma_gpu_enabled()`（取决于 `SUF_INV_MODE`，在 `include/gates/inv_sigma_gpu.hpp`）

2. **Softmax 的 exp 是否真的走了 `NExpSigmaGpuTask`**

    * 门控：`gates::nexp_sigma_gpu_enabled()`（`SUF_NEXP_MODE`，在 `include/gates/nexp_sigma_gpu.hpp`）
    * plan 里 `nexp_sigma` 是否被填上

> 如果 Softmax 还在走 `RecipTask` 或 SUF 的 exp 多轮链条，那么你现在的 124 open flush 非常合理；把它切回 sigma_gpu，softmax 这块的轮数会显著下降。

### Softmax 的进一步“减轮数”方向

假设你确实已经在 Softmax 走 sigma_gpu（inv/exp 都是 sigma），那 Softmax 依然高 flush 的来源就更可能是：

* `NExpSigmaGpuTask` 的最后一步 `TruncTask`（它为了回到目标 qf 会做 1 次 trunc）
* `prob_trunc_task_`（softmax 输出截断/量化）

这两个都是“**额外多一次 trunc = 额外一轮 open + pfss**”的典型来源。

你可以考虑的优化（按侵入程度从小到大）：

#### (1) 让 NExpSigma 的 LUT 输出直接对齐目标 qf，尽量去掉最后的 trunc

`NExpSigmaGpuTask` 里现在是：LUT -> mul -> trunc

如果你能在 material 生成时把 LUT 的输出标度（qf）设计成“乘完系数/缩放后正好落在目标 qf”，那最后那次 trunc 可以省掉（或至少变成便宜的 shift/round）。

这能直接减少：

* Softmax 的 pfss flush
* 以及 pfss_flush_eval_eval_ns

#### (2) Softmax 的 prob trunc：能不能前移/合并，或者用更少 bit-width 的 trunc

目前 prob trunc 的目的多半是把输出压回目标定点格式。你可以检查：

* prob 的范围其实是 [0,1]，很多时候不需要那么高精度
* 如果你允许 prob 用更粗的 qf（例如少一些 fractional bits），可以减少 trunc 的 bit-width 或者甚至用更便宜的 trunc 路线（例如更小的 in_bits/num_bits）

---

## 3) PhaseExecutor：pfss_flushes_fallback 很高，意味着你可能“丢了 overlap 的机会”

你最后那条结论非常关键：

> 各 phase 的 open_flushes_budget 基本为 0，几乎全是 demand；pfss_flushes 也几乎全是 fallback。

对应到 `include/runtime/phase_executor.hpp:367~441` 的逻辑就是：

* task 没显式返回 Need::Pfss（`want_pfss_* = false`）
* 但 PFSS batch 里实际上已经有 pending/flushed 的活
* executor 死锁后走 fallback，把 pending 的 pfss 强制冲掉

这会带来两个问题：

1. **错过你本来想用的 overlap_pfss_open 路径**
   你现在 overlap 分支的触发条件是（简化）：

   ```cpp
   if (R.overlap_pfss_open && want_open && (want_pfss_coeff || want_pfss_trunc)) { ... }
   ```

   如果 tasks 没把 want_pfss 置起来，即使 pfss_batch 里有 pending，也不会走 overlap 分支。

2. **pfss 工作启动得更晚，无法和 open comm 并行**
   你 profile 里 `pfss_flush_eval_eval_ns` 很大（~245ms）且 `open_flush_ns` 也不小（~90ms），但 wall-time 没怎么被 overlap 吃掉——这通常就是 PFSS flush 启动晚导致的。

### 立刻可做的一个小 patch：让 overlap 分支在“pfss 有活”时也触发

建议你把 overlap 分支从“看 want_pfss”改成“看 pfss batch 有没有 pending/flushed”：

* 位置：`include/runtime/phase_executor.hpp` deadlock 处理分支（你引用的那段，约 367 行附近）

* 思路：计算一个 `have_pfss_work`：

  ```cpp
  const bool have_pfss_work =
      (pfss_coeff_.has_pending() || pfss_coeff_.has_flushed()) ||
      (&pfss_trunc_ != &pfss_coeff_ &&
       (pfss_trunc_.has_pending() || pfss_trunc_.has_flushed()));
  ```

* 然后把条件改成：

  ```cpp
  if (R.overlap_pfss_open && want_open && (want_pfss_coeff || want_pfss_trunc || have_pfss_work)) {
      // pfss_thread 里 flush_one 不再只看 want_pfss_*，而是“哪个 batch 有活就 flush/materialize/clear”
  }
  ```

这样即便 tasks 没把 `Need::Pfss` 显式抛出来，只要 pfss_batch 里有 pending，executor 也会把 pfss flush 提前，并和 open flush 并行跑掉。

> 这类 patch 的价值：它不一定减少 flush 次数，但通常能把 `open_flush_ns` 和 `pfss_flush_eval_eval_ns` 的 wall-time 从“相加”拉到更接近“取 max”，对你现在这种“PFSS 245ms + open 90ms”非常敏感。

---

## 4) 你这轮 SoA/SPC3 修复后，下一步我会怎么排优先级

按“最可能直接把 0.37s 拉近 0.21s”的排序：

### P0：把 LN 和 Softmax 的交互轮数压下去

* LN：优先看 rsqrt 的迭代/截断链条（要么减 iter，要么换 sigma rsqrt）
* Softmax：确认 exp/inv 分支确实 sigma_gpu；然后想办法省掉末尾的 trunc（LUT 量纲对齐）

### P1：PhaseExecutor overlap 补洞（减少 pfss fallback，增加 overlap）

* 这是最少代码侵入、最可能立刻看到 online_time 下降的一刀
* 你已经定位到 `pfss_flushes_fallback` 很高，这种 patch 很对症

### P2：再考虑 kernel 侧的吞吐（例如你已做的 SPC3 SoA）

* 你已经证明 PackCmp 不是最大头，只是“避免回退地雷”
* 后续如果要再抠 kernel，我更倾向于抠 “trunc/rsqrt 的 trunc eval” 这类高频路径，而不是 packcmp 本身

---

如果你愿意，我可以基于你现在的“按 phase/task 拆来源”的数据，再进一步给一个**“把 flush 次数映射回具体 Task 类型/次数”的核对表**（例如：每个 RsqrtTask/RecipTask 理论上应贡献多少 open/pfss flush），这样你能更快判断某个 phase 的异常 flush 来自“分支没切到 sigma_gpu”还是“调度没有 overlap”。
