## 1) 真实 batching 需要改很多代码吗？

先把概念钉死：

* 你现在的 *internal batch*（binary 里 loop `batch` 次 forward）本质是 **顺序多次推理**，不会改变算子张量形状，也基本不会改变每次推理的协议结构/rounds，只是重复执行。
* 真正的 *true batching* 是指：一次 forward 里张量显式带 batch 维度，例如输入 token shape 从 `[L]` 变成 `[B, L]`，中间激活从 `[L, H]` 变成 `[B, L, H]`（或 flatten 成 `[B·L, H]`），从而**同一轮通信里处理更多元素**、更少的 kernel launch/调度开销，可能改变端到端 amortization。

你问“要改很多代码吗”，答案是：**对你们这种系统，改动量通常是“中等”，且集中在少数几个地方；标量非线性（FuseFSS 的强项）反而改动最小，主要工作在 attention/softmax 这一类 vector blocks。**

为什么我这么判断（结合你论文的架构）：

### 1.1 非线性/标量门（FuseFSS 编译出的 gates）——通常几乎不用大改

FuseFSS 的 gate evaluation 本来就是对一个张量的所有元素做“批处理”：打开公共 masked 值、做一次 packed comparisons + 一次 interval lookup，然后对整块张量跑 share-based 后处理（Protocol 1/2、Section 5.3 的 batched opening 与 compiled gate module 结构）。
对这些算子而言，true batch 只是把元素数 `N` 变成 `B·N`（flatten 即可），**kernel 逻辑不变**，只需要让上层把更大的张量传进来。

> 换句话说：FuseFSS 的“非线性部分”天然适配 true batching。

### 1.2 线性层（GEMM/FFN）——多数情况下改动也不大

Transformer 里很多线性层可以把 `[B, L, H]` flatten 成 `[B·L, H]`，然后仍然用同样的 GEMM。
如果你们 backend 是用 cuBLAS/CUTLASS 或自己封装了“任意 M×K × K×N”矩阵乘，那么 batch 只需要改上游 shape 与 leading dimension 的计算，改动往往比较局部。

### 1.3 真正麻烦的点：Attention 的 `QK^T` / Softmax / `Softmax·V`

你论文里也明确把像 softmax / layernorm 的 “max/sum reduction” 这类 **vector-level blocks** 当作 operator-spec IR 之外、由标准 MPC 子协议拼出来的部分（Appendix G 的 scope 说明）。
因此 true batching 主要要改的就是这些地方：

* `QK^T` 会变成 **(B·heads) 个 L×L 矩阵** 的 batched GEMM（或 strided batched GEMM）
* softmax 的 reduction 需要在 **每个 batch、每个 head、每个 query position** 上做（行数从 `heads·L` 变成 `B·heads·L`）
* `Softmax·V` 同样是 batched GEMM

如果 Sigma 当前路径是“默认 B=1，batch=loop”，那么 true batching 需要你把 attention 相关 kernel/调度改成显式支持 batched shape（否则没法称为 true batching）。

### 1.4 Preprocessing / key-stream / keyBuf：改动不一定大，但必须非常小心

你论文里强调每个 wire 必须独立 mask，不能跨 wires 重用，否则会泄露关系（Section 3 的 wire-level masking 语义）。
True batching 会把 wires 数量放大到 `B×`，因此：

* offline：mask/keys/triples 都要生成 `B×` 份（总量线性增长）
* online：必须以完全一致的顺序消费 key stream（你们 1/27 internal batch 的 crash 就是这类问题的典型）

好消息是：这类改动往往是“确保指针推进/索引一致”，不是重写协议；坏消息是：**一旦顺序或尺寸错一点就会 crash 或 silent wrong**。

### 1.5 最现实的结论

* **不需要“重写 FuseFSS 编译器”**；你们最核心的标量编译结构不动。
* 主要需要在 **attention/softmax/layernorm 的 shape 与内核**、以及 **key consumption 的一致性** 上做工程。
* 改动量一般不是几十行，而更像：**attention 子图 + 软最大化 + 若干张量布局/dispatcher 的一小段重构**。
  是否算“很多”，取决于你们现有实现是不是已经“shape 泛化”。如果现在所有 tensor 都隐含 B=1，那就是中等偏多；如果你们内核本来就支持任意 `numRows`，那就偏少。

---

## 2) 设计一套完整的“真正 batching”实验方案

下面这套方案目标是：**用最少额外解释，一次性把 reviewer 关于 batch 的质疑关掉**，并且和你们系统的特点（非线性编译）强相关。

我把方案分成：实现路径、实验矩阵、指标、统计方法、呈现方式、以及（可选）更强的扩展。

---

### 2.1 明确实验目标与论文要回答的问题

你最终要在 paper/rebuttal 里回答的不是“batch>1 能不能跑”，而是：

1. **当 batching 改变 amortization 时，FuseFSS 的 end-to-end 优势是否仍在？**
2. speedup 随 batch 增大是 **保持/变大/变小**？为什么？
3. 通信与 key material 的“每条推理成本”是否随 batch 改变？（理论上不该变，除非你引入跨 batch 复用或改了协议。）
4. 如果优势变小：是不是因为 linear/attention 主导（你论文里已有这个叙事，但 reviewer 要定量）。

---

### 2.2 True batching 的实现约束（为保证公平与可解释性）

为了让实验对 Sigma/SUF 都公平、且 reviewer 不会说 “你们 batch 是假的/不可比”，我建议你固定下面约束：

**(A) batch 是公开参数**
在你们泄露模型里，shape（比如比较数量、lookup 维度）是显式 leakage（Appendix I 的 Lshape 讨论），batch 作为 shape 也公开是合理的。

**(B) 不做跨样本复用 mask/keys/triples**
每个样本的每个 wire 都要 fresh mask（你们论文也明确要求）。
所以 true batching 的意义是**并行执行**与**调度/通信 amortization**，不是“减少 cryptographic 工作量”。

**(C) Sigma 与 SUF 走同一条 batched code path**
否则 reviewer 会说“你们只给自己做了 batching”。
最理想：你在 Sigma 和 SUF 共用的算子/调度层实现 batch shape，SUF 只在非线性 kernel 上替换实现（这是你们系统本来就在做的对比）。

---

### 2.3 实现路径（建议按这个顺序做，最稳）

#### Step 0：定义统一的张量布局

把所有激活统一成两种表示之一（选一种贯彻到底）：

* **方案 1（推荐）：flatten tokens**
  把 `[B, L, H]` 视为 `[T, H]`，其中 `T = B·L`。
  优点：线性层/FFN 改动极小，LayerNorm 也可把“token 维”当成 `T`。
  难点：attention 仍然需要知道 `B` 和 `L` 的分块结构来做 `L×L` softmax。

* **方案 2：显式三维** `[B, L, H]`
  优点：attention 写起来更直观。
  缺点：你需要让 GEMM/层之间的接口都支持 3D stride。

工程上，**flatten tokens + attention 单独处理**通常最省事。

#### Step 1：先把“非 attention”的图跑通

在 BERT/GPT2 中，把 embedding、FFN、残差、LayerNorm、activation 都改为支持 `[T, H]`（或 `[B,L,H]`），暂时把 attention 固定在 B=1 的路径验证 correctness。

#### Step 2：实现 batched attention（核心）

你需要把 attention 的三个重块改为 batched：

* `Q,K,V` 投影：这本质是 GEMM，flatten 后就是 `[T,H]×[H,3H]`，无需特殊 batched kernel。
* `QK^T`：对每个 `(b, head)` 做一个 `L×d` 与 `d×L` 的乘法
  → 用 **strided batched GEMM**：batchCount = `B·heads`，strideA/strideB/strideC 按 L×d、d×L、L×L 设置。
* Softmax：对 `B·heads·L` 行做长度 `L` 的 softmax（含 max/sum reduction）
  → 你原来 B=1 的 kernel 如果写成“rows×cols”，只要把 rows 变成 `B·heads·L` 就行。
* `Softmax·V`：同理 strided batched GEMM。

#### Step 3：key/material 管理

true batching 后，一次 forward 消耗的 key material 总量是 `B×`。
你需要保证：

* keygen 侧生成顺序与 online 消费顺序一致
* keyBuf 足够大（你 Table 1 的 key size 本来就很大，batch>1 总量会更大）
* 记录并报告：总 key size / batch，以及除以 batch 的 per‑inf key size（应与 batch=1 接近一致）

#### Step 4：日志与指标（必须做）

你现在论文默认 batch=1（6.1 Metrics 段落里写了，reviewer 也正是卡这个）。
因此你需要在 runner 里把以下都打印成**per-batch 与 per‑inf**两套：

* Online wall time：`time_batch_ms` 与 `time_ms_per_inf = time_batch_ms / B`
* Online comm：`comm_batch_GB` 与 `comm_GB_per_inf = comm_batch_GB / B`
* Keygen time、Key size：同上

---

### 2.4 实验矩阵（够回应 reviewer，且不至于爆炸）

我建议最小但“审稿人挑不出毛病”的矩阵：

**模型**：BERT-base、GPT-2（你论文主对比也是这两个方向）
**序列长度**：128 + 512（一个短，一个长；512 能直接回应“长上下文/attention dominate”）
**batch**：1, 2, 4, 8（足够说明趋势）

如果资源允许，再加一个 256（让曲线更平滑），但不是必须。

---

### 2.5 你必须报告的指标（写进论文/appendix）

每个 (model, L, B, system) 报告：

1. **Online latency**

* `ms/batch`
* `ms/inf`（ms/batch 除以 B）

2. **Throughput（强烈建议）**

* `tokens/s = (B·L) / (online_time_s)`
  这比 “ms/inf” 更直观展示 batching 的收益。

3. **Online communication**

* `GB/batch`
* `GB/inf`

4. **Preprocessing**

* `keygen_s/batch` 和 `keygen_s/inf`
* `key_GB/batch` 和 `key_GB/inf`
  （并在文字里解释：per‑inf 应基本不随 B 变动，batch 只是线性放大总量；这与每个 wire 需要 fresh mask/key 的安全要求一致。）

5. **（可选但很加分）message/round count**
   如果你能在通信层统计“发送次数/同步点次数”，就能直接回应 reviewer 的 amortization 关切（批量更大时，round 不变但每轮 payload 更大）。

---

### 2.6 统计方法（避免 reviewer 说噪声/偶然）

对每个点：

* warmup 1 次丢弃
* 正式跑 5 次
* 报告 median（主表）
* appendix 可给 IQR 或 min/max（可选）

并固定环境：

* `SIGMA_PINNED_KEYBUF=1` 这类关键开关要保持一致（你们之前已发现 pinned off 会严重退化）
* `OMP_NUM_THREADS`、GPU 绑定、mempool 开关固定
* SUF 的 `SUF_*_BITS` 固定

---

### 2.7 论文呈现方式（不挤主文页数）

**主文只加一句话**（放在 6.3 或 6.1 Metrics 后面）：

> “We additionally evaluate true batched-tensor execution (batch sizes 1–8). FuseFSS consistently outperforms SIGMA under batching, and we report per-inference latency/communication and throughput in Appendix X.”

**Appendix 放一张表 + 一张图**

* 表：BERT-base, L=128/512 的 batch sweep（两块）
* 图：speedup vs batch（两条曲线：L=128 与 L=512），或者 throughput vs batch

---

### 2.8 关键风险与避坑（不提前处理会很痛）

1. **“batch=1 数字不一致”风险**
   你现在不同日期/不同脚本下 batch=1 结果有差异（你自己也观察到了）。因此 true batching 实验一定要：

* 用同一个脚本/同一个二进制生成 batch=1..8 的结果
* appendix 的 batch=1 行要与主文 Table 1/2 的 batch=1 来自同一套 pipeline
  否则 reviewer 会抓“不一致”。

2. **keybuf/内存爆炸**
   Table 1 里单次推理 key size 已经是十几到几十 GB 级别（比如 GPT‑Neo 更大）。
   true batching 的总 key size 是线性 `×B`。
   建议先在 **BERT-base/GPT-2、L=128** 上做 batch sweep，把 B 上到 8；L=512 可以只做到 B=2 或 B=4（视内存决定），论文里如实说明限制即可。

3. **“你们 batch 其实只是 internal loop”质疑**
   所以 appendix 一定要写清楚：

* true batching：一次 forward 输入为 `[B,L]`，中间张量为 `[B,L,H]` 或 flatten `[B·L,H]`
* internal batch：loop B 次 forward
  两者差别用一句话说清楚即可。