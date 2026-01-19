According to a document from **2025** (the IEEE S&P paper *“Improved Constructions for Distributed Multi-Point Functions”*), **IntervalLUT 这种“k 段常值区间函数”在理论上确实可以被归约成 DMPF/DMIF（Distributed Multi-Point / Multi-Interval Function）这一类结构**，并且能做到“**单次（或近单次）树遍历**”的 Eval——这跟你在代码里 `SecureIntervalLutHeaderV2` 的注释里写的 “DPF/CDPF-based, single/near-single traversal” 在方向上是一致的。

但关键点在于：**DMPF/DMIF 在你这个 SUF 场景（大域、每元素做一次 Eval、元素数巨大、且 payload 是矢量 u64）时，往往并不“天然”更省 key bytes**，甚至可能更贵；它主要优势更偏向 “FullEval（把整张表都评出来）” 的场景，而你这里是“对每个元素做一次 point Eval”。下面我按你问的两个问题来回答，并给出我理解的 **Route B1（DMPF/DMIF 路线）可落地方案** + 我对它的风险评估。

---

## 1) DMPF/DMIF 跟 IntervalLUT 的对应关系：为什么“看起来很对”

你现在的 IntervalLUT（k 段）本质是：

* 输入：`x ∈ [0, 2^n)`（这里其实是 `hatx`，公开）
* 秘密区间端点：`a0 < a1 < ... < ak`（由 mask shift + wrap split + rotate 得到）
* 输出：`β_i ∈ (Z_{2^64})^{out_words}`，当 `x ∈ [a_i, a_{i+1})` 输出 β_i

这就是 **k-interval function**。

DMPF 论文里给了 **DMIF（Distributed Multi-Interval Function）** 的概念，并说明可以把区间函数视为 **“差分点函数的前缀和”**：把每段常值函数写成“在区间左端点 +β，在右端点 -β，再对输入做 prefix-sum”。这正是你现在 SIL1（k-1 个 DCF）在干的事情，只不过你是把 prefix-sum 拆成 k-1 次比较再加权求和，而 DMIF 试图用一个“单遍历”的结构把 prefix-sum 内化掉。

更重要的是：论文给了 DMIF.Eval 的“单遍历”算法（他们的 Fig.5）：沿着输入位从高到低走树，一边更新 `(seed, sign)`，一边在 `x_i=1` 时把“左兄弟子树的累积贡献”加到输出累计量 `S` 上；最终输出就是 `S`。

这套结构在概念上就是你注释里说的 **CDPF / cumulative** 那味道：

> 当走到右子树，就把左子树整块贡献一次性加入。

---

## 2) 但 DMPF/DMIF 在 SUF 的“key bytes 爆炸”场景里是否真的合适？

### 2.1 DMPF 的 key size / eval time 标度（论文 Table 1 直接告诉你它怎么长）

论文 Table 1 给了几种 DMPF 构造（Naive / PBC / Big-state / OKVS-based）在 **key size**、**Eval time**、**Gen time** 上的标度关系。核心结论（对你重要的）是：

* **Big-state** 的 key size 里有 `t(λ+2t) log N`，也就是 **对 t 有二次项**；t 大了会非常痛。
* **OKVS-based** 的 key size 是 `log N × OKVS.Codesize + OKVSconv.Codesize`，本质还是每层一个 OKVS（规模跟 t 线性相关），所以对 point-Eval 场景来说 key bytes 通常还是会带着 `t log N` 的因子跑。

而且论文在讨论 big-state 时也明确提到：随着 t 增大，big-state 会变得不那么有竞争力（主要因为 key/gen 的增长太快）。

### 2.2 用你这类参数粗算一下（为什么我会怀疑 DMPF/DMIF 是“救 key bytes 的正路”）

假设典型 SUF 里：

* `n = in_bits ≈ 37`（比如 37/50 都常见）
* `k = interval_count ≈ 32`（piecewise 多项式很常见 16/32 段）
* DMIF 需要的“差分点”大致是 `t ≈ 2k`（左右端点各一个；即使你优化到 k+1 也还是 Θ(k)）
* `out_words = 8`（比如 8 个 64-bit 系数/中间量）
* 输出群大小 `log|G| = 64*out_words = 512` bits
* `λ = 128` bits

那么仅用 Table 1 的 big-state DMPF key size 公式估算（先不加 DMIF 的额外开销）：

* `t(λ+2t) log N = 64*(128+128)*37 = 606,208 bits ≈ 74 KiB`
* `t log|G| = 64*512 = 32,768 bits ≈ 4 KiB`
* 合计 **≈ 78 KiB / element**（这还没算 DMIF 需要把输出累积“塞进树节点”的额外结构）。

**78 KiB/element** 这个量级，在你“元素数几十万/上百万”的模型上，依然是“几十 GB”级别的 key 体积——这解释了我为什么说：**DMPF/DMIF 可能并不是你这个场景下“靠它就能把 key bytes 砍到 Sigma 以下”的必然解。**

再强调一遍：DMPF/DMIF 的原始动机是把 “t 次 DPF/DCF 的 FullEval” 压到一次树遍历，提升 FullEval；但你现在是“每个元素只 Eval 一次”，所以它的优势不一定能覆盖它的结构开销。

---

## 3) Route B1（按你设想走 DMPF/DMIF）怎么做才算“可落地”

下面给你一个“真的能在 SUF 里塞进去”的 B1 方案，我会同时标注 **你最可能踩的坑** 和 **怎么把它改到 GPU 还算能跑**。

### 3.1 B1 的目标定义

你要的 SIL2 不是“把 SIL1 的 k-1 次 DCF eval 并行化”，而是：

* **Keygen**：从 `k` 段区间 + `out_words` payload（u64 向量）生成两份 key
* **Eval**：输入公开 `hatx`，每方单次遍历 `n` 层（或接近一次），输出 `out_words` 个 u64 share
* **Key bytes**：希望去掉 SIL1 那种 `O(k * n * out_words)` 的大项

Route B1 里，你把 IntervalLUT 视为 DMIF，按 DMPF 论文的“从 DMPF 改 DMIF”的路做：

* 把区间常值 `β_i` 转成端点的差分点值（`+β` / `-β`）
* 让 eval 在树上走时，把经过的“左兄弟子树贡献”累积到输出（单遍历）

### 3.2 Key 结构（你代码里 `SecureIntervalLutHeaderV2` 要承载什么）

建议你把 V2 key 设计成 “DMIF key” 而不是 “k 份 DCF key 的拼接”，即：

**Header（固定）**

* `version = 2`
* `in_bits`、`out_words`、`k`（interval_count）
* `t`（差分点个数，通常 2k 或 k+1）
* `okvs_kind`（如果选 OKVS-based）
* `prg_kind`（ChaCha12/AES-CTR）

**Body（每 party 一份）**

* `root_seed`：λ bits
* `root_sign`：l bits（取决于你用 big-state 还是 OKVS-based；big-state l=t，OKVS-based l=1）
* 对每一层 `i=1..n`：

    * `CW(i)`：这一层的 correction structure（big-state 是一个长度 t 的数组；OKVS-based 是一个 OKVS code）
* （可选）`CW_conv`：如果你把输出搬到 convert layer（DMPF 模板这样做）

**注意：** DMIF.Eval（Fig.5）那版是把输出累积 `S` 放在树节点 PRG 输出里，所以不一定需要 `CW_conv`（它的伪码里 key 只 parse 到 `CW(n)`）。

### 3.3 选哪个底层 DMPF 构造：big-state 还是 OKVS-based？

#### 我建议：优先 OKVS-based（l=1）——但必须换一个 GPU-friendly 的 OKVS

理由：

* big-state 的 key size 有 `2t` 项，t 稍大就很难看（Table 1）。
* big-state 每层 correction 是“t 个大字符串”的线性组合（等价要做一个按 sign 的内积），这对 GPU 来说很不友好。
* OKVS-based 的 Correct 逻辑是 **Decode 一个 code**，如果你选 XOR-filter / 3-hash OKVS，Decode 就是 3 次 load + XOR（常数很小），更适合 GPU（虽然 key size 仍会随 t 增长）。Table 1 也把 OKVS-based 的 Eval time 写成每层一个 `OKVS.Decode`。

> 论文里他们实现用的是 RB-OKVS，并且提到 OKVS 有显著实现开销（甚至提到 ×20 级别），所以你如果照搬 RB-OKVS 也会很痛。

**落地建议：**
在 SUF 里实现一个“XOR-filter 风格的线性 OKVS”（3 个哈希、m≈1.23t），Decode 3 次 load + XOR。Encode 在 dealer 端做 peeling（失败重试），失败概率可控。这样你把 OKVS decode 成本压到真正 GPU 友好的常数级。

### 3.4 DMIF 的 Eval（你可以直接照 Fig.5 写 GPU kernel）

DMIF.Eval 的骨架就是（对应 Fig.5）：

* state：`seed`, `sign`, `prefix`（prefix 可以不显式存，直接用层号和输入位）
* 维护累积输出 `S`（向量 out_words）
* 每层：

    1. `C ← Correct(prefix, sign, CW(i))`（OKVS decode 或 big-state inner product）
    2. `(seed0, seed1, sign0, sign1, res0, res1) ← PRG(seed)`
    3. 如果 `x_i==0`：走左孩子更新 seed/sign（带 correction）；如果 `x_i==1`：走右孩子更新 seed/sign（带 correction），并把 **左兄弟子树的贡献**累积到 `S`（Fig.5 在 `x_i==1` 时加 `(res0 ⊕ Cres0)`）。

**你需要做的“SUF 特化”有两点：**

1. 把所有 `⊕` 替换成你选择的群运算：如果输出要是 `Z_{2^64}` 的 share，那就是 `+ mod 2^64`（对 seed 仍是 XOR）。
2. `res0/res1` 不是比特串，而是 `out_words` 个 u64 —— PRG 输出要能吐出足够多位（ChaCha block 不够就多 block）。

### 3.5 DMIF.Gen（Keygen）怎么跟你的 compiler 输出对接

你 compiler 现在已经能给出：

* “无 wrap 的有序 cutpoints + 旋转后 payload 表”
  （你说它做了 wrap interval split + rotate，这对 DMIF 其实是好事：保证区间是 disjoint 且覆盖域。）

把它转成 DMIF 的差分点集合：

* 设区间为 `[a_0,a_1), [a_1,a_2), ..., [a_{k-1}, a_k)`，其中 `a_0=0, a_k=2^n`
* payload 为 `β_0..β_{k-1}`
* 差分点（最干净的一种写法）：

    * 在 `a_0=0` 放一个点值 `Δ_0 = β_0`
    * 对 `i=1..k-1`：在 `a_i` 放 `Δ_i = β_i - β_{i-1}`（环上减法）
    * （可选）在 `a_k` 放 `-β_{k-1}`，但 `a_k=2^n` 超域；如果你域是 `[0,2^n)`，那最后一项可以省略（因为不会被查询到），或者把它并入你 rotate/normalize 的逻辑里。

然后 DMIF 要实现：`f(x) = Σ_{a_i ≤ x} Δ_i`（prefix sum），这刚好就是 Fig.5 那种“遇到右转就加左兄弟子树贡献”的语义。

接下来 DMIF.Gen 的实现路线就是：

* 先按 DMPF template（Fig.1）理解 Gen 是怎么在每层通过 PRG 展开并计算 correction 的
* 再按论文说明，把 PRG 输出扩展为包含 `res0,res1`，并修改 `GenCW`/`Correct` 以 enforce “差分点的 prefix sum” 关系

> 这一步是 B1 里最“工程量大”的地方：你需要写一套 DMIF.Gen（而不只是 Eval）。但它确实是可以按论文给的 template 改出来的（不是凭空造轮子）。

### 3.6 B1 的现实风险总结（我建议你在开工前先接受这些事实）

1. **B1 的 key size 仍然会跟 `t·logN` 走**（因为你每层都要对“special nodes”做 correction 的 encode；不管 big-state 还是 OKVS，本质每层要处理 O(t) 个东西）。这点 Table 1 已经很清楚。
2. DMIF 相比 DMPF 还要额外承载输出累积（PRG 输出变长、correction 里多 res 的部分），论文也说 DMIF “adds two additional g-bit strings” 并需要改 GenCW/Correct。
3. 论文实现里也明确提到 OKVS 的实现开销很显著；你若直接上 RB-OKVS，很可能把 GPU eval / keygen 都拖死。

所以：**如果你的 interval_count 在很多 gate 上接近 32/64，且 out_words 还不小，那么 B1（DMPF/DMIF）大概率不是“终局方案”**。它更像是：

* 你想把 SIL2 做出来、把“单遍历 IntervalLUT”跑起来（功能上补齐论文框架）的一个路线；
* 但不保证能立刻把 key bytes 拉到你想要的量级。

---

## 4) 我给你的建议：B1 该怎么用，才“值回票价”

如果你坚持走 B1，我建议把它定位成两个阶段：

### 阶段 1：只用在 **t 很小** 的地方（比如 ≤ 8 或 ≤ 16）

这类 gate 往往就是你统计里爆炸的那几个（StepDcf/Pred/GAPARS），它们很多时候其实只需要少数区间/阈值。
在这种 t 小的 regime，big-state 或 OKVS-based DMIF 才可能有现实收益，而不会被 `t logN` 的 key size 吃掉。

### 阶段 2：t 大的 IntervalLUT 另开路线（不要硬上 DMPF/DMIF）

你真正想要的 “SIL2 级别、key bytes 不再乘 `k·logN`” 的 IntervalLUT，**很可能需要一个“利用你 cutpoints 是固定 α_i + mask shift”的专用构造**，而不是通用 DMIF。通用 DMIF 把端点当作任意集合，信息论上就要花 `Θ(t logN)` 的描述量（除非你能利用结构把描述量降成 `Θ(logN)`）。

---

如果你愿意我继续往下写：我可以把“阶段 2 的专用构造”也展开成一个可实现的设计草案（包括：如何利用 `α_i` 固定、只把 mask shift 编码一次、怎么把 carry/wrap 处理进 key、以及怎么把输出矢量只放在 O(k) 的位置而不是 O(k logN)），但那条路就不再是 DMPF/DMIF，而是更接近你注释里说的 **CDPF / cumulative + instance separation** 的“专用 IntervalLUT”。
