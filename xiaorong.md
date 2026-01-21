这核心诉求是“**性能提升的归因**”：你现在同时做了编译器层（双模板结构、掩码规则、填充/固定形状）、运行时层（批处理/打包）、以及 GPU 内核工程。只给 end-to-end 的加速曲线，会让读者（和审稿人）很难判断：

* 哪个改动是真正的主因、哪个只是辅助；
* 这些技术之间有没有强耦合/互相“叠加”；
* 你的结论能否迁移到别的模型/输入分布。

所以从系统/编译器/高性能实现类论文的常见标准来说，这个评价**总体是合理的**；其中 (i)(ii) 属于典型 ablation 诉求，(iii) 更像是 profiling/成本拆分，但也很常见、也很有说服力。

下面给你一个**可以直接落地**、尽量最小化额外工程量、同时能覆盖审稿人三点要求的“完整消融实验设计方案”。

---

## 总体原则：把“归因”做成可复现实验开关

先把你的系统拆成 4 个可控因子（尽量用编译/运行参数或宏开关实现）：

1. **P：Padding/固定形状**

    * P=1：所有动态维度（如 seq_len）统一 pad 到固定上界，所有 key/中间张量形状固定
    * P=0：保留“依赖掩码/实际长度”的 key 形状（或按实际长度生成/选择）

2. **T：模板调用/实例化优化（双模板结构等）**

    * T=1：启用双模板结构、模板缓存/融合/去重，使模板调用次数最少
    * T=0：退化到“每个算子/每个 shape 都走一次模板路径”（禁用缓存/融合/去重）

3. **B：运行时批处理/打包（batching/packing）**

    * B=1：启用你的 runtime batching、请求聚合、打包（packing）策略
    * B=0：关闭 batching/packing，按请求逐个执行（或固定 micro-batch=1）

4. **K：GPU 内核工程（可选但强烈建议加）**

    * K=1：启用自研/优化内核
    * K=0：回退到 reference 实现（如朴素 kernel、或 cuBLAS/cuDNN/Thrust 等 “非工程化”版本）

审稿人的 (i)(ii)(iii) 主要围绕 P/T/B（以及 TFSS/Beaver/AND/B2A 的成本分解），但**前半句已经点到了 GPU 内核工程**，所以我建议你至少准备一个 K 的开关（哪怕只是“关闭自研 kernel，回退到通用实现”），否则很难解释“GPU kernel engineering 的贡献”。

---

## 统一实验设置（所有消融共用）

### 硬件/软件与控制变量

* 固定同一台机器/同一张 GPU（型号、显存、驱动、CUDA 版本写清楚）
* 固定 CPU、内存、NUMA 设置；尽量固定 GPU 时钟（避免自动 boost 带来的方差）
* 固定安全参数、协议配置、线程数、通信后端（若是两方/多方）
* 固定模型权重、量化/比特宽配置、输入预处理
* 每个配置：

    * warm-up 若干次（比如 20 次）
    * 正式测量 N 次（比如 200 次），报告 **p50/p95** + 均值±标准差（或置信区间）
    * 同一随机种子/同一输入集合，保证可比

### 指标（至少这些）

* **在线（online）**：端到端延迟（ms），p50/p95；吞吐（req/s 或 tokens/s）；GPU 利用率
* **离线（offline / preprocessing）**（若你的系统区分）：key 生成/预处理时间；预处理吞吐
* **通信**：总字节数、通信轮次、通信耗时
* **模板相关**：

    * 模板调用次数（per inference、per layer）
    * 模板实例数量（unique templates/unique shapes）
    * 模板缓存命中率
    * 模板 dispatch CPU 开销
* **原语相关**（用于 (iii)）：TFSS / Beaver / AND / B2A 的时间与调用次数

### 计时与剖析方法（非常关键）

建议你做“分层计时”，避免只看 wall-clock：

* Host 侧：高精度 timer（如 `std::chrono`）
* GPU 侧：CUDA events（kernel 执行时间更准）
* 通信：在 send/recv/等待处计时并记录字节
* 每个 inference 记录一个 breakdown：

    * `template_dispatch`
    * `packing/unpacking`
    * `gpu_kernels`
    * `communication`
    * `other (sync/memcpy/etc.)`

---

## 消融实验 1：回答 (i)——填充/固定形状相比掩码依赖 key 形状节省多少时间

### 目的

量化 **P=1 vs P=0** 带来的收益，并解释收益来自哪里：

* 形状固定 → 更少 unique shape → 更少 key/template 组合 → 更高缓存复用率
* vs padding 带来的额外计算/额外通信（若存在）

### 适用模型与输入

优先选 **Transformer 类**（有 mask/变长输入的模型），因为 (i) 的争议主要在这里。
对每个 Transformer 模型，准备一个“真实/代表性”的长度分布：

* 取你论文实验里常见的 `Lmax`（如 128/256/512）
* 构造输入长度集合 `L ∈ {L1, L2, ...}`（例如 64/128/256 或根据数据统计分位数）
* 生成一个请求序列（例如 1000 个请求），长度按真实分布采样并随机打乱（模拟线上到达）

### 对比配置（只动 P，其他尽量固定）

* **Cfg-A（mask-shape）**：P=0，T 固定为某个值（建议 T=1，避免把模板问题混进来），B=0（先关 batching，降低耦合），K 固定
* **Cfg-B（padded fixed-shape）**：P=1，其余同上

> 为了让审稿人闭嘴，强烈建议你至少做两组：
>
> 1. **B=0**（不 batching）测“纯 P 的影响”
> 2. **B=1**（开 batching）测“P 对 batching 可用性的放大效应”

所以完整建议是 2×2（P × B）：

* P0B0、P1B0（隔离 P 的直接影响）
* P0B1、P1B1（展示 P 让 batching/packing 更有效）

### 度量与记录

对每个请求序列，记录：

* 端到端延迟（p50/p95）
* 模板实例数量：`#unique_templates`、`#unique_shapes`
* key 相关：`#unique_key_shapes`、key 缓存命中率、key 总大小
* breakdown：dispatch / GPU / comm / packing

### 如何“解释时间节省”

你最终最好给一个“净收益解释”，避免审稿人反咬“padding 增加计算量”：

* 报告两条曲线：

    1. **总延迟**：P1 vs P0
    2. **额外计算成本**：由于 padding 导致的 `gpu_kernels` 增量（如果有）
* 再报告一张表：

    * `template_dispatch` 降了多少
    * `unique templates` 降了多少
    * `key cache hit` 提升了多少

这样即使 P1 在某些短序列上多算了一点，你也能清楚地说：
**“padding 牺牲了一点算力，但换来了模板/缓存/批处理的更大收益，净效应为 X%。”**

---

## 消融实验 2：回答 (ii)——延迟增益有多少来自更少模板调用，有多少来自运行时 batching/packing

这是典型的**2×2 因子实验**，最容易“可解释地归因”。

### 因子定义

* **T（模板调用优化）**

    * T=0：禁用双模板结构/模板缓存/融合/去重，让调用次数回到“朴素版本”
    * T=1：启用你的优化（双模板结构等），模板调用最少

* **B（运行时 batching/packing）**

    * B=0：关闭 batching/packing（micro-batch=1，或完全不聚合）
    * B=1：启用 batching/packing（按你的默认策略）

### 4 个配置

在固定 P、K 的前提下（建议固定 P=1，K=1，避免混杂）：

1. **T0B0**：无模板优化 + 无 batching（最朴素）
2. **T1B0**：有模板优化 + 无 batching（纯模板收益）
3. **T0B1**：无模板优化 + 有 batching（纯 batching 收益）
4. **T1B1**：两者都有（完整系统）

### 工作负载（建议两类）

1. **单请求延迟**（更贴近 “template call overhead”）

* 并发 = 1
* 记录 p50/p95 以及分解：dispatch / GPU / comm / packing
* 特别关注 `template_dispatch` 的占比

2. **稳态吞吐/并发场景**（更贴近 batching 的真实价值）

* 并发客户端数：1/2/4/8/16/...（你可以选到 GPU 饱和）
* B=1 时记录：

    * 平均 batch size 分布
    * 排队等待时间（batching 带来的 queueing）
    * 端到端延迟与吞吐的 trade-off 曲线

### 归因计算（直接写进 rebuttal/论文）

用 **difference-in-differences** 给出“模板 vs batching”的贡献，并可量化交互项：

* 模板带来的增益（无 batching 条件下）：

    * `ΔT = Lat(T0B0) - Lat(T1B0)`

* batching 带来的增益（无模板优化条件下）：

    * `ΔB = Lat(T0B0) - Lat(T0B1)`

* 二者同时启用的总增益：

    * `ΔTB = Lat(T0B0) - Lat(T1B1)`

* 交互/叠加项（是否“1+1>2”）：

    * `Interaction = ΔTB - (ΔT + ΔB)`

然后你可以非常清晰地回答审稿人：

* “模板优化贡献了 X ms（Y%）”
* “batching/packing 贡献了 A ms（B%）”
* “交互项为 C ms（说明固定形状/模板减少使 batching 更有效/或反之）”

### 额外建议：把“模板调用次数”也报出来

审稿人写了“更少的模板调用”，那你最好给他一个硬指标：

* `#template_calls / inference`
* `#unique_templates loaded`
* `avg dispatch time per call`

这样他就没法说你是在“拍脑袋归因”。

---

## 消融实验 3：回答 (iii)——按模型细分 TFSS 与 Beaver/AND/B2A 的贡献

这本质是一个**按原语分类的性能剖析**（profiling），工作量通常不大，但说服力很强。

### 你需要输出什么

对每个模型（你论文里所有模型，至少代表性地覆盖 Transformer/CNN/MLP 等），输出：

* 总在线延迟 `Latency_total`
* 其中：

    * `Latency_TFSS`
    * `Latency_Beaver`（如果你能细分：MatMul/Conv 的 Beaver 消耗更好）
    * `Latency_AND`
    * `Latency_B2A`（必要时也加 A2B）
    * `Latency_Communication`
    * `Latency_Other`（packing、memcpy、sync、template dispatch 等）

同时给出每类原语的 **调用次数/规模**：

* `#TFSS_calls`、TFSS 处理的元素数量
* Beaver triples 消耗数量（或乘法数量）
* AND gate 数量（或 AND 运算数量）
* B2A 转换次数与规模

> 审稿人要的是“贡献”，所以你至少要给：
>
> * **绝对时间（ms）**
> * **占比（%）**
    >   两个维度。

### 如何实现（建议做一个“profiling mode”）

在 runtime/执行器里给每类原语加统一计时器：

* CPU wall-time：统计调用前后时间
* GPU kernel time：用 CUDA events 统计（避免异步导致计时错位）
* 通信：在 send/recv/wait 处统计时间与字节

注意事项：

* 如果原语内部同时包含“通信 + kernel”，要么拆成两项，要么统一记到原语项里，但要保持一致并在文中说明
* 统计时避免每次计时都强制 `cudaDeviceSynchronize()`（会扭曲时间）。正确做法是用事件和流同步来取时间。

### 展示方式（强烈建议）

* 每个模型一张 **堆叠柱状图**（各原语占比）
* 再来一个表格列出（ms, %）以及调用次数
  这样你可以直接回应 (iii)，并且还能顺带解释“为什么某些模型加速更大/更小”（例如 TFSS 在某类激活/比较上占主导）。

---

##（可选但推荐）补一组“编译器 vs 运行时 vs kernel 工程”的高层消融

因为审稿人第一句还点名了 GPU 内核工程，你可以用一个“leave-one-out”把三大块拆开（不必做 16 个组合，做 4 个就够有力）：

* **Full**：P=1, T=1, B=1, K=1
* **No-Compiler**：P=0, T=0（或尽量回退编译器优化），B=1, K=1
* **No-Runtime**：P=1, T=1, B=0, K=1
* **No-Kernel**：P=1, T=1, B=1, K=0

然后报告端到端延迟变化（以及关键 breakdown）。这能一锤定音地回答“你每一块到底值多少”。

---

## 最终你在论文/回复里怎么写（建议组织结构）

你可以把新增实验放成三个小节（或一个小节 + appendix）：

1. **Padding vs mask-shape**：展示在变长输入下的模板/缓存/吞吐收益与额外计算开销
2. **Template vs batching（2×2）**：给出 ΔT、ΔB、Interaction 的归因
3. **Primitive breakdown**：按模型给出 TFSS/Beaver/AND/B2A 的时间占比和调用规模

这三点基本能完整覆盖审稿人 (i)(ii)(iii)，并且逻辑闭环：

* (i) 解释“固定形状”为什么有效
* (ii) 解释“编译器 vs runtime”各自贡献
* (iii) 解释“密码学原语层面”不同模型的成本结构

---

如果你愿意，我也可以把上面的方案进一步“落地成你论文实验表格/图的模板”（比如每张图/每张表应该放哪些列、怎么写 caption 才能精准回应审稿人），并给出一份可直接复制到 rebuttal 的文字框架。
