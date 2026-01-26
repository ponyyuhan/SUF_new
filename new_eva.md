## 1) 解决 batch=1：加 **吞吐** 与 **微批处理** 曲线（必须补）

**新增实验 D’：Batch scaling（端到端 + block）**

* **设置**：batch ∈ {1,2,4,8}（至少到 8；显存允许再加 16）
* **模型**：至少选 2 个代表（BERT-base + GPT2/GPT-Neo），保持与 Table 1 主结果一致。
* **指标**：

    * latency（ms / inference）
    * throughput（tokens/s 或 elements/s）
    * online comm（GB）与 projected time（LAN/WAN）
* **展示方式**：

    * 1 张图：batch-吞吐曲线（Sigma vs SUF）
    * 1 张图：batch-延迟曲线（或表格给出 batch=1/4/8 三点）


---

## 2) 解决 seq≤128：把 seq sweep 做到 **≥256**，并解释/修复你现在的 256 崩溃（必须补）

你目前 BERT-base 在 seq=256 崩溃（`cudaMemcpy invalid argument`，gpu_mem.cu），而 GPT2 已经做到 256 了。

**新增实验 D’’：Longer sequence scaling**

* **目标覆盖**：

    * Encoder：BERT-base seq ∈ {128,256,384,512}（至少拿下 256）
    * Decoder：GPT2 seq ∈ {128,256,512}（你已到 256，可再补 512）
* **如果 BERT 的 attention O(L²) 太重**：仍建议至少做 **block-level** 的 softmax/norm 在 L=256/512（这样能证明“算子/块”可扩展，即使端到端太慢）。
* **你需要在论文里明确**：

    * 端到端长序列受限于 attention 计算/显存，并不等价于你的标量算子编译器瓶颈
    * 但你仍提供：**(a) block-level 长序列** + **(b) decoder 端到端长序列** + **(c) 解释为何 encoder 端到端更难**

**工程上怎么把 seq=256 跑通（建议最小排查路径）**

* 先确认是否是 **缓冲区大小/对齐/32-bit size 溢出** 导致 memcpy 参数非法（尤其是按 byte 计的 size/offset）。
* 打开一次“形状与 buffer size 打印”：在出错前打印每次分配与 memcpy 的 byte 数，确认是否 >2³¹ 或出现负/未初始化。
* 如果是固定上限：把相关 `MAX_*` / keybuf / scratch 大小改成随 L 动态增长，或按 layer 分块执行（即便慢，也能生成有效数据点）。

---

下面这三套系统里，**BOLT / BumbleBee 有可用的开源代码路径**；**IRON 虽然有 GitHub 仓库，但从仓库自述看并不包含完整可复现的加密推理实现**（至少目前公开的主要是“明文/定点编码下的实验脚本”，并写了“other codes in submission: to appear”）。([GitHub][1])

---

## 1) 开源状态核对（以及“能不能当可执行基线”）

* **IRON**：有公开 repo（`xingpz2008/Iron`），但 README 明确说明当前只包含明文 CPP/Tensorflow BERT/CCT(ViT) 的固定点编码实验脚本，且加密推理相关“other codes… to appear”。因此**大概率无法作为“已执行 baseline”复现实验**；更适合走你说的 (b) 路线（解释不可比/不可复现）。([GitHub][1])
* **BOLT (Oakland/S&P’24)**：作者声明已开源，并给出 repo；其 README 指向“可复现实现”在其 EzPC fork 的特定分支路径。([GitHub][2])
  论文还写明实现基于 **EzPC/SCI + SEAL**。([encrypto.de][3])
* **BumbleBee (NDSS’25)**：有公开 repo（`AntCPLab/OpenBumbleBee`，自述为 proof-of-concept），并且论文给出它在 2PC setting 下覆盖 **BERT/GPT2/LLaMA-7B/ViT** 等多模型。([GitHub][4])
  代码依赖 SPU；SPU 官方 README 里注明 **NVIDIA GPU 目前是 experimental**，所以多数情况下它更偏 CPU/分布式运行环境。([GitHub][5])

---

## 2) 如果要把它们加入“比较基线”，推荐怎么设计实验（尽量少踩不可比坑）

### A. 把 baseline 分成两组呈现（避免 reviewer 说你“硬比”）

**组 1：同威胁模型/同 setting 的主比较（正文）**

* 仍以 **Sigma / 你的 FuseFSS** 为主（同为两服务器预处理模型、在线阶段由两服务器交互）。

**组 2：两方私有推理系统（Appendix / “Broader comparison”小节）**

* **BOLT、BumbleBee** 放在这里：明确它们是 **2PC（client–server）** 私有推理，而你是 **two-server preprocessing / non-colluding servers**。
* 这里可以报告：端到端时间、通信（以及可选的“按你 LAN/WAN 参数投影”），但**措辞上避免“我们更快”式结论**，强调“不同 setting，仅作参考”。

（BumbleBee 自己也提醒：不同执行环境下 timing 直接对比不一定公平。）

---

### B. BOLT：如何做“可执行 baseline”

**目标任务（交集最大、最不容易扯皮）**：BERT-base, seq=128, batch=1（与你现有 Table 1/2 对齐）

**实验步骤**

1. **复现环境**：按 BOLT 论文实现说明（SCI + SEAL）搭建；记录版本/commit。([encrypto.de][3])
2. **网络设置对齐**：BOLT 在论文里用 `tc` 做带宽/延迟模拟（LAN/WAN 多档），你可以直接把他们的 `tc` 配置替换成你论文的 LAN/WAN（例如 1GB/s & 0.5ms；400MB/s & 4ms）。([encrypto.de][3])
3. **指标**（建议与你表格字段对齐）

    * end-to-end latency（含全部在线交互）
    * total comm bytes（send/recv）
    * 你自己的投影公式若要用：需要对 BOLT 抽取 rounds；如果不方便，就直接用 `tc` 实测 wall-clock 更干净。
4. **公平性说明**：BOLT 是 2PC/HE+MPC 混合（且没有你这种“预处理键/两服务器”结构），所以把它放 Appendix，标题写清 “2PC baselines (different threat model)”。

---

### C. BumbleBee：如何做“可执行 baseline”

**优先选择他们 repo 自带可跑的例子**，避免你为适配模型浪费时间：OpenBumbleBee README 里给了基于 SPU 的 2PC 启动方式和示例（如 flax ViT）。([GitHub][4])
论文还宣称对 BERT/GPT2/LLaMA/ViT 都做过基准。

**实验设计建议**

1. **最小可复现**：先跑通 repo 自带 example（例如 ViT），把它作为 “coverage proof”。([GitHub][4])
2. **再做一项与主文对齐的交集点**：BERT-base seq=128（如果 repo/脚本支持；否则只做 ViT 并在文中说明原因）。
3. **硬件/平台说明**：SPU 的 GPU 支持标注为 experimental，跑在 CPU 上很常见——因此把 BumbleBee 的结果也放 Appendix，并在 caption 中注明硬件与 setting。([GitHub][5])
4. **指标同 BOLT**：端到端时间 + 总通信；网络同样用 `tc` 对齐到你的 LAN/WAN。

---

## 3) IRON：更推荐走 (b) ——“为何不可直接比较/为何无法执行”

你可以写得非常具体、避免 reviewer 觉得你在找借口：

* **可复现性**：IRON 公开 repo 目前自述只包含明文定点编码实验脚本，未给出完整 secure inference 复现实装（至少公开部分如此）。([GitHub][1])
* **威胁模型差异**：IRON/BOLT/BumbleBee 是典型 **2PC client–server**（一方持输入、一方持模型），而你是 **two-server preprocessing / non-colluding servers**；在线阶段参与方、泄露面、离线物料口径都不同。
* **统计口径差异**：你报告 online/offline 分解；2PC 系统多为端到端（且包含/依赖不同的 HE/OT 开销）。
* **“参考数值”**：如果你想给读者一个量级感，可以引用 BOLT 文中其对 Iron 的复现实验量级描述（他们给出 Iron 的通信/耗时示例），但要明确那是 BOLT 作者的实现与网络设置。([encrypto.de][3])

---

[1]: https://github.com/xingpz2008/Iron?utm_source=chatgpt.com "Iron: Private Inference on Transformer"
[2]: https://github.com/Clive2312/BOLT "GitHub - Clive2312/BOLT: Privacy-Preserving, Accurate and Efficient Inference for Transformers"
[3]: https://encrypto.de/papers/PZMZS24.pdf "BOLT: Privacy-Preserving, Accurate and Efficient Inference for Transformers"
[4]: https://github.com/AntCPLab/OpenBumbleBee?utm_source=chatgpt.com "AntCPLab/OpenBumbleBee: The repo of \"BumbleBee ..."
[5]: https://github.com/secretflow/spu "GitHub - secretflow/spu: SPU (Secure Processing Unit) aims to be a provable, measurable secure computation device, which provides computation ability while keeping your private data protected."
