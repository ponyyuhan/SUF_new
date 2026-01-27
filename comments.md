### (A) 微基准（B.2）仍缺少“近似参数”的可复现描述

你现在写了“FuseFSS 和 Sigma 用同样的 fixed-point activation approximations”。
这能回应“公平性”方向，但 **仍不够可复现**，reviewer 仍可能追问：

* GELU/SiLU 各自用的 **分段数 m（或区间数）、多项式 degree d、每段系数、边界 α_i** 是什么？
* fixed-point 格式：字长 n、fraction bits f（以及是否做 range reduction）
* 如果有不同模型（BERT/GPT）用不同 hidden，对应的激活近似是否一致？

**建议你补一条非常具体的一句话**放在 B.2 或实验设置里：

> “We use a piecewise polynomial approximation with **m=… intervals** and **degree d=…** (same as SIGMA), under fixed-point format **(n=…, f=…)**; the exact boundaries and coefficients are listed in …/released code.”

（如果不想列全系数，至少把 m、d、n、f 写出来。）

### (B) 全文基本没有报告方差/重复次数

目前 Tables/正文的 latency、keygen time、通信量等大多是单点值，没有：

* 跑了多少次（trials）
* mean±std 或 median+IQR
* 是否 warm-up、是否排除第一次 kernel compile 的 outlier 等

这会让 reviewer 很容易写 “no statistical significance / no variance reporting”。

**建议最小补丁**（不改表格也行）：在 Metrics 段落加一句：

> “All reported numbers are averaged over K runs (K=…), with standard deviation …; GPU kernels are warmed up ….”

### (C) Hardware 仍略有歧义（比原 review 好，但可能还会被问）

你现在写的是 “2× NVIDIA RTX PRO 6000 GPUs … CUDA 13.0”。
这已经去掉 “Blackwell” 的潜在命名冲突，但 reviewer 仍可能要求更精确：

* RTX PRO 6000 是否是 Ada / 具体显存大小
* driver 版本、GPU memory、clock 是否锁频等

**建议**：加括号补齐 “48GB / driver version / exact SKU”。

### (D) padding 开销解释仍偏弱（尤其是 runtime 0.264ms → 1.014ms 近 4×）

你解释了 predicate list 从 384B → 768B，instance size 增长 1.16×，但 **runtime 却增长 3.8×**。
只说“因为 padding predicate list”可能还不够说服 reviewer。

**建议**：加一个更“系统/工程可解释”的 breakdown（哪怕是定性也行），例如：

* predicate 数量从 T₁ 到 T₂（写出具体数值）
* packed comparison kernel 是否按 block size（例如 32/64）分批处理，padding 是否导致多一轮 kernel / 多一次 memory pass
* interval lookup 是否也变成 worst-case M 导致更多 work

如果你能用 profiler 给一段 “time breakdown: pred_eval / lut_eval / postproc”，效果会很好。

### (E) Microbenchmark workload 维度不完全明确

B.2 说 tensor shape 是 (128, hidden)，但 hidden 对应各模型到底是多少没写明。
建议直接写一行：

* BERT-base hidden=768，BERT-large hidden=1024，GPT2-small hidden=768，LLaMA7B hidden=4096（按你实际实验的）

### (F) baseline 组件公平性（实现共享程度）仍可能被追问

ICML reviewer 很容易问：

* 你是否复用了 SIGMA 的 linear layer / attention kernels？还是分别实现？
* FuseFSS 的 end-to-end 增益是否只来自 nonlinearity 编译？
* SIGMA 是否提供/不提供 mask-independent shape 的保护，你现在的比较是否“同泄露等级”？

如果你确实是“只替换 nonlinear gates，其余共用”，请明确写出来；如果不是，也要解释为何仍公平。
