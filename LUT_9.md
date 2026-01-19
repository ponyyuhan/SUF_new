下面这份是**严格基于代码现状**给出的优化路线：我会按你要的 **P0 / P1** 来写，并把 **Sigma 论文/工程里能直接借鉴的点**明确映射到你们项目的“该开哪些开关 / 该改哪些文件”。最后我会非常直白地回答：**在当前实现与配置下**，以及**做完这些改动后**，你们是否真的能“稳定优于 Sigma”。

---

## 我在代码里确认到的两个“硬伤”与一个“机会点”

### 硬伤 A：sigma_gpu 的 XOR-open bit 仍然是 “1bit 用 8B 传”

在 **`include/runtime/phase_tasks.hpp`** 里：

* `GeluSigmaGpuTask::OpenBits`：把 **(d,i) 两个bit**当成 `2*N` 个 `uint64_t` 发送/接收
* `NExpSigmaGpuTask::OpenBit`：把 **1个bit**当成 `N` 个 `uint64_t` 发送/接收

这类 bit-open 不会体现在你 `OpenCollector` 的 `open_wire_bytes_sent` 统计里（因为它绕过 OpenCollector 走了 `net_chan->send_u64s`），但它确实在 wall time 里烧 **带宽 + CPU memcpy + cache miss**。

> Sigma 论文里明确强调了 “把 ring 元素打包再传输” 能带来显著收益（它们甚至为任意 bitwidth 做了 GPU packing kernel）。你们 OpenCollector 已经做得很像 Sigma 了，但 **这条 bit-open 旁路还没吃到打包优化**。

---

### 硬伤 B：你们当前“overlap 的理论能力”在 transformer_layer 默认被阉割了（budget=0）

你自己已经观察到：

* `open_flushes_budget = 0`
* `open_flushes_demand = open_flushes`

我在 **`src/nn/transformer_layer.cpp`**（你贴过的那段，SUF_newest_2 里也存在）看到：你们把 `lazy_limits` 的触发阈值设成了 **hard limit 的最大值**：

```cpp
lazy_lim.open_pending_words = open_lim.max_pending_words;
lazy_lim.coeff_pending_jobs = pfss_lim.max_pending_jobs;
lazy_lim.hatx_pending_words = pfss_lim.max_hatx_words;
```

这等价于：**永远不做“预算触发的提前 flush”**，所有 flush 都是“我被依赖卡死了才 flush”。
在大模型上这会把 round-trip latency 和 PFSS eval latency **结构性放大**（你自己也解释得非常清楚）。

---

### 机会点：你们 OpenCollector / SecureGpuPfssBackend 已经具备 Sigma 风格的“工程底座”

我看了 `src/runtime/open_collector.cpp` 和 `cuda/pfss_backend_gpu.cu`：

* OpenCollector 已经有：**effbits packing、device pack/scatter、pinned staging、async flush** 等 Sigma 同款工程点（只是需要确保配置开对，且不要被旁路绕开）
* PFSS GPU backend 已经有：**key blob cache**、若干 `eval_*_many_device` 的 staged 结构

所以你们要赢 Sigma，路线很明确：
**把“旁路的 bit-open”纳入 packing + 把 overlap 真的跑起来 + 把 PFSS eval 做到更像“流水线”而不是“拷贝-同步-算-同步”。**

---

# ✅ P0（立刻落地，低风险）：把 sigma_gpu 的 XOR-open bits 改成 bit-pack（32×/64×减少通信+CPU拷贝）

## P0.1：我已经基于 SUF_newest_2 做好了可直接用的补丁

* 位置：`include/runtime/phase_tasks.hpp`
* 改动：

    * `GeluSigmaGpuTask::OpenBits`：把每元素 2 bit 的 `(d,i)` 打包成 **2-bit field**（每 `uint64_t` 承载 32 个元素）
      通信从 `2*N*u64` → `ceil(N/32)*u64`（**约 32×**减少）
    * `NExpSigmaGpuTask::OpenBit`：把 1 bit 打包成 bitset（每 `uint64_t` 承载 64 个元素）
      通信从 `N*u64` → `ceil(N/64)*u64`（**约 64×**减少）

补丁如下：
--- a/include/runtime/phase_tasks.hpp
+++ b/include/runtime/phase_tasks.hpp
@@ -3811,26 +3811,32 @@
}
case St::OpenBits: {
// XOR-open the masked DReLU bits (d_hat, i_hat). These are uniform and safe to reveal.
-        std::vector<uint64_t> my_bits(2 * N, 0ull);
+        //
+        // Engineering: pack the 2-bit (d,i) share per element to reduce wire bytes and host memcpy by ~32x.
+        const size_t packed_words = (N + 31) / 32;  // 32 elems per u64 when eff_bits=2.
+        std::vector<uint64_t> my_packed(packed_words, 0ull);
+        other_bits_.assign(packed_words, 0ull);
         for (size_t i = 0; i < N; ++i) {
-          my_bits[i] = d_mask_share_[i];
-          my_bits[N + i] = i_mask_share_[i];
+          const uint64_t di = (d_mask_share_[i] & 1ull) | ((i_mask_share_[i] & 1ull) << 1);
+          const size_t w = i >> 5;  // /32
+          const unsigned sh = static_cast<unsigned>((i & 31u) << 1);  // *2
+          my_packed[w] |= (di << sh);
         }
-        other_bits_.assign(2 * N, 0ull);
         if (party == 0) {
-          R.net_chan->send_u64s(my_bits.data(), 2 * N);
-          R.net_chan->recv_u64s(other_bits_.data(), 2 * N);
+          R.net_chan->send_u64s(my_packed.data(), packed_words);
+          R.net_chan->recv_u64s(other_bits_.data(), packed_words);
         } else {
-          R.net_chan->recv_u64s(other_bits_.data(), 2 * N);
-          R.net_chan->send_u64s(my_bits.data(), 2 * N);
+          R.net_chan->recv_u64s(other_bits_.data(), packed_words);
+          R.net_chan->send_u64s(my_packed.data(), packed_words);
         }
         d_hat_.assign(N, 0u);
         i_hat_.assign(N, 0u);
         for (size_t i = 0; i < N; ++i) {
-          const uint64_t od = other_bits_[i];
-          const uint64_t oi = other_bits_[N + i];
-          d_hat_[i] = static_cast<uint8_t>((d_mask_share_[i] ^ od) & 1ull);
-          i_hat_[i] = static_cast<uint8_t>((i_mask_share_[i] ^ oi) & 1ull);
+          const size_t w = i >> 5;
+          const unsigned sh = static_cast<unsigned>((i & 31u) << 1);
+          const uint64_t open_di = (my_packed[w] ^ other_bits_[w]) >> sh;
+          d_hat_[i] = static_cast<uint8_t>(open_di & 1ull);
+          i_hat_[i] = static_cast<uint8_t>((open_di >> 1) & 1ull);
         }
         st_ = St::SelectLin;
         [[fallthrough]];
@@ -4082,18 +4088,29 @@
#endif
}
case St::OpenBit: {
-        // Exchange masked XOR-share bits to obtain observed d_hat (still masked).
-        other_bits_.assign(N, 0ull);
+        // Exchange masked XOR-share bits to obtain observed d_hat.
+        // Pack 1-bit shares to reduce wire bytes by ~64x.
+        const size_t words = (N + 63) / 64;
+        std::vector<uint64_t> my_packed(words, 0ull);
+        other_bits_.assign(words, 0ull);
+        for (size_t i = 0; i < N; ++i) {
+          const size_t w = i >> 6;  // /64
+          const unsigned sh = static_cast<unsigned>(i & 63u);
+          my_packed[w] |= ((d_mask_share_[i] & 1ull) << sh);
+        }
         if (party == 0) {
-          R.net_chan->send_u64s(d_mask_share_.data(), N);
-          R.net_chan->recv_u64s(other_bits_.data(), N);
+          R.net_chan->send_u64s(my_packed.data(), words);
+          R.net_chan->recv_u64s(other_bits_.data(), words);
         } else {
-          R.net_chan->recv_u64s(other_bits_.data(), N);
-          R.net_chan->send_u64s(d_mask_share_.data(), N);
+          R.net_chan->recv_u64s(other_bits_.data(), words);
+          R.net_chan->send_u64s(my_packed.data(), words);
         }
         d_hat_.assign(N, 0u);
         for (size_t i = 0; i < N; ++i) {
-          d_hat_[i] = static_cast<uint8_t>((d_mask_share_[i] ^ other_bits_[i]) & 1ull);
+          const size_t w = i >> 6;
+          const unsigned sh = static_cast<unsigned>(i & 63u);
+          const uint64_t open_bit = (my_packed[w] ^ other_bits_[w]) >> sh;
+          d_hat_[i] = static_cast<uint8_t>(open_bit & 1ull);
         }
         st_ = St::Select16;
         [[fallthrough]];


## P0.2：这步在 bert-base/gpt2 上大概能省多少？

以 BERT-base（L=128,B=1）大致量级：

* MLP GeLU：N≈128*3072=393,216
  原先每层 bit 交换 ~ `2*N*8B ≈ 6.0MB/方向`
  现在 ~ `ceil(N/32)*8B ≈ 0.098MB/方向`
  **单层省 ~ 5.9MB/方向，12层省 ~ 71MB/方向**

* Softmax nExp：N≈12*128*128=196,608
  原先每层 ~ `N*8B ≈ 1.5MB/方向`
  现在 ~ `ceil(N/64)*8B ≈ 0.024MB/方向`
  **12层再省 ~ 18MB/方向**

合计仅这两处旁路 bit-open，**每次推理每方向可减少 ~90MB 级别的同步传输与CPU拷贝**。
它不一定能把你 10 秒级 PFSS eval 直接砍没，但它属于“Sigma 工程里最确定的那种：白送的通信/拷贝浪费”。

---

# ✅ P1（真正决定 bert-base/gpt2 能不能翻盘）：让 overlap 真的发生 + PFSS 评估更像 Sigma 的流水

P1 我分成 3 个“最值钱”的子方向，你可以按收益/风险排序推进。

---

## P1.1：把 sigma_gpu 的 DReLU 输出从 “u64/elem D2H” 变成 “packed bits D2H”，彻底吃满 P0 的收益

### 现状（你现在代码里发生的事）

`SecureGpuPfssBackend::eval_drelu*_many_device` 在 GPU 上算完以后，会把 `bool_buf_`（u64/elem）整体 D2H 拷到 host 的 `d_mask_share_ / i_mask_share_`。

即便你 P0 把**网络**传输压缩了，如果 D2H 仍然是 `N*8B`，你依然在：

* GPU→CPU 拷贝大量 “只有 1 bit 有效” 的数据
* CPU 再 pack，再 send

### Sigma 风格的做法（建议你们实现）

在 `cuda/pfss_backend_gpu.cu` 里给 `eval_drelu1_many_device / eval_drelu3_many_device` 增加一个分支：

1. GPU 计算产生 `bool_buf_ (u64/elem)` 后，立刻运行一个 `pack_bits_kernel`：

    * DReLU1：打包成 `ceil(N/64)` 个 u64
    * DReLU3：把 `(d,i)` 合成 2-bit field，打包成 `ceil(N/32)` 个 u64
2. 只把 packed buffer D2H（最好用 pinned host buffer）
3. Gelu/NExp 的 OpenBits/OpenBit 直接使用 packed buffer 通信并 unpack

这样 P0 的收益会同时体现在：

* 网络字节数
* CPU memcpy
* GPU→CPU 拷贝字节数

> 这一步对大模型更“工程正确”，因为你现在大模型瓶颈之一就是各种同步点的拷贝/打包/通信链条。

---

## P1.2：把 `open_flushes_budget` 从 0 拉起来：让“提前 flush”发生，从而真正 overlap PFSS 和 Open

你现在 `open_flushes_budget=0` 的根因在 `transformer_layer.cpp`：lazy limit 等于 hard max，等价禁用预算 flush。

### 目标

让你统计里出现：

* `open_flushes_budget > 0`
* `pfss_flushes_overlapped_in_open > 0`（或者至少 wall time 看到 open_comm 被 PFSS eval 隐藏一部分）

### 最小侵入改法（我建议你直接在 transformer_layer.cpp 做）

在 `transformer_layer.cpp` 里保持 hard limit 不变（避免 OOM），但把 lazy limit 改成“更小的阈值”，例如：

* `lazy_lim.open_pending_words = 1<<18`（约 2MB words 级别，按你 word 定义换算）
* `lazy_lim.hatx_pending_words = 1<<19`（根据 PFSS job 平均 hatx 量调）
* `lazy_lim.coeff_pending_jobs = 1` 或 `2`（避免很多 job 堆在一起才 flush）

这样会把 flush 从 “被依赖卡死才 flush” 推到 “攒到一个合理 batch 就 flush”，就有机会出现：

* CPU 还在准备下一步时，open 已经在后台走了
* PFSS eval 在 GPU 上跑时，open_comm 在 NIC 上跑
* 或者 OpenCollector 的 pack/scatter 在 copy stream 上跑，不挡 compute stream

> 这一步非常像 Sigma 的工程哲学：你不一定减少 rounds，但你要把 rounds 的等待尽可能隐藏。

---

## P1.3：PFSS GPU backend 做双缓冲 staging：把 H2D copy 与 kernel eval overlap（Sigma 同款套路）

你现在 `SecureGpuPfssBackend` 的 pattern 仍然偏“单缓冲 + 同步收尾”：

* copy keys → launch kernel → copy outs → sync copy stream
  这会导致：
* GPU compute 和 H2D copy/格式整理 很难 overlap
* job 与 job 之间空泡明显（尤其是 384 jobs / inference 这种）

### 建议

在 `cuda/pfss_backend_gpu.cu` 的 `SecureGpuPfssBackend` 做一个 2-buffer ring：

* `keys_buf_[2]` + `bool_buf_[2]` + `event_copy_done[2]`
* 对 job i：

    * buffer b = i&1
    * 先 `cudaMemcpyAsync(keys_buf_[b])` 到 copy stream，并 record event
    * kernel stream wait event，然后 launch kernel
    * 同时 copy stream 可以开始 job i+1 的 keys

如果你再配合 **pinned host memory**（keys 与 outs 都 pinned），这在大模型上经常能拿到 **10~30% 的 PFSS wall-time 改善**（取决于你现在 copy/compute 比例）。

---

# ✅ Sigma 的工程优化：你们项目里“能直接对齐”的配置清单

Sigma 论文里明确提到的工程点，你们现在基本都“有框架”，但需要确保**开关与路径真的生效**：

## 1) 通信/打开值打包（Sigma: pack ring elements before transmitting）

你们对应的是 OpenCollector 的 packing 路线（`src/runtime/open_collector.cpp`）。

建议配置（按是否有 GPU-aware net chan 分两档）：

### A. 你们 net chan 能提供 CUDA stream（或你们希望 pack/scatter 在 GPU）

* 开启 device packing / scatter / keep opened（避免 host round-trip）
* pinned staging 让 H2D/D2H 真异步

（具体 env 名称你们代码里是存在的，README 也写了类似项；原则是：**device_pack=1, device_scatter=1, pinned=1, keep_opened=1**）

### B. 你们 net chan 不支持 CUDA（benchmark CountingNetChan 这种）

* 仍然建议保持 open_pack=1（CPU pack 也能减少网络字节数）
* 但重点应放在：**P0/P1.1 把旁路 bit-open 也 pack 掉**（否则你 pack 的只是 OpenCollector 那部分）

## 2) overlap 通信与计算（Sigma 强调 pipeline）

你们对应的是 PhaseExecutor 里的 overlap 逻辑（`include/runtime/phase_executor.hpp`），以及上面说的 **lazy budget flush**。

关键不是“开了 overlap flag 就行”，而是：

* **要出现预算触发的提前 flush**（否则永远 demand flush）
* 以及避免 open packing 占用 compute stream（你们代码里甚至对 open_uses_caller_stream 有专门的 overlap 禁用逻辑）

## 3) key staging / GPU buffer reuse（Sigma 强调 key transfer + GPU kernels）

你们对应的是：

* `cuda/pfss_backend_gpu.cu` 的 staged eval
* 以及你们已经有的 key blob cache

最值钱的是 P1.3（双缓冲）+ pinned memory。

---

# 最后：你们的方案“正确配置后”是否真的能优于 Sigma？

我给一个**不粉饰、但也不悲观**的判断，分两层说清楚：

## 结论 1：在你目前“主要靠 sigma_gpu primitives”的路径下，你们的上限更像“逼近 Sigma + 靠小幅调度/packing 超过一点点”

你们现在已经在 bert-tiny 上出现了 “SUF < Sigma(缓存)” 的情况，但你自己也指出 sigma 日志有噪声、不同缓存文件差异很大。
在这种路径下（大部分非线性用 sigma_gpu 实现），**理论上你们不太可能拿到 paper 里那种 1.7~2.9× 的稳定优势**，原因很简单：

* 你们做的很多核心算子和 Sigma 是同款协议/同款 primitive
* 你们能赢的空间更多来自：

    * 你们移除了某些多余 trunc（比如你们 LUT_7/skip prob trunc 的路线）
    * 你们比 Sigma 更 aggressive 的 packing / overlap
    * 你们 key/material cache 更稳、更少误用（你刚修的 inv cache key 就是典型）

所以：**能赢，但幅度通常是常数级（几 % ~ 十几 %），且对测试噪声敏感**。

## 结论 2：想要达到 paper.tex 里的“明显优于 Sigma”，你们必须让 SUF 的“结构优势”真正落地，而不是在大头上回退到 sigma_gpu

你的 paper.tex 里写的是 “SUF+TFSS 统一两模板结构 → 更少 material、更少冗余、并能更好批处理/流水”。

但你目前 repo 的实际表现（尤其 key bytes 与 paper 表格差距很大）说明：

* 要么你们现在的 secure3 配置/实现路径与 paper 实验口径不同
* 要么你们为了工程可用性在大头上走了 sigma_gpu，从而把 SUF 的“结构优势”抵消了

如果你们把 SUF 的结构优势真正做出来（典型就是：**减少 PFSS eval 的结构性工作量 / 更强的全局 batching / 更低的 round 等待**），那确实存在在 bert-base/gpt2 上**稳定优于 Sigma**的路径；但这就必须落到我上面 P1.2（让 overlap 发生）+ P1.3（PFSS 流水）+ 更激进的 SUF gate 端到端工程上。

## 由于你当前环境无法重跑 Sigma，我能给的“最诚实的验证建议”

在你修完 P0（我给的补丁）并做了 P1.2（lazy budget）后，你可以用你现有统计直接判断“是不是走在能赢 Sigma 的轨道上”：

* `open_flushes_budget` 是否 > 0
* `pfss_flushes_overlapped_in_open` 是否从 0 变成明显 >0
* `open_comm_ns` 的 wall-time 是否开始被隐藏（总在线时间下降大于 open_comm_ns 的下降）
* 同时观察 `pfss_flush_eval_eval_ns` 是否因 P1.3 出现显著下降

这些指标一旦改善，在大模型上一般不是“几毫秒”，而是“几百毫秒到秒级”的趋势变化。

---

## 你现在马上可以做的 2 个动作

1. **先把 P0 补丁打上**（不改协议语义，纯编码方式）

2. **把 transformer_layer.cpp 的 lazy budget 从“等于最大值”调下来**
   目标是：让 `open_flushes_budget > 0`，并观察 overlap 相关计数是否从 0 变正。

---

