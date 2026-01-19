下面这份方案完全按你现在这份代码结构来写（文件/函数名都对应你 zip 里的实现），目标是**在不改变协议语义**的前提下，把你现在 online 0.318s（bert‑tiny L128 B1）压到接近/优于 Sigma 0.206s。核心抓手就两条：

1. **让 Open flush 真正与 PFSS flush 并行/重叠**（你现在的 bench 里其实“能重叠但被 gate 住了”）
2. **把 trunc 的 pred eval 变成“PackedPred + 低 qn 不浪费 warp”的版本**（否则你一开 packed pred 就会因为 SPC1/warp 访问模式直接炸穿性能）

---

## 0. 先用你给的 profile 结论定位瓶颈（对应你日志）

你目前（SUF + PPCS）这组数据里最关键的是：

* **open_comm_ns ≈ 0.116s**（Open 的网络/同步成本非常显著）
* **pfss_flush_eval_eval_ns ≈ 0.276s**（PFSS eval 段也很重）
* open_flushes=264，pfss_flushes=104（flush 次数高，而且大部分是 demand flush）

你还提到：

> “PhaseExecutor 的 async-open 只有 overlap_pfss_open=true 才启用；bench 默认 pfss_net_chan=null，导致 overlap_pfss_open=false；所以 SUF_OPEN_FLUSH_ASYNC=1 没有预期收益”

这句话对你当前代码来说是**关键命门**：你 bench 的 PFSS channel（CountingChan）和 Open channel（CountingNetChan）本来就是两套独立 ring buffer，但 transformer_layer_forward 里把 overlap_pfss_open 错误地只绑定到 `pfss_net_chan`，导致 PhaseExecutor 的 overlap 逻辑（`want_open && want_pfss` 的并行 flush）根本没用上。

---

## 1) 立刻能见效的改动：修正 overlap_pfss_open 的判定（bench 直接吃到并行）

### 1.1 问题点（你代码里的具体位置）

在 `src/nn/transformer_layer.cpp` 的 `transformer_layer_forward(...)` 里（你已经有 `ProtoChanFromNet` 的构造逻辑）：

* `phase_R.pfss_chan` 可能是：

    * 直接用传进来的 `pfss_ch`（bench 下是 `CountingChan`，独立于 open 的 net_chan）
    * 或者用 `ProtoChanFromNet(*pfss_nc)`（真实网络场景时，你用 pfss_net_chan 单独跑）

但你现在的 `phase_R.overlap_pfss_open` **只看 `pfss_nc != nullptr`**，导致 bench 明明两个通道独立也被判成不能 overlap。

### 1.2 修复策略

“能不能 overlap”的本质是：**PFSS 的 proto channel 是否在复用同一个 net::Chan**。
在你的工程里，唯一会复用 net::Chan 的典型 wrapper 就是 `runtime::ProtoChanFromNet`（`include/runtime/pfss_superbatch.hpp`）。

因此只要做到：

* 如果 pfss_chan 是 `ProtoChanFromNet`（而且没传 pfss_net_chan），那基本意味着它复用了 open 的 net_chan → **不 overlap**
* 如果 pfss_chan 不是 `ProtoChanFromNet`（bench 下是 CountingChan）→ **允许 overlap**

### 1.3 推荐 patch（最小侵入）

修改 `src/nn/transformer_layer.cpp` 里构造 `PhaseResources` 的那段（你现在已经 include 了 ProtoChanFromNet 的头，所以能 dynamic_cast）：

```cpp
// 原来：phase_R.overlap_pfss_open = (pfss_nc != nullptr);
phase_R.overlap_pfss_open = (pfss_nc != nullptr);

// bench 场景：pfss_ch 是 CountingChan，不走 ProtoChanFromNet，应该允许 overlap
if (!phase_R.overlap_pfss_open) {
  if (dynamic_cast<runtime::ProtoChanFromNet*>(phase_R.pfss_chan) == nullptr) {
    phase_R.overlap_pfss_open = true;
  }
}
```

### 1.4 为什么这会“确实变快”

你的 `PhaseExecutor::run_lazy()` 里已经写了很完整的 overlap 路径（你自己也修过一次 async-open 分支）：

* 当 `want_open && want_pfss && R.overlap_pfss_open` 时：

    * 要么 `flush_open_async()` 然后立刻 `flush_pfss_*()`（并行）
    * 要么开 pfss_thread + 当前线程 flush open（并行）

这段逻辑在你 bench 里**之前被 overlap_pfss_open=false 完全禁用**。
修完之后，`open_comm_ns` 有机会被 PFSS eval 遮住一大块（甚至多数），online 会直接掉一截。

### 1.5 对应 bench 的验证方式

你不用改 benchmark harness，只要：

* 打开 profiling（你已有）
* 跑一遍原版 vs 打 patch 版
* 观察：

    * `open_comm_ns` 不一定变小，但 **online_time 会显著下降**
    * `pfss_flush_eval_eval_ns` 不变或略变
    * `open_flushes`/`pfss_flushes` 次数基本不变（重叠不改变次数，只隐藏时间）

---

## 2) 第二个硬点：让 trunc 的 PackedPred 真正可用（否则你一开就回到 1.2s）

你现在 trunc 的 packed pred 默认是关的（而且 min_qn 默认 32）：

在 `include/gates/composite_fss.hpp`（你 trunc 专用 keygen 这段）：

```cpp
const bool trunc_packed_pred = env_flag_enabled_default_local("SUF_TRUNC_PACKED_PRED", false);
const size_t trunc_packed_pred_min_qn = ... default 32;
```

这会导致 trunc 走 unpacked（每个 query 一把 DCF key）路径；而你之前一旦打开 packed pred，就会因为 **SPC1 的 warp‑per‑element + AoS(stride)** / **小 qn warp 浪费**导致性能崩。

### 2.1 你代码里已经有“正确方向”的基础：SPC3（转置布局）+ SPC1

你 `cuda/pfss_kernels.cu` 里其实已经同时存在：

* `packed_cmp_sigma_dcf_kernel_keyed`（SPC1，按 bit AoS，warp lanes 访问 stride）
* `packed_cmp_sfd2_transposed_kernel_keyed`（SPC3，SFD2-only，按 level 转置，warp coalesced）

并且 `cuda/pfss_backend_gpu.cu` 的 `SecureGpuPfssBackend::gen_packed_lt()` 里也已经能生成 SPC3（你源码里就有 `magic "SPC3"` 的 keygen 分支）。

所以你要做的不是“再造轮子”，而是：

1. **让小 qn 不要走 warp-per-element（浪费 30 lanes）**
2. **让大 qn 不要走 SPC1 AoS（stride 读 key）**，强制走 SPC3

---

## 3) 具体落地：给 SPC1 加一个“thread‑per‑element 标量 kernel”，并在 keygen 里按 qn 选择 SPC1/3

### 3.1 新增 kernel：SPC1 的 scalar 版本（解决小 qn warp 浪费）

文件：`cuda/pfss_kernels.cu`

新增（建议同时提供 keyed + broadcast 两个版本，逻辑几乎一样）：

* `packed_cmp_sigma_dcf_kernel_keyed_scalar(...)`
* `packed_cmp_sigma_dcf_kernel_broadcast_scalar(...)`

核心逻辑：**一个线程处理一个元素，循环 b=0..num_bits-1**，每次用你已有的 `dcf_eval_one_dispatch(...)` 得到 1 bit share，打包进 `out_words`（<=4）。

伪码轮廓（贴近你现有风格）：

```cpp
__global__ void packed_cmp_sigma_dcf_kernel_keyed_scalar(
    const uint8_t* __restrict__ keys_flat, uint32_t key_bytes,
    const uint64_t* __restrict__ in, uint64_t* __restrict__ out, size_t N) {

  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  const uint8_t* keyp = keys_flat + i * key_bytes;
  const auto* hdr = reinterpret_cast<const SecurePackedLtHeader*>(keyp);

  const uint32_t num_bits = hdr->num_bits;
  const uint32_t in_bits  = hdr->in_bits;
  const uint32_t dcf_key_bytes = hdr->dcf_key_bytes;
  const uint32_t out_words = (num_bits + 63u) / 64u;

  uint64_t acc0=0, acc1=0, acc2=0, acc3=0;

  const uint64_t x = in[i];  // dcf_eval_one_dispatch 内部按 in_bits 截断/处理

  for (uint32_t b=0; b<num_bits; ++b) {
    const uint8_t* dcf = keyp + sizeof(SecurePackedLtHeader) + b * dcf_key_bytes;
    uint64_t tmp[1];
    dcf_eval_one_dispatch(dcf, dcf_key_bytes, in_bits, /*out_bits_hint=*/1,
                          /*group_size=*/1, x, tmp);
    if (tmp[0] & 1ull) {
      const uint32_t w = b >> 6;
      const uint32_t sh = b & 63u;
      if (w==0) acc0 |= (1ull<<sh);
      else if (w==1) acc1 |= (1ull<<sh);
      else if (w==2) acc2 |= (1ull<<sh);
      else acc3 |= (1ull<<sh);
    }
  }

  uint64_t* outp = out + i * out_words;
  if (out_words > 0) outp[0]=acc0;
  if (out_words > 1) outp[1]=acc1;
  if (out_words > 2) outp[2]=acc2;
  if (out_words > 3) outp[3]=acc3;
}
```

这个 kernel 的意义非常明确：

* 对 `num_bits` 小（比如 trunc 常见 1~4、或者 2）时：

    * **没有 warp 浪费**
    * 内存访问是线程线性读自己那把 key（不再是 warp lane stride）

> 这正好对症你之前“qn 小导致 packed pred 慢”的问题。

---

### 3.2 secure backend 的 dispatch：SPC1 小 qn -> scalar kernel；SPC3 大 qn -> 转置 warp kernel

文件：`cuda/pfss_backend_gpu.cu`
函数：`SecureGpuPfssBackend::eval_packed_lt_many_device(...)`

你现在 SPC1 的 keyed 走的是：

```cpp
packed_cmp_sigma_dcf_kernel_keyed<<<...>>>(keys_dev, key_bytes, in_dev, out_dev, N);
```

改成：读取 header 的 num_bits，按阈值选择 scalar：

* 新增 env（建议）：

    * `SUF_SECURE_GPU_PACKCMP_SPC1_SCALAR_MAX_BITS`，默认 31（<=31 都走 scalar）

伪码：

```cpp
bool use_scalar_spc1 = false;
if (is_spc1) {
  const auto* hdr = reinterpret_cast<const SecurePackedLtHeader*>(keys[0]);
  const size_t max_bits = env_size_t_default("SUF_SECURE_GPU_PACKCMP_SPC1_SCALAR_MAX_BITS", 31);
  use_scalar_spc1 = (hdr->num_bits <= max_bits);
}

if (is_spc3) {
  packed_cmp_sfd2_transposed_kernel_keyed<<<...>>>(...);
} else if (is_spc1) {
  if (use_scalar_spc1) {
    packed_cmp_sigma_dcf_kernel_keyed_scalar<<<...>>>(...);
  } else {
    packed_cmp_sigma_dcf_kernel_keyed<<<...>>>(...);
  }
} else if (is_dpf) {
  packed_cmp_dpf_prefix_kernel_keyed<<<...>>>(...);
}
```

同理 broadcast 版本也做一份（如果你后面发现 broadcast 的小 qn 也常见，会同样收益）。

---

### 3.3 keygen：按 qn 自动选 SPC1/3（避免 SPC1 AoS 在大 qn 下 stride 读 key）

文件：`cuda/pfss_backend_gpu.cu`
函数：`SecureGpuPfssBackend::gen_packed_lt(...)`

你现在是：

* 默认 use_dpf 只有 thresholds>255 才启
* DCF 模式下：如果 `use_spc3 && use_sfd2` 就直接 SPC3

这里我建议加一个最关键的“门槛”：

* `SUF_SECURE_GPU_PACKCMP_SPC3_MIN_BITS` 默认 32

    * `num_bits >= 32`：强制 SPC3（warp-coalesced）
    * `num_bits < 32`：用 SPC1（配合上面的 scalar kernel）

伪码：

```cpp
const size_t spc3_min_bits =
    env_size_t_default("SUF_SECURE_GPU_PACKCMP_SPC3_MIN_BITS", 32);

const bool want_spc3 = use_spc3 && use_sfd2 && (thresholds.size() >= spc3_min_bits);

if (want_spc3) {
  // 走你已有的 SPC3 keygen 分支
} else {
  // 走 SPC1 keygen 分支
}
```

这样组合起来就是你想要的策略：

* trunc 典型 qn 很小 → **SPC1 + scalar**（不浪费 warp）
* 真的有大 qn（例如某些 packed compare/interval LUT 触发很多阈值）→ **SPC3 + warp**（不 stride）

---

## 4) 打开 trunc packed pred（并把默认 min_qn 改到真正可用）

等上面 3) 落地后，你就可以把 trunc packed pred 变成默认“开且低门槛”。

文件：`include/gates/composite_fss.hpp`（你 trunc keygen 那段）

当前默认：

* `SUF_TRUNC_PACKED_PRED` default false
* `SUF_TRUNC_PACKED_PRED_MIN_QN` default 32

建议默认改成：

* default true
* min_qn default 1（或 2）

```cpp
const bool trunc_packed_pred =
    env_flag_enabled_default_local("SUF_TRUNC_PACKED_PRED", true);

const size_t trunc_packed_pred_min_qn =
    env_u64_default_local("SUF_TRUNC_PACKED_PRED_MIN_QN", 1);
```

理由：

* 你现在 trunc unpacked 会对同一批元素重复做多次 key flatten + H2D（每个 query 一次），而 packed pred 能把它合并成一次（哪怕 qn=2 也能省掉一整趟 staging + launch 开销）
* 有了 SPC1 scalar kernel，小 qn 不再是性能坑

---

## 5) 你现在 bench 的一个“隐藏开关”：bert‑tiny 默认关 device pipeline，建议直接打开验证

在 `src/demo/bench_suf_transformer.cpp` 你已经写了：

* GPU bench 默认 **bert‑tiny 关闭 device pipeline**，其他模型开
* 可以用环境变量强制：`SUF_BENCH_DEVICE_PIPELINE=1`

建议你在做上述修改的同时，把对比实验分成两组：

1. `SUF_BENCH_DEVICE_PIPELINE=0`（纯看 overlap + packed pred 的收益）
2. `SUF_BENCH_DEVICE_PIPELINE=1`（看 D2H/materialize 被消掉后 online 还能再降多少）

因为你现在 TruncTask/CompositeFSS/PhaseExecutor 已经支持 device-only 路径（你也刚修过 device-only hatx crash），bert‑tiny 这个规模（L=128，hidden 维一般不小）通常并不是真的“太小不值得 device pipeline”。

---

## 6) 一套“你可以直接照着跑”的验证矩阵（不用猜）

### 6.1 验证 overlap 修复是否生效

只改 1) 的 patch，先不动 packed pred：

* 观察点：

    * online_time 是否明显下降
    * open_comm_ns 可能不变，但**online_time**会降（被遮住）
    * PhaseExecutor 里 `open_flushes_demand` 仍高，但不再完全串行阻塞

### 6.2 验证 trunc packed pred “从慢变快”

在 3)+4) 都落地后：

* 先用环境变量快速验证（不用改默认）：

    * `SUF_TRUNC_PACKED_PRED=1`
    * `SUF_TRUNC_PACKED_PRED_MIN_QN=1`
    * `SUF_SECURE_GPU_PACKCMP_SPC1_SCALAR_MAX_BITS=31`
    * `SUF_SECURE_GPU_PACKCMP_SPC3_MIN_BITS=32`

预期观察点（方向性）：

* pfss_flush_eval_eval_ns 会明显下降（trunc 的 pred eval 少了重复 staging/launch）
* key_bytes 可能也下降（尤其你走 SPC3 的场景，能少掉一些 per-key header/布局开销）
* 如果你之前开 packed pred 会直接飙到 1.2s，这次应该不会再出现“性能灾难性回退”

### 6.3 再叠加 device pipeline

开启 `SUF_BENCH_DEVICE_PIPELINE=1` 看：

* pfss_flush_eval_copy_ns / materialize 相关项是否明显下降
* online_time 是否进一步下降

---

## 7) 如果你还差一点点：两个“更深但仍然可控”的加速点

### 7.1 让 eval_packed_lt_many_device 识别“keys 已经是 contiguous stride”，绕过 gather

你 secure backend 目前 `eval_packed_lt_many_device` 接的是 `vector<const uint8_t*> keys`，通常会走 gather→pinned→H2D。

可以加一个小优化（对 packed pred 尤其有效）：

* 检测 `keys[i] == keys[0] + i*key_bytes`
* 成立则直接 `stage_keys_flat_ptr_cached(keys[0], n*key_bytes)`，不要循环 memcpy gather

这一步对“你未来如果把 per‑elem packed keys 存成 flat blob”是必备的，而且改动集中在 `cuda/pfss_backend_gpu.cu` 的 staging 层。

### 7.2 把 trunc 的 per‑elem keys 从 `vector<CompositeKeyPair>` 变成“flat arena”

这是结构性优化：减少 allocator/指针追逐/缓存 miss。
但改动范围比前面大（涉及 `compiler::TruncationLoweringResult::per_elems` 里 keys 的存储形式）。你现有的 `TruncPredDeviceKeyCache`/`TruncPackedPredDeviceKeyCache` 已经证明“flat + 一次 upload”的方向是对的，只是目前 cache 对“一次性 keys”帮助不大。

如果你要继续往 Sigma 靠近，这块是长期最值回票价的一点。

---

## 结论：按优先级给你一个最短路径

1. **先修 overlap_pfss_open 判定**（`src/nn/transformer_layer.cpp`）：bench 立刻把 open 与 pfss 并行跑起来，online 直接降一截。
2. **实现 SPC1 scalar kernel + keygen 按 qn 选 SPC1/3**（`cuda/pfss_kernels.cu` + `cuda/pfss_backend_gpu.cu`）：让 trunc packed pred 在小 qn 下也能跑得快。
3. **把 trunc packed pred 默认打开/降低门槛**（`include/gates/composite_fss.hpp`）：让 trunc 不再重复多趟 pred staging/launch。
4. **在 bert‑tiny 上打开 device pipeline 做对比**（`SUF_BENCH_DEVICE_PIPELINE=1`）：进一步削掉 host materialize 开销。

如果你愿意，我也可以按你现在这份代码把上述改动分别整理成 3 个独立 commit 的 diff（每个 commit 对应一个“可回滚的性能实验开关”），方便你逐项跑 bench 验证收益。
