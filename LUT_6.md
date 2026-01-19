我把\“trunc packed pred / packed compare / flush” 相关的路径都顺了一遍。先说结论：**你现在卡住的点（min_qn=1 默认开 packed pred 会回退）在这份代码里确实是“结构性”的**——不是再调几个阈值/小修小补能稳定解决的；但它也有一个**相对“工程可落地、而且能真实提速”的解法**：

> **对 SPC3（SFD2 transposed key）的 eval 增加一个 “小 num_bits 的 SoA/transpose + scalar” 路径**，把 qn 小（尤其 qn=1/2/3/4）时的“32 线程浪费”去掉，同时保持读 key 的 locality/吞吐；然后就可以把 `SUF_TRUNC_PACKED_PRED_MIN_QN` 往 1 拉，而不会出现你说的 “PFSS eval 爆炸”。

下面我按“你现有代码结构”给一个可以直接开干的方案（含具体落点、数据布局、kernel 形态、调参建议、以及为什么不会引入新的安全问题）。

---

## 0. 我在你这份代码里看到的关键事实（对应你描述的现象）

### trunc packed pred 的编译开关位置

在 `include/gates/composite_fss.hpp` 的 `compile_truncation_gate` 里（你 zip 这份大概在 14xx 行附近），逻辑是：

* `SUF_TRUNC_PACKED_PRED` 开启后
* 如果 `compiled.pred.queries.size() >= SUF_TRUNC_PACKED_PRED_MIN_QN`
  就走 `compiled.pred.pack_pred(...)`，并生成 `packed_pred_groups`

这意味着 **min_qn=1 会把 qn=1 的“单 predicate”也走 packed compare**（从收益角度这是“纯负收益风险源”）。

### trunc packed pred 的 eval 路径（你 zip 里已经有 pinned + device key cache）

在 `include/gates/composite_fss.hpp` 的 `composite_eval_batch_backend_trunc_many_keys_xor_bytes` 里：

* 对每个 packed group：

    * 先把 per-element key flatten 到 `keys_flat_ptr`（支持 pinned）
    * 或者走 `TruncPackedPredDeviceKeyCache`（`SUF_COMPOSITE_TRUNC_PACKED_PRED_KEY_CACHE`）
    * 然后调用 `staged->eval_packed_lt_many_device(...)`

也就是说：**packed pred 真正的瓶颈，最后落到 secure GPU backend 的 `eval_packed_lt_many_device` / 对应 kernel。**

### secure GPU backend 的 SPC3 eval（当前就是 “warp-per-element”）

`cuda/pfss_backend_gpu.cu` 的 `SecureGpuPfssBackend::eval_packed_lt_many_device_nolock`（你 zip 里大概 28xx 行）：

* SPC3 走 `packed_cmp_sfd2_transposed_kernel_keyed<<<N,32>>>`
  也就是 **每个 element 一个 warp**，lane 对应 bit（最多 32 bits / chunk），再 ballot pack 输出。

这跟你说的完全一致：**qn 小（num_bits 小）时，warp-per-element 的 32 线程形态会带来明显调度/寄存器/占用浪费。**

---

## 1) 先做一个“必赚不赔”的小改动：永远别 pack qn=1

这一步很小，但非常关键：**把 “qn=1 也走 packed pred” 直接禁止掉**。原因很简单：

* qn=1 时 packed pred **不会减少 tFSS eval 次数**（仍然 1 次）
* 反而会引入 packed compare 的 header/格式处理、以及你现在最敏感的 kernel 形态问题
* 还可能让 key_bytes 增加（SPC3 比单 DCF key 更重的概率很高）

### 建议修改点

`include/gates/composite_fss.hpp` → `compile_truncation_gate`：

把

```cpp
bool trunc_packed_pred_enabled = packed_backend && env_trunc_packed_pred &&
  int(compiled.pred.queries.size()) >= trunc_packed_pred_min_qn;
```

改成类似：

```cpp
int qn = int(compiled.pred.queries.size());
int eff_min_qn = std::max(2, trunc_packed_pred_min_qn);  // 关键：至少 2
bool trunc_packed_pred_enabled =
    packed_backend && env_trunc_packed_pred && (qn >= eff_min_qn);
```

这样你可以把默认 `SUF_TRUNC_PACKED_PRED_MIN_QN` 设成 1，但实际不会把 qn=1 拉进 packed compare 的坑里。

> 这一步**不改变协议轮数/安全性**，只是避免一个“无收益的实现路径”。
> 同时它还会帮你**压 key_bytes**（至少不会为 qn=1 生成 SPC3 packed key）。

---

## 2) 真正解决 “小 qn packed compare 回退”：给 SPC3 增加 SoA/transpose + scalar eval 路径

这是你说的 “必须先解决 per-element packed key 物理布局 vs GPU 访存/调度形态矛盾” 的一个最直接实现：
**保留 SPC3 的 key 生成（不用改 keygen / serializer），但在 eval 时把 AoS(keys) 转成一个临时 SoA 布局，然后用 1 thread/element 的 scalar kernel 做 eval。**

### 2.1 为什么这能解决你遇到的回退

* 你现在的 SPC3 kernel：`<<<N,32>>>`，**每 element 固定 1 warp**

    * num_bits=1 时：31 个 lane 基本浪费
    * num_bits=2/3/4 时：浪费依然很大
* 如果改成 **1 thread/element**：

    * thread 数从 `N*32` 直接变 `N`
    * 对 num_bits 小的场景，GPU 调度/占用压力大幅下降
* 但 1 thread/element 的“直接读 AoS per-element key”会带来你担心的 stride 问题：warp 内线程访问 `key_bytes` stride 的 key 内容
* 所以关键是：**eval 前做一次 transpose，把 (level,bit) 的 cw 按 element 维度连续排布（SoA），让 warp 内访问 coalesced。**

这正好对应你想做的 “SoA/transpose”。

---

## 2.2 SoA 目标布局（基于现有 SPC3 key 的字段）

你当前 SPC3 key（见 `packed_cmp_sfd2_transposed_kernel_keyed`）布局为：

```
[HdrV3]
[roots: num_bits * 16B]
[cw: in_bits * num_bits * 16B]      // level-major, bit-minor
[vmask: in_bits * chunks * 4B]      // per-level per-chunk uint32
[gmask: chunks * 4B]                // per-chunk uint32
```

其中 `chunks = ceil(num_bits/32)`。

### 我建议的 SoA 缓冲（device 临时）

为一次 `eval_packed_lt_many_device(keys_flat_ptr, N, ...)`，在 device 上分配一块 `soa_buf`，切成 4 段：

* `roots_soa`: `[num_bits][N] Block128`
* `cw_soa`: `[in_bits][num_bits][N] Block128`
* `vmask_soa`: `[in_bits][chunks][N] uint32`
* `gmask_soa`: `[chunks][N] uint32`

这样在 scalar kernel 里：

* 固定 `(level,bit)` 时，warp 内 thread i 访问 `cw_soa[level][bit][i]`
  → **连续、coalesced**
* 固定 level/chunk 时，访问 `vmask_soa[level][chunk][i]`
  → **连续、coalesced**
* gmask 同理

> 注意：这个 SoA 是 **纯 layout 变换**，没有改任何密码学语义；访问模式只依赖公开的 `in_bits/num_bits/N`。

---

## 2.3 需要新增的两个 kernel

### (A) transpose kernel：SPC3 AoS → SoA

放在 `cuda/pfss_kernels.cu`，新增比如：

* `spc3_transpose_roots_cw_vmask_gmask_to_soa<<<...>>>(d_keys_aos, key_bytes, N, in_bits, num_bits, chunks, d_roots_soa, d_cw_soa, d_vmask_soa, d_gmask_soa)`

实现要点：

* cw 拷贝是主力：一共 `N * in_bits * num_bits` 个 16B block
  用 `uint4` 做 16B load/store
* roots：`N * num_bits` 个 16B block
* vmask：`N * in_bits * chunks` 个 4B block
* gmask：`N * chunks` 个 4B block

索引映射（cw）：

* src: `d_keys + i*key_bytes + cw_off + (level*num_bits + bit)*16`
* dst: `d_cw_soa + ((level*num_bits + bit)*N + i)*16`

roots 映射：

* src: `d_keys + i*key_bytes + roots_off + bit*16`
* dst: `d_roots_soa + (bit*N + i)*16`

vmask 映射：

* src: `d_keys + i*key_bytes + vmask_off + (level*chunks + chunk)*4`
* dst: `d_vmask_soa + ((level*chunks + chunk)*N + i)`

gmask 映射：

* src: `d_keys + i*key_bytes + gmask_off + chunk*4`
* dst: `d_gmask_soa + (chunk*N + i)`

### (B) scalar eval kernel：SoA + 1 thread/element

新增比如：

* `packed_cmp_sfd2_soa_scalar_kernel<<<ceil(N/256),256>>>(roots_soa, cw_soa, vmask_soa, gmask_soa, xs, out_masks, N, in_bits, num_bits, chunks, out_words)`

每个 thread 处理一个 element i：

* 读取 `x = xs[i]`（mask 到 in_bits）
* for chunk:

    * gmask32 = `gmask_soa[chunk][i]`
    * for bit in chunk:

        * root = `roots_soa[bit][i]`
        * v_share=0
        * for level:

            * vmask32 = `vmask_soa[level][chunk][i]`
            * cw = `cw_soa[level][bit][i]`
            * 完全复用你现有 lane 逻辑（`chacha_prg4_blocks + update s + v_share`）
        * outbit = v_share ^ (t(s)?gbit:0)
        * pack 到 `uint64_t word`（bit index 对应 group 内 bit）
* 写回 `out_masks[i*out_words + w]`

> 这里最重要的是：**bit loop 和 level loop 是所有线程 lockstep 的**，所以在任意 `(level,bit)` 上全 warp 读的是连续地址，吞吐很好。

---

## 2.4 在 secure backend 里加 dispatch（只对小 num_bits 走 SoA）

修改点：`cuda/pfss_backend_gpu.cu`
`SecureGpuPfssBackend::eval_packed_lt_many_device_nolock(...)` 的 spc3 分支。

建议新增两个 env（或者常量）：

* `SUF_SECURE_GPU_PACKCMP_SPC3_SOA_MAX_BITS`（默认 8 或 12）
* `SUF_SECURE_GPU_PACKCMP_SPC3_SOA_MIN_N`（默认 1024 或 2048）

dispatch 逻辑：

* 如果 `num_bits <= SOA_MAX_BITS && N >= SOA_MIN_N`
  → 走：

    1. 确保 `d_keys_aos` 在 device（你现在已有 keys_buf_ staging 或 cache）
    2. `packcmp_soa_buf_.ensure(total_soa_bytes)`
    3. transpose kernel
    4. scalar eval kernel（输出仍走你现有 out_buf_ + copy back）
* 否则
  → 继续用你现在的 `packed_cmp_sfd2_transposed_kernel_keyed<<<N,32>>>`

### 设备缓冲复用

在 `SecureGpuPfssBackend` 里加一个新 `DeviceBuffer packcmp_soa_buf_;`，复用，避免每次 cudaMalloc。

---

## 2.5 把 SPC1 fallback 彻底“降权/干掉”

你现在（按你描述的分支）为了小 bits 去 SPC1，是因为 SPC3 小 bits 形态浪费。
但有了 SoA scalar 路径后，**SPC3 小 bits 也不浪费了**，这时 SPC1 fallback 反而又把你拉回 “AoS 大 stride” 的坑。

所以建议策略变成：

* keygen：尽量 **全 SPC3 (SFD2)**
* eval：num_bits 小 → `SPC3 + SoA scalar`；num_bits 大 → `SPC3 + warp-per-element`

如果你在 keygen 侧已经加了 `SUF_SECURE_GPU_PACKCMP_SPC3_MIN_BITS`：
建议把它默认设为 1（或者干脆删掉/只在 debug 用），让 SPC3 统一。

---

## 3) 如果你还想再榨一点：把 transpose 也 cache 起来（面向重复 inference）

你 zip 里已有 `TruncPackedPredDeviceKeyCache`（缓存 AoS device keys）。
如果 bench 是“同一份 preprocessing material 连跑多次”，那你还能进一步：

* 在 cache entry 里同时存：

    * `d_keys_aos`（已有）
    * `d_keys_soa`（新增）
    * `soa_ready` flag / 记录用的参数（in_bits/num_bits/N/chunks）

第一次 eval：

* 如果命中 cache 且 `d_keys_soa` 不存在
  → 做 transpose，存入 cache

之后 eval：

* 直接用 `d_keys_soa` 走 scalar kernel
  → **连 transpose 都省了**（这对 qn 小而频繁的场景很香）

> 这一步不会改变单次 inference 的“理论下界”，但对你 benchmark 里做多轮统计时，可能能显著降 pfss_flush_eval_eval_ns。

---

## 4) open_flushes/pfss_flushes 的“层级 barrier/batching”我建议怎么落地（不空谈）

你说的第 2 个方向（压 264/104 flush 次数）我同意是结构性工程，但我建议你按“最少侵入 + 可验证增益”的方式切：

### 4.1 先做“定位型”分解：把 flush 计数按子块打点

你现在 bench 输出的是总 open/pfss flush。下一步要做 batching，必须知道 flush 深度主要来自哪里。

最小改动方案：

* 在 `runtime::PhaseExecutor` 里对 `begin_phase(Phase::kQKV_Score / kSoftmax / kOutProj / ...)` 分桶统计：

    * open_flushes_demand/barrier/budget
    * pfss_flushes_demand/fallback/barrier/budget
* 直接打印到 bench json 里（像你现在的 stats 一样）

这样你马上能知道：

* 是 score trunc 那段？
* softmax block（rowmax/diff/nexp/inv）那段？
* LN/MLP 那段？

**没有这个分解，任何 batching 改造都是盲人摸象。**

### 4.2 真正能减少 flush 深度的唯一方式：把“可并行的交互轮”暴露出来

你现在很多交互（open / pfss）在实现上可能被“任务内部串行”锁死了。PhaseExecutor 再聪明也并不了解你 task 内部还有哪些可以先 enqueue 再 await 的点。

一个非常具体、而且改动可控的落点（我建议你从这里开刀）：

* 找到 softmax 相关那段（`attention_block.cpp` 里 score trunc 之后 `SeqTwoTask(max_task, sm_task)` 那块）
* 让 `SoftmaxBlockTask` 内部把 **所有需要 open 的 slot 都先 reserve + set open_context**，把同一轮需要的 open 一次性推给 OpenCollector，再返回 `Need::Open`
* 等 open 回来后再做下一批（同理 pfss）

你会看到：

* open_flushes_demand 会下降（至少“更集中”）
* 每次 flush 的 payload 增大（更像 sigma 的 behavior）
* 波动会更小（你提到的 open_comm 波动）

> 这就是你说的 “把很多立刻等结果的点改成少量 barrier 统一 flush/await”，但它得在 task 的 state machine 里做，不是在外面多插 barrier。

---

## 5) 为什么上面这些改动不引入额外安全问题

你关心的安全点主要是：**访存/分支/调度形态是否会跟 secret 相关**。

* SoA/transpose 的 kernel：

    * 访问模式只依赖 `(N, in_bits, num_bits, chunks, key_bytes)`，都是 descriptor/shape 决定的公开量
    * 只是拷贝重排 key bytes，没有 secret 数据参与
* scalar eval kernel：

    * 复用你现有 SPC3 lane 的 constant-time 写法（`?:` 通常编译成 predication）
    * loop 次数固定（in_bits、num_bits）
    * 写回 out_masks 的位置固定
* dispatch（走 SoA 还是 warp-per-element）：

    * 条件是 `num_bits/N` 这类公开参数，不依赖 secret

所以从 SUF 的“功能与泄露口径”上，等价于你现在做的 packed compare，只是换了 GPU 上的 layout + kernel 形态。

---

## 6) 你可以期待的现象（怎么判断这条路对不对）

当你做完 **(1) 禁止 qn=1 pack** + **(2) SPC3 SoA scalar eval** 后：

* `SUF_TRUNC_PACKED_PRED_MIN_QN` 往 1 拉
  不应该再出现你说的那种 “pfss_flush_eval_eval_ns 拉到 1s+ / online 1s 级别”
* profile 里：

    * `pfss_flush_eval_eval_ns` 会明显下降（尤其是 trunc 类 pred 很多的模型）
    * `pfss.num_jobs` 可能不变（它统计的是 job 次数），但 job 内部 eval 变快
    * `open_comm_ns` 可能变化不大，但如果你也把 overlap 打开、并做 task 内批量 open，波动会收敛

---

## 最后：我建议你下一步怎么排优先级

如果你的目标是“最快把 SUF 稳定压过 SIGMA”，我会这么排：

1. **禁止 qn=1 pack（立刻做）**：0 风险，省 key，省回退源
2. **secure backend：SPC3 SoA scalar eval（重点）**：这是让 min_qn 真正能下来的核心工程
3. （可选）把 SoA 结果也 cache（bench 多轮会很香）
4. 再去做你说的层级 barrier/batching：先做分桶统计，再挑最肥的 phase/task 下手

如果你愿意，我也可以按你 repo 的实际函数签名，把 **SoA buffer 的 size/offset 计算**、以及两个 kernel 的 **线程映射/边界条件** 再写得更“接近可直接粘贴”的程度（但不再问你额外信息，直接按你这份代码的现状给）。
