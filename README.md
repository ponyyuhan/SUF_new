# SUF GPU Prototype (Sigma Integration + Secure LUT)

本仓库实现 paper.tex 中的 **SUF（Structured‑Univariate Functions）编译与 GPU 原型**，并接入 **Sigma GPU‑MPC**，用于替换 Softmax / LayerNorm / 激活函数中的标量非线性。

当前代码已按 `security.md` 的语义修复：
- **每个 wire 独立 mask**（wire‑level masking）
- **LUT 仅保存一份公共表**（SIL2/DPF‑LUT），每个元素独立 key
- **没有“整张 tensor 共享 r”** 的不安全路径

---

## 0. TL;DR

- **SUF 编译器**：Typed IR + mask‑correct predicate rewriting + two‑template（谓词 + LUT）。
- **GPU 后端**：DMPF/DCF + Interval LUT（SIL2 v2），小域可直接表。
- **Sigma 集成**：Softmax / LayerNorm / GELU / SiLU 的非线性改用 **DPF‑LUT 公表 + per‑wire mask**。
- **评测结果**：见 `evaluation_report.md`（2026‑01‑20）。

---

## 1. 仓库结构

```
include/suf/             SUF IR / 编译器 / GPU 接口
src/                     SUF 编译 + GPU 实现（DMPF / Interval LUT / Masked compile）
third_party/EzPC/        Sigma (GPU‑MPC) + SUF 集成补丁（vendored）
third_party/EzPC_vendor/ Sigma 完整 vendored 版本（便于迁移/镜像）
ezpc_upstream/           上游干净版本（baseline 对比用）
shaft/                   SHAFT baseline（CrypTen）
scripts/                 对比/准确性脚本
paper.tex / security.md  论文与安全语义说明
```

---

## 2. 构建

### 2.1 构建 SUF core（库 + 测试 + bench）

```bash
cmake -S . -B build -DSUF_ENABLE_CUDA=ON -DSUF_ENABLE_TESTS=ON
cmake --build build -j
ctest --test-dir build
```

如需指定 GPU 架构（例如 Blackwell）：

```bash
cmake -S . -B build -DSUF_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j
```

### 2.2 构建 Sigma (SUF 版本)

```bash
CUDA_VERSION=13.0 GPU_ARCH=120 make -C third_party/EzPC/GPU-MPC sigma
```

### 2.3 构建 Sigma baseline（无 SUF）

```bash
CUDA_VERSION=13.0 GPU_ARCH=120 make -C ezpc_upstream/GPU-MPC sigma
```

---

## 3. 运行（Sigma + SUF）

Sigma 两方运行（P0/P1）：

```bash
cd third_party/EzPC/GPU-MPC/experiments/sigma
SUF_NONLINEAR=1 ./sigma gpt2 128 0 <peer_ip> 64
SUF_NONLINEAR=1 ./sigma gpt2 128 1 <peer_ip> 64
```

更细粒度控制（只替换部分模块）：

```bash
SUF_SOFTMAX=1 SUF_LAYERNORM=1 SUF_ACTIVATION=1 ./sigma bert_base 128 0 <peer_ip> 64
```

### 常用环境变量（Sigma 侧）

- `SIGMA_RANDOM_WEIGHTS=1`：使用随机权重（若当前 build 不支持则回退为 0）
- `SIGMA_DUMP_OUTPUT=1`：导出 `output.bin` 与 `output_meta.json`
- `SIGMA_CLEAR_REF=1`：尝试 ClearText 参考（当前 build 对 MHA 不支持）
- `SIGMA_PINNED_KEYBUF=1`：使用 pinned key buffer

---

## 4. SUF 参数表（LUT 规模 / 精度）

| 参数 | 作用 | 默认值 |
|---|---|---|
| `SUF_GELU_INTERVALS` | GELU 区间数 | 256 |
| `SUF_GELU_BITS` | GELU LUT 输入 bit 数 | log2(intervals) |
| `SUF_SILU_INTERVALS` | SiLU 区间数 | 1024 |
| `SUF_SILU_BITS` | SiLU LUT 输入 bit 数 | log2(intervals) |
| `SUF_NEXP_XMAX` | nExp clamp 上界 | 16 |
| `SUF_NEXP_BITS` | nExp LUT 输入 bit 数（≤16） | 自动 |
| `SUF_INV_FRAC` | inv 输入小数位 | 6 |
| `SUF_INV_BITS` | inv LUT 输入 bit 数（≤16） | 自动 |
| `SUF_RSQRT_FRAC` | rsqrt 输出小数位 | 6 |
| `SUF_RSQRT_VMAX` | rsqrt clamp 上界 | 16 |
| `SUF_RSQRT_EPS` | rsqrt clamp 下界 | 0 |
| `SUF_RSQRT_BITS` | rsqrt LUT 输入 bit 数（≤16） | 自动 |
| `SUF_DIRECT_LUT_BITS` | Interval LUT 直接表阈值 | 16 |
| `SUF_DEBUG=1` | 打印 SUF keygen/eval 日志 | 关 |

> 注意：Sigma LUT 输入使用 `uint16`，`*_BITS` 最终会限制在 ≤16。

---

## 5. 设计与实现（与 paper.tex 对齐）

### 5.1 编译链路概览

1. **Typed SUF IR**：描述标量非线性（piecewise polynomial）。
2. **Mask‑correct predicate rewriting**：将谓词重写为基于公开掩码值的比较式。
3. **Two‑template compilation**：
   - 谓词/比较模板：DCF‑LT / DMPF
   - 区间/系数选择模板：Interval LUT
4. **GPU 批处理执行**：predicate batch + LUT batch + polynomial eval。

### 5.2 DMPF + Interval LUT (SIL2 v2)

- **Direct Table**（小域）：`in_bits <= SUF_DIRECT_LUT_BITS` 时直接查表
- **DMPF‑suffix**（通用）：`base + sum(delta * 1[x<cut])`

文件对应：
- `include/suf/dmpf.hpp` / `src/cuda/dmpf_kernels.cu`
- `include/suf/interval_lut.hpp` / `src/interval_lut.cpp`

### 5.3 Mask‑aware 编译（SUF core）

`GpuSecureSufProgram` 支持 `mask_aware=true`：
- 对谓词进行 mask‑correct 重写
- 多项式系数做移位补偿

对应文件：
- `src/masked_compile.cpp`
- `src/secure_program.cpp`

---

## 6. 安全语义（与 Sigma 对齐）

Sigma 语义要求：**每个 wire 独立 mask**，并在 `(x_i + r_i)` 上进行 LUT/比较。
本仓库已按 `security.md` 修复：

- **per‑wire input mask**：keygen 接收 `d_input_mask`（每个元素独立 mask）
- **per‑wire DPF keys**：每个元素都有自己的 LUT key
- **公共 LUT 表**：表只保存一份（SIL2/DPF‑LUT），不会生成 per‑wire shifted table

这样避免了“共享 r 导致张量内差分泄露”的问题。

---

## 7. Sigma 集成细节

### 7.1 入口函数

- Keygen：`suf_sigma_keygen_*`
- Eval：`suf_sigma_eval_*`

对应文件：
- `src/sigma_suf_bridge.cu`
- `third_party/EzPC/GPU-MPC/backend/sigma.h`

### 7.2 Softmax / LayerNorm / Activation

- **Softmax**：max/sum 仍走 Sigma；nExp + inv 走 SUF LUT
- **LayerNorm**：mean/var 仍走 Sigma；rsqrt 走 SUF LUT
- **GELU / SiLU**：直接走 SUF LUT

---

## 8. SUF Bench（本仓库）

### 8.1 bench_suf_gpu

```bash
./build/bench_suf_gpu --n 1048576 --iters 50 --intervals 16 --degree 3 --helpers 4
./build/bench_suf_gpu --secure --mask-aware --mask 12345 --verify
```

### 8.2 bench_suf_model（按模型规模估算 gate 开销）

```bash
./build/bench_suf_model --model bert-base --seq 128 --iters 20
./build/bench_suf_model --model llama13b --seq 128 --json
```

输出包含 key size / keygen time / per‑gate eval latency。

---

## 9. 评测结果（2026‑01‑20）

环境与完整结果见 `evaluation_report.md`。关键指标如下：

**硬件**：2× RTX PRO 6000 Blackwell (sm_120), CUDA 13.0

**SUF 设置**：
```
SUF_SOFTMAX=1 SUF_LAYERNORM=1 SUF_ACTIVATION=1
SUF_NEXP_BITS=10 SUF_INV_BITS=10 SUF_RSQRT_BITS=9
```

### 9.1 End‑to‑end（Sigma vs SUF）

| Model | Sigma online (ms) | SUF online (ms) | Speedup | Sigma comm (GB) | SUF comm (GB) | Sigma keygen (s) | SUF keygen (s) | Sigma key (GB) | SUF key (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BERT‑base‑128 | 1017.17 | 722.54 | 1.41× | 1.062 | 0.891 | 7.03 | 0.48 | 18.076 | 13.678 |
| BERT‑large‑128 | 2785.84 | 1805.22 | 1.54× | 2.833 | 2.376 | 13.48 | 1.21 | 48.800 | 37.076 |
| GPT‑2‑128 | 801.38 | 699.68 | 1.15× | 0.885 | 0.778 | 7.15 | 0.44 | 15.346 | 11.920 |

### 9.2 Seq 长度扩展（BERT‑base）

| Seq | Sigma time (ms) | SUF time (ms) | Speedup | Sigma comm (GB) | SUF comm (GB) |
|---:|---:|---:|---:|---:|---:|
| 32 | 391.24 | 298.62 | 1.31× | 0.198 | 0.180 |
| 64 | 579.78 | 433.68 | 1.34× | 0.442 | 0.388 |
| 128 | 1017.17 | 722.54 | 1.41× | 1.062 | 0.891 |

> Seq=256 在当前环境下触发 `cudaMemcpy invalid argument`，详见报告。

---

## 10. 准确性

- `accuracy_sweep/bert_tiny_accuracy.csv`：MAE / RMSE / MaxAbs = 0（当前 sweep 设置下与 Sigma 完全一致）。

---

## 11. 已知问题

- Seq=256 在 Sigma 与 SUF 端均出现 `cudaMemcpy invalid argument`（见 `evaluation_report.md`）。
- SHAFT 的 GPT‑2 e2e 在当前 PyTorch nightly 环境下失败（CrypTen op 兼容问题）。

---

## 12. 相关文档

- `paper.tex`：论文主体与算法描述
- `security.md`：Sigma‑level 安全语义修复方案与理由
- `evaluation_report.md`：完整评测结果（2026‑01‑20）
- `DMPF.md` / `intervalLUT.md`：DMPF 与 Interval LUT 细节
- `Evaluation.md`：评测计划与指标设计

---

如需我继续补充（例如：复现实验脚本、绘图、或更详细的安全证明映射），告诉我即可。
