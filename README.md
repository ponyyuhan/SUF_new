# SUF GPU Prototype (with Sigma Integration)

本仓库实现了 paper.tex 中的 SUF（Structured‑Univariate Functions）编译与 GPU 原型，并在 Sigma 的 GPU-MPC 推理中替换 Softmax/LayerNorm 的非线性部分进行端到端对比。

---

## 1. 论文核心创新与当前实现的对应关系

paper.tex 的核心创新点（摘要/贡献部分）与当前代码的对应如下：

1) **Typed SUF IR（类型化 SUF 表达）**  
   - 论文要点：以类型化 IR 表示定点标量非线性，区分算术输出与布尔辅助输出。  
   - 代码对应：`include/suf/ir.hpp`、`include/suf/secure_program.hpp`、`src/secure_program.cpp`。  

2) **Mask‑correct predicate rewriting（掩码正确的谓词重写）**  
   - 论文要点：在公开掩码 $\hat{x}=x+r_{in}$ 上重写比较谓词，wrap/carry 等掩码依赖信息只以秘密常量出现。  
   - 代码对应：`src/masked_compile.cpp`、`include/suf/masked_compile.hpp`。  

3) **Two‑template compilation（最多两次 tFSS 模板调用）**  
   - 论文要点：每个 SUF gate 至多两次 tFSS 实例（谓词+系数/常量选择），其余为标准 Beaver/AND/B2A。  
   - 代码对应：  
     - **谓词/比较模板**：DMPF（GPU 版本）  
       - `src/dmpf.cpp`、`src/cuda/dmpf_kernels.cu`、`include/suf/dmpf.hpp`  
     - **区间/系数选择模板**：Interval LUT  
       - `src/interval_lut.cpp`、`src/cuda/interval_lut_kernels.cu`、`include/suf/interval_lut.hpp`  

4) **Composite‑FSS Gate 复用与批处理**  
   - 论文要点：将编译后的 gate 作为复用单元，便于批处理和统一的安全性论证。  
   - 代码对应：`include/suf/secure_program.hpp` 与 `include/suf/pfss_plan.hpp` 中的 gate/plan 组织与批处理路径。  

**安全泄露面一致性**  
当前实现遵循 paper.tex 的泄露模型：只公开掩码值 $\hat{x}$ 和 gate 形状（例如 LUT 大小、比较数量等）。  
我们确保 key 大小不依赖掩码采样（形状由公开的 SUF 描述与定点参数决定）。  
注意：本实现仍依赖 tFSS（DMPF/Interval LUT）的一键隐私假设，与论文一致；未引入额外泄露。  

---

## 2. 代码结构概览

```
include/suf/         SUF IR、编译与 GPU 接口
src/                SUF 编译与 GPU 实现
src/cuda/           CUDA 核心（DMPF、Interval LUT 等）
scripts/            对比脚本与准确性分析
third_party/EzPC/   Sigma (GPU-MPC) 子模块 + 我们的集成修改
third_party/EzPC_vendor/  Sigma 的“完整拷贝”，用于便携迁移（无子模块依赖）
```

Sigma 集成桥接：
- `include/suf/sigma_bridge.hpp`
- `src/sigma_suf_bridge.cu`
- `third_party/EzPC/GPU-MPC/fss/gpu_softmax.cu`
- `third_party/EzPC/GPU-MPC/fss/gpu_layernorm.cu`
- `third_party/EzPC/GPU-MPC/backend/sigma.h`

可移植版本（不依赖子模块）：
- `third_party/EzPC_vendor/GPU-MPC/...`

---

## 3. 构建

```
CUDA_VERSION=12.9 GPU_ARCH=120 make -C third_party/EzPC/GPU-MPC sigma
```

---

## 4. 运行与参数

Sigma 标准运行方式（两方 P0/P1）：
```
cd third_party/EzPC/GPU-MPC/experiments/sigma
./sigma gpt2 128 0 <peer_ip> 64
./sigma gpt2 128 1 <peer_ip> 64
```

### SUF 相关环境变量

核心开关：
- `SUF_NONLINEAR=1`：用 SUF 替换 Softmax/LayerNorm 的非线性

关键参数：
- `SUF_NEXP_XMAX`：nExp 输入最大值（默认 16）
- `SUF_INV_FRAC`：inv 输入小数位（默认 6）
- `SUF_RSQRT_VMAX`：rsqrt 输入最大值（默认 16）
- `SUF_NEXP_BITS` / `SUF_INV_BITS`：直接 LUT 的输入 bit 上限（默认自动）
- `SUF_RSQRT_FRAC`：rsqrt 输出小数位（默认 6）

推荐参数（在 gpt2‑128/256 上验证正确且更快）：
```
SUF_NONLINEAR=1 SUF_NEXP_XMAX=16 SUF_INV_FRAC=6 SUF_RSQRT_VMAX=8 SUF_NEXP_BITS=12
```

Sigma 运行时调节（可选）：
- `SIGMA_KEYBUF_GB` / `SIGMA_KEYBUF_MB`：CPU 侧 key 缓冲
- `SIGMA_MEMPOOL_GB` / `SIGMA_MEMPOOL_MB`：GPU mempool 预留（影响显存占用）
- `SIGMA_DUMP_OUTPUT=1`：导出 `output.bin` + `output_meta.json`
- `SIGMA_RANDOM_WEIGHTS=1`：随机权重（否则默认全 0）

---

## 5. Bench（端到端）

**基准对齐方式**  
- P0/P1 同机、同模型、同序列长度、同线程数  
- 仅切换 `SUF_NONLINEAR` 与 SUF 参数  
- 统计 `dealer.txt`（gen/keysize）与 `evaluator.txt`（online time / comm）

### 已完成结果（gpt2）

**gpt2‑128**
- Baseline：
  - gen 9.34s，key 14.29 GiB
  - eval 48.66s，comm 0.824 GiB
- SUF（推荐参数）：
  - gen 2.98s，key 13.23 GiB
  - eval 3.17s，comm 0.751 GiB
- 提升：
  - gen 3.14×
  - eval 15.34×
  - key ‑7.45%
  - comm ‑8.88%

**gpt2‑256**
- Baseline：
  - gen 20.86s，key 33.19 GiB
  - eval 52.41s，comm 1.983 GiB
- SUF（推荐参数）：
  - gen 5.80s，key 28.96 GiB
  - eval 6.20s，comm 1.688 GiB
- 提升：
  - gen 3.60×
  - eval 8.46×
  - key ‑12.74%
  - comm ‑14.89%

### 未完成/受限

**gpt2‑512、bert‑base‑256/512** 在单张 RTX 5060 Ti 上无法稳定完成（P1 被 kill 或 cudaMemcpy invalid）。  
建议使用双 GPU 或两台机器对齐基准。

---

## 6. 准确性曲线（速度‑准确性）

导出输出并计算精度差异：
```
SIGMA_RANDOM_WEIGHTS=1 SIGMA_DUMP_OUTPUT=1 ./sigma gpt2 128 0 ...
SIGMA_RANDOM_WEIGHTS=1 SIGMA_DUMP_OUTPUT=1 ./sigma gpt2 128 1 ...

python scripts/sigma_accuracy.py \
  --base-dir /tmp/sigma_baseline_dump \
  --sweep-dir /tmp/suf_sweep_acc \
  --out-csv /tmp/suf_sweep_acc/accuracy_curve.csv
```

目前 sweep 的结果 **与原 Sigma 输出完全一致**（MAE/RMSE/MaxAbs=0，Top‑1=1.0）。  
- 结果文件：`/tmp/suf_sweep_acc/accuracy_curve.csv`  
- 进一步 LUT/截断优化结果：`/tmp/suf_lut_opt2/accuracy_curve.csv`

---

## 7. Sigma 集成逻辑（Softmax/LayerNorm）

Softmax：
1) `max` 与 `sum` 仍由 Sigma 的 GPU 协议处理（非 SUF 部分）。  
2) nExp / inverse 使用 SUF gate（DMPF + Interval LUT）替换。  

LayerNorm：
1) `mean/var` 仍沿用 Sigma 逻辑。  
2) rsqrt 使用 SUF gate（DMPF + Interval LUT）替换。  

这与 paper.tex 中“SUF 只覆盖标量非线性，向量归约仍由标准 MPC 子协议实现”的定位一致。

---

## 8. 注意事项

1) `SIGMA_CLEAR_REF` 不可用：ClearText backend 缺 MHA，已在代码中直接跳过并提示。  
2) 大模型/长序列对显存要求高，单卡可能失败。  
3) 若需更明显的精度‑速度折中曲线，可继续调低 `SUF_NEXP_BITS` 或 `SUF_RSQRT_FRAC`。
