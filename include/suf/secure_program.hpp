#pragma once

#include "suf/gpu_backend.hpp"
#include "suf/pfss_plan.hpp"
#include "suf/interval_lut.hpp"

#ifdef SUF_HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include <random>

namespace suf {

#ifdef SUF_HAVE_CUDA

class GpuSecureSufProgram {
public:
  GpuSecureSufProgram(const SUFDescriptor& d, int party, std::uint64_t seed);
  ~GpuSecureSufProgram();

  GpuSecureSufProgram(const GpuSecureSufProgram&) = delete;
  GpuSecureSufProgram& operator=(const GpuSecureSufProgram&) = delete;

  void eval(const u64* d_in, std::size_t n,
            u64* d_out_arith, u64* d_out_helpers,
            cudaStream_t stream = nullptr) const;

  std::size_t num_predicates() const { return plan_.queries.size(); }
  std::size_t num_helpers() const { return desc_.helpers.size(); }

private:
  SUFDescriptor desc_;
  PfssPlan plan_;
  GpuSufProgram gpu_prog_; // used for poly + helper eval

  DpfKeyBatchGpu dpf_gpu_{};
  bool dpf_loaded_ = false;

  bool use_interval_lut_ = false;
  IntervalLutKeyV2Gpu interval_key_{};
};

#endif // SUF_HAVE_CUDA

} // namespace suf
