#pragma once

#include "suf/gpu_backend.hpp"
#include "suf/pfss_plan.hpp"
#include "suf/interval_lut.hpp"

#ifdef SUF_HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include <random>
#include <memory>

namespace suf {

#ifdef SUF_HAVE_CUDA

class GpuSecureSufProgram {
public:
  GpuSecureSufProgram(const SUFDescriptor& d, int party, std::uint64_t seed,
                      int in_bits_override = 0,
                      bool mask_aware = false,
                      u64 mask_in = 0);
  ~GpuSecureSufProgram();

  GpuSecureSufProgram(const GpuSecureSufProgram&) = delete;
  GpuSecureSufProgram& operator=(const GpuSecureSufProgram&) = delete;

  void eval(const u64* d_in, std::size_t n,
            u64* d_out_arith, u64* d_out_helpers,
            cudaStream_t stream = nullptr) const;

  std::size_t num_predicates() const { return desc_.predicates.size(); }
  std::size_t num_helpers() const { return desc_.helpers.size(); }

private:
  u8* ensure_pred_bits(std::size_t n) const;
  u8* ensure_query_bits(std::size_t n) const;
  u64* ensure_coeffs(std::size_t n) const;

  SUFDescriptor desc_;
  PfssPlan plan_;
  std::unique_ptr<GpuSufProgram> gpu_prog_; // used for poly + helper eval

  DpfKeyBatchGpu dpf_gpu_{};
  bool dpf_loaded_ = false;
  mutable u8* d_pred_bits_ = nullptr;
  mutable u8* d_query_bits_ = nullptr;
  mutable std::size_t pred_capacity_ = 0;
  mutable std::size_t query_capacity_ = 0;

  bool mask_aware_ = false;
  u64 r_in_ = 0;
  std::vector<int> pred_to_query_;
  std::vector<u8> const_pred_bits_;
  int* d_query_to_pred_ = nullptr;
  u8* d_const_pred_bits_ = nullptr;

  int poly_degree_ = 0;
  bool use_interval_lut_ = false;
  IntervalLutKeyV2Gpu interval_key_{};
  bool use_coeff_lut_ = false;
  IntervalLutKeyV2Gpu coeff_key_{};
  mutable u64* d_coeffs_ = nullptr;
  mutable std::size_t coeff_capacity_ = 0;
};

#endif // SUF_HAVE_CUDA

} // namespace suf
