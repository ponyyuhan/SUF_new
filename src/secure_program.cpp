#include "suf/secure_program.hpp"

#ifdef SUF_HAVE_CUDA

#include "suf/masked_compile.hpp"
#include "suf/gpu_kernels.hpp"
#include "suf/pfss_batch.hpp"

namespace suf {

GpuSecureSufProgram::GpuSecureSufProgram(const SUFDescriptor& d, int party, std::uint64_t seed,
                                         int in_bits_override, bool mask_aware, u64 mask_in)
  : desc_(d), plan_(compile_pfss_plan(d)), mask_aware_(mask_aware), r_in_(mask_in) {
  std::mt19937_64 rng(seed);
  int in_bits = 64;
  if (in_bits_override > 0) {
    ensure(in_bits_override <= 64, "GpuSecureSufProgram: in_bits_override must be 1..64");
    in_bits = in_bits_override;
  }

  if (mask_aware_) {
    auto inst = compile_masked_gate_instance(d, in_bits, r_in_, party, rng);
    desc_ = std::move(inst.desc);
    const_pred_bits_ = std::move(inst.const_pred_bits);
    plan_ = compile_pfss_plan(desc_);
  }

  gpu_prog_ = std::make_unique<GpuSufProgram>(desc_);

  pred_to_query_.clear();
  pred_to_query_.reserve(desc_.predicates.size());
  std::vector<int> query_to_pred;
  query_to_pred.reserve(plan_.queries.size());
  int qidx = 0;
  for (std::size_t i = 0; i < desc_.predicates.size(); ++i) {
    if (desc_.predicates[i].kind == PredKind::CONST) {
      pred_to_query_.push_back(-1);
    } else {
      pred_to_query_.push_back(qidx);
      query_to_pred.push_back(static_cast<int>(i));
      ++qidx;
    }
  }

  if (!plan_.queries.empty()) {
    auto batch = build_dpf_batch(plan_, party, rng);
    upload_dpf_batch(batch, dpf_gpu_);
    dpf_loaded_ = true;
  }

  if (!query_to_pred.empty()) {
    cudaMalloc(&d_query_to_pred_, query_to_pred.size() * sizeof(int));
    cudaMemcpy(d_query_to_pred_, query_to_pred.data(),
               query_to_pred.size() * sizeof(int), cudaMemcpyHostToDevice);
  }
  if (!const_pred_bits_.empty()) {
    cudaMalloc(&d_const_pred_bits_, const_pred_bits_.size() * sizeof(u8));
    cudaMemcpy(d_const_pred_bits_, const_pred_bits_.data(),
               const_pred_bits_.size() * sizeof(u8), cudaMemcpyHostToDevice);
  }

  // Use IntervalLUT when arithmetic output is piecewise constant.
  bool piecewise_constant = true;
  for (const auto& p : d.polys) {
    if (p.coeffs.size() != 1) {
      piecewise_constant = false;
      break;
    }
  }
  if (piecewise_constant) {
    std::vector<u64> cutpoints = d.cuts;
    std::vector<std::vector<u64>> payloads(d.polys.size(), std::vector<u64>(1));
    for (std::size_t i = 0; i < d.polys.size(); ++i) {
      payloads[i][0] = d.polys[i].coeffs[0];
    }
    auto lut_key = gen_interval_lut_v2(cutpoints, payloads, in_bits, party, rng);
    upload_interval_lut_v2(lut_key, interval_key_);
    use_interval_lut_ = true;
  }
}

GpuSecureSufProgram::~GpuSecureSufProgram() {
  if (dpf_loaded_) free_dpf_batch(dpf_gpu_);
  if (use_interval_lut_) free_interval_lut_v2(interval_key_);
  if (d_pred_bits_) cudaFree(d_pred_bits_);
  if (d_query_bits_) cudaFree(d_query_bits_);
  if (d_query_to_pred_) cudaFree(d_query_to_pred_);
  if (d_const_pred_bits_) cudaFree(d_const_pred_bits_);
  d_pred_bits_ = nullptr;
  pred_capacity_ = 0;
  d_query_bits_ = nullptr;
  query_capacity_ = 0;
  d_query_to_pred_ = nullptr;
  d_const_pred_bits_ = nullptr;
}

u8* GpuSecureSufProgram::ensure_pred_bits(std::size_t n) const {
  const std::size_t needed = desc_.predicates.size() * n;
  if (needed == 0) return nullptr;
  if (needed > pred_capacity_) {
    if (d_pred_bits_) cudaFree(d_pred_bits_);
    cudaMalloc(&d_pred_bits_, needed * sizeof(u8));
    pred_capacity_ = needed;
  }
  return d_pred_bits_;
}

u8* GpuSecureSufProgram::ensure_query_bits(std::size_t n) const {
  const std::size_t needed = plan_.queries.size() * n;
  if (needed == 0) return nullptr;
  if (needed > query_capacity_) {
    if (d_query_bits_) cudaFree(d_query_bits_);
    cudaMalloc(&d_query_bits_, needed * sizeof(u8));
    query_capacity_ = needed;
  }
  return d_query_bits_;
}

void GpuSecureSufProgram::eval(const u64* d_in, std::size_t n,
                               u64* d_out_arith, u64* d_out_helpers,
                               cudaStream_t stream) const {
  u8* d_pred_bits = ensure_pred_bits(n);
  u8* d_query_bits = ensure_query_bits(n);
  if (d_query_bits && dpf_loaded_) {
    eval_dpf_batch_gpu(d_in, n, dpf_gpu_, d_query_bits, stream);
  }
  if (d_pred_bits) {
    launch_fill_const_pred_bits(d_const_pred_bits_, n,
                                static_cast<int>(desc_.predicates.size()),
                                d_pred_bits, stream);
    if (d_query_bits && d_query_to_pred_) {
      launch_scatter_pred_bits(d_query_bits, n,
                               static_cast<int>(plan_.queries.size()),
                               d_query_to_pred_, d_pred_bits, stream);
    }
  }

  if (d_out_arith) {
    if (use_interval_lut_) {
      eval_interval_lut_v2_gpu(d_in, n, interval_key_, d_out_arith, stream);
    } else {
      gpu_prog_->eval_poly_only(d_in, n, d_out_arith, stream);
    }
  }
  if (d_out_helpers && d_pred_bits) {
    gpu_prog_->eval_helpers_from_pred_bits(d_pred_bits, n, d_out_helpers, stream);
  }

}

} // namespace suf

#endif // SUF_HAVE_CUDA
