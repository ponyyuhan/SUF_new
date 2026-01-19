#include "suf/secure_program.hpp"

#ifdef SUF_HAVE_CUDA

#include "suf/pfss_batch.hpp"

namespace suf {

GpuSecureSufProgram::GpuSecureSufProgram(const SUFDescriptor& d, int party, std::uint64_t seed)
  : desc_(d), plan_(compile_pfss_plan(d)), gpu_prog_(d) {
  std::mt19937_64 rng(seed);
  auto batch = build_dpf_batch(plan_, party, rng);
  upload_dpf_batch(batch, dpf_gpu_);
  dpf_loaded_ = true;

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
    auto lut_key = gen_interval_lut_v2(cutpoints, payloads, 64, party, rng);
    upload_interval_lut_v2(lut_key, interval_key_);
    use_interval_lut_ = true;
  }
}

GpuSecureSufProgram::~GpuSecureSufProgram() {
  if (dpf_loaded_) free_dpf_batch(dpf_gpu_);
  if (use_interval_lut_) free_interval_lut_v2(interval_key_);
}

void GpuSecureSufProgram::eval(const u64* d_in, std::size_t n,
                               u64* d_out_arith, u64* d_out_helpers,
                               cudaStream_t stream) const {
  u8* d_pred_bits = nullptr;
  if (!plan_.queries.empty()) {
    cudaMalloc(&d_pred_bits, plan_.queries.size() * n * sizeof(u8));
    eval_dpf_batch_gpu(d_in, n, dpf_gpu_, d_pred_bits, stream);
  }

  if (d_out_arith) {
    if (use_interval_lut_) {
      eval_interval_lut_v2_gpu(d_in, n, interval_key_, d_out_arith, stream);
    } else {
      gpu_prog_.eval_poly_only(d_in, n, d_out_arith, stream);
    }
  }
  if (d_out_helpers && d_pred_bits) {
    gpu_prog_.eval_helpers_from_pred_bits(d_pred_bits, n, d_out_helpers, stream);
  }

  if (d_pred_bits) cudaFree(d_pred_bits);
}

} // namespace suf

#endif // SUF_HAVE_CUDA
