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
}

GpuSecureSufProgram::~GpuSecureSufProgram() {
  if (dpf_loaded_) free_dpf_batch(dpf_gpu_);
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
    gpu_prog_.eval_poly_only(d_in, n, d_out_arith, stream);
  }
  if (d_out_helpers && d_pred_bits) {
    gpu_prog_.eval_helpers_from_pred_bits(d_pred_bits, n, d_out_helpers, stream);
  }

  if (d_pred_bits) cudaFree(d_pred_bits);
}

} // namespace suf

#endif // SUF_HAVE_CUDA
