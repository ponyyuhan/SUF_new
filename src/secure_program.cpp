#include "suf/secure_program.hpp"

#ifdef SUF_HAVE_CUDA

#include "suf/masked_compile.hpp"
#include "suf/gpu_kernels.hpp"
#include "suf/pfss_batch.hpp"

namespace suf {

namespace {
inline u64 mul_mod64(u64 a, u64 b) {
  return static_cast<u64>(static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b));
}

std::vector<std::vector<u64>> build_binom(int degree) {
  std::vector<std::vector<u64>> binom(static_cast<std::size_t>(degree + 1),
                                      std::vector<u64>(static_cast<std::size_t>(degree + 1), 0));
  binom[0][0] = 1;
  for (int k = 1; k <= degree; ++k) {
    binom[k][0] = 1;
    binom[k][k] = 1;
    for (int i = 1; i < k; ++i) {
      binom[k][i] = binom[k - 1][i - 1] + binom[k - 1][i];
    }
  }
  return binom;
}

std::vector<u64> build_pow_neg_r(int degree, u64 r) {
  std::vector<u64> pow(static_cast<std::size_t>(degree + 1), 0);
  pow[0] = 1;
  const u64 r_neg = static_cast<u64>(0ULL - r);
  for (int i = 1; i <= degree; ++i) {
    pow[static_cast<std::size_t>(i)] = mul_mod64(pow[static_cast<std::size_t>(i - 1)], r_neg);
  }
  return pow;
}

std::vector<u64> shift_poly_coeffs(const std::vector<u64>& coeffs,
                                   int degree,
                                   const std::vector<std::vector<u64>>& binom,
                                   const std::vector<u64>& pow_neg_r) {
  std::vector<u64> out(static_cast<std::size_t>(degree + 1), 0);
  const int max_k = std::min<int>(degree, static_cast<int>(coeffs.size()) - 1);
  for (int k = 0; k <= max_k; ++k) {
    const u64 c = coeffs[static_cast<std::size_t>(k)];
    if (c == 0) continue;
    for (int i = 0; i <= k; ++i) {
      const u64 term0 = mul_mod64(c, binom[k][i]);
      const u64 term = mul_mod64(term0, pow_neg_r[static_cast<std::size_t>(k - i)]);
      out[static_cast<std::size_t>(i)] += term;
    }
  }
  return out;
}
} // namespace

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

  poly_degree_ = 0;
  for (const auto& p : desc_.polys) {
    if (!p.coeffs.empty()) {
      poly_degree_ = std::max<int>(poly_degree_, static_cast<int>(p.coeffs.size()) - 1);
    }
  }

  if (poly_degree_ == 0) {
    std::vector<u64> cutpoints = desc_.cuts;
    std::vector<std::vector<u64>> payloads(desc_.polys.size(), std::vector<u64>(1));
    for (std::size_t i = 0; i < desc_.polys.size(); ++i) {
      payloads[i][0] = desc_.polys[i].coeffs.empty() ? 0ULL : desc_.polys[i].coeffs[0];
    }
    auto lut_key = gen_interval_lut_v2(cutpoints, payloads, in_bits, party, rng);
    upload_interval_lut_v2(lut_key, interval_key_);
    use_interval_lut_ = true;
  } else {
    const auto binom = build_binom(poly_degree_);
    const u64 shift_r = mask_aware_ ? r_in_ : 0;
    const auto pow_neg_r = build_pow_neg_r(poly_degree_, shift_r);
    std::vector<u64> cutpoints = desc_.cuts;
    std::vector<std::vector<u64>> payloads(desc_.polys.size());
    for (std::size_t i = 0; i < desc_.polys.size(); ++i) {
      payloads[i] = shift_poly_coeffs(desc_.polys[i].coeffs, poly_degree_, binom, pow_neg_r);
    }
    auto lut_key = gen_interval_lut_v2(cutpoints, payloads, in_bits, party, rng);
    upload_interval_lut_v2(lut_key, coeff_key_);
    use_coeff_lut_ = true;
  }
}

GpuSecureSufProgram::~GpuSecureSufProgram() {
  if (dpf_loaded_) free_dpf_batch(dpf_gpu_);
  if (use_interval_lut_) free_interval_lut_v2(interval_key_);
  if (use_coeff_lut_) free_interval_lut_v2(coeff_key_);
  if (d_pred_bits_) cudaFree(d_pred_bits_);
  if (d_query_bits_) cudaFree(d_query_bits_);
  if (d_coeffs_) cudaFree(d_coeffs_);
  if (d_query_to_pred_) cudaFree(d_query_to_pred_);
  if (d_const_pred_bits_) cudaFree(d_const_pred_bits_);
  d_pred_bits_ = nullptr;
  pred_capacity_ = 0;
  d_query_bits_ = nullptr;
  query_capacity_ = 0;
  d_coeffs_ = nullptr;
  coeff_capacity_ = 0;
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

u64* GpuSecureSufProgram::ensure_coeffs(std::size_t n) const {
  if (!use_coeff_lut_) return nullptr;
  const std::size_t stride = static_cast<std::size_t>(poly_degree_ + 1);
  const std::size_t needed = n * stride;
  if (needed == 0) return nullptr;
  if (needed > coeff_capacity_) {
    if (d_coeffs_) cudaFree(d_coeffs_);
    cudaMalloc(&d_coeffs_, needed * sizeof(u64));
    coeff_capacity_ = needed;
  }
  return d_coeffs_;
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
    } else if (use_coeff_lut_) {
      u64* d_coeffs = ensure_coeffs(n);
      eval_interval_lut_v2_gpu(d_in, n, coeff_key_, d_coeffs, stream);
      launch_eval_poly_from_coeffs(d_in, d_coeffs, n, poly_degree_, d_out_arith, stream);
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
