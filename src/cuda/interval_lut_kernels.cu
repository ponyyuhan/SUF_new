#include "suf/interval_lut.hpp"

#ifdef SUF_HAVE_CUDA

#include <cuda_runtime.h>

namespace suf {

__global__ void kernel_interval_lut_combine(const u64* base, const u64* deltas, const u64* bits,
                                            u64* out, std::size_t n, std::size_t num_keys, int out_words) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  for (int w = 0; w < out_words; ++w) {
    u64 acc = base[w];
    for (std::size_t k = 0; k < num_keys; ++k) {
      u64 bshare = bits[k * n + idx];
      acc += bshare * deltas[k * out_words + w];
    }
    out[idx * out_words + w] = acc;
  }
}

void interval_lut_combine_gpu(const u64* d_base, const u64* d_deltas, const u64* d_bits,
                              u64* d_out, std::size_t n, std::size_t num_keys, int out_words,
                              cudaStream_t stream) {
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_interval_lut_combine<<<blocks, threads, 0, stream>>>(d_base, d_deltas, d_bits, d_out, n, num_keys, out_words);
}

} // namespace suf

#endif // SUF_HAVE_CUDA
