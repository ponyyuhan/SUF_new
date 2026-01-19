#pragma once

#include "suf/common.hpp"
#include "suf/dcf_batch.hpp"

#ifdef SUF_HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include <vector>

namespace suf {

struct IntervalLutHeaderV2 {
  u32 magic = 0x53494C32; // 'SIL2'
  u8 version = 2;
  u8 in_bits = 0;
  u8 out_bits = 64;
  u8 out_words = 0;
  u32 core_bytes = 0;
  u32 payload_bytes = 0;
  u32 intervals = 0;
  u32 flags = 0;
};

struct IntervalLutKeyV2 {
  IntervalLutHeaderV2 hdr;
  std::vector<u64> base_share; // out_words
  DcfKeyBatch dcf_batch;
  std::vector<u64> deltas; // (intervals-1) * out_words, public
};

#ifdef SUF_HAVE_CUDA
struct IntervalLutKeyV2Gpu {
  IntervalLutHeaderV2 hdr;
  u64* d_base = nullptr;
  u64* d_deltas = nullptr;
  DcfKeyBatchGpu dcf;
};
#endif

IntervalLutKeyV2 gen_interval_lut_v2(const std::vector<u64>& cutpoints,
                                    const std::vector<std::vector<u64>>& payloads,
                                    int in_bits,
                                    int party,
                                    std::mt19937_64& rng);

void eval_interval_lut_v2_cpu(const IntervalLutKeyV2& key,
                              const std::vector<u64>& inputs,
                              std::vector<u64>& outputs);

#ifdef SUF_HAVE_CUDA
void upload_interval_lut_v2(const IntervalLutKeyV2& key, IntervalLutKeyV2Gpu& out);
void free_interval_lut_v2(IntervalLutKeyV2Gpu& key);
void eval_interval_lut_v2_gpu(const u64* d_in, std::size_t n,
                              const IntervalLutKeyV2Gpu& key,
                              u64* d_out, cudaStream_t stream = nullptr);

void interval_lut_combine_gpu(const u64* d_base, const u64* d_deltas, const u64* d_bits,
                              u64* d_out, std::size_t n, std::size_t num_keys, int out_words,
                              cudaStream_t stream = nullptr);
#endif

} // namespace suf
