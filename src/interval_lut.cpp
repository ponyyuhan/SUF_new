#include "suf/interval_lut.hpp"
#include "suf/crypto/dcf.hpp"

namespace suf {

IntervalLutKeyV2 gen_interval_lut_v2(const std::vector<u64>& cutpoints,
                                    const std::vector<std::vector<u64>>& payloads,
                                    int in_bits,
                                    int party,
                                    std::mt19937_64& rng) {
  ensure(!cutpoints.empty(), "interval LUT: empty cutpoints");
  ensure(cutpoints.size() == payloads.size(), "interval LUT: payload size mismatch");
  const std::size_t intervals = cutpoints.size();
  const std::size_t out_words = payloads[0].size();
  for (const auto& p : payloads) {
    ensure(p.size() == out_words, "interval LUT: payload word mismatch");
  }

  IntervalLutKeyV2 key;
  key.hdr.magic = 0x53494C32;
  key.hdr.version = 2;
  key.hdr.in_bits = static_cast<u8>(in_bits);
  key.hdr.out_bits = 64;
  key.hdr.out_words = static_cast<u8>(out_words);
  key.hdr.intervals = static_cast<u32>(intervals);

  // base = last payload
  key.base_share.resize(out_words);
  std::vector<u64> base(out_words);
  for (std::size_t w = 0; w < out_words; ++w) base[w] = payloads[intervals - 1][w];
  // random split base into shares
  std::vector<u64> base0(out_words);
  for (std::size_t w = 0; w < out_words; ++w) base0[w] = rng();
  if (party == 0) {
    key.base_share = base0;
  } else {
    for (std::size_t w = 0; w < out_words; ++w) {
      key.base_share[w] = base[w] - base0[w];
    }
  }

  // deltas
  key.deltas.resize((intervals - 1) * out_words);
  for (std::size_t i = 0; i + 1 < intervals; ++i) {
    for (std::size_t w = 0; w < out_words; ++w) {
      key.deltas[i * out_words + w] = payloads[i][w] - payloads[i + 1][w];
    }
  }

  // DCF keys for each boundary
  key.dcf_batch.keys.reserve(intervals - 1);
  key.dcf_batch.scw.reserve((intervals - 1) * in_bits);
  key.dcf_batch.vcw.reserve((intervals - 1) * in_bits);

  for (std::size_t i = 0; i + 1 < intervals; ++i) {
    auto kp = keygen_dcf_lt(in_bits, cutpoints[i + 1], rng);
    const DcfKey& k = (party == 0) ? kp.k0 : kp.k1;

    DcfKeyPacked packed;
    packed.root = k.root;
    packed.tcwL = k.tcwL;
    packed.tcwR = k.tcwR;
    packed.scw_offset = static_cast<int>(key.dcf_batch.scw.size());
    packed.vcw_offset = static_cast<int>(key.dcf_batch.vcw.size());
    packed.n_bits = k.n_bits;
    packed.mask = (k.n_bits == 64) ? ~0ULL : ((1ULL << k.n_bits) - 1ULL);
    packed.g = k.g;
    packed.party = static_cast<u8>(party & 1);

    key.dcf_batch.keys.push_back(packed);
    key.dcf_batch.scw.insert(key.dcf_batch.scw.end(), k.scw.begin(), k.scw.end());
    key.dcf_batch.vcw.insert(key.dcf_batch.vcw.end(), k.vcw.begin(), k.vcw.end());
  }

  key.hdr.core_bytes = static_cast<u32>(key.dcf_batch.keys.size() * sizeof(DcfKeyPacked)
                                       + key.dcf_batch.scw.size() * sizeof(Seed)
                                       + key.dcf_batch.vcw.size() * sizeof(u64));
  key.hdr.payload_bytes = static_cast<u32>(key.deltas.size() * sizeof(u64) + key.base_share.size() * sizeof(u64));

  return key;
}

void eval_interval_lut_v2_cpu(const IntervalLutKeyV2& key,
                              const std::vector<u64>& inputs,
                              std::vector<u64>& outputs) {
  const std::size_t n = inputs.size();
  const std::size_t out_words = key.hdr.out_words;
  outputs.assign(n * out_words, 0);

  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t w = 0; w < out_words; ++w) {
      outputs[i * out_words + w] = key.base_share[w];
    }
  }

  for (std::size_t b = 0; b < key.dcf_batch.keys.size(); ++b) {
    const DcfKeyPacked& packed = key.dcf_batch.keys[b];
    const DcfKey& k = (packed.party == 0) ? DcfKey{} : DcfKey{}; // unused
    (void)k;
    for (std::size_t i = 0; i < n; ++i) {
      // rebuild a DcfKey view for eval
      DcfKey view;
      view.n_bits = packed.n_bits;
      view.root = packed.root;
      view.scw.assign(key.dcf_batch.scw.begin() + packed.scw_offset,
                      key.dcf_batch.scw.begin() + packed.scw_offset + packed.n_bits);
      view.vcw.assign(key.dcf_batch.vcw.begin() + packed.vcw_offset,
                      key.dcf_batch.vcw.begin() + packed.vcw_offset + packed.n_bits);
      view.tcwL = packed.tcwL;
      view.tcwR = packed.tcwR;
      view.g = packed.g;
      const u64 share = eval_dcf_lt_cpu(packed.party, view, inputs[i]);
      for (std::size_t w = 0; w < out_words; ++w) {
        outputs[i * out_words + w] += share * key.deltas[b * out_words + w];
      }
    }
  }
}

#ifdef SUF_HAVE_CUDA
void upload_interval_lut_v2(const IntervalLutKeyV2& key, IntervalLutKeyV2Gpu& out) {
  out.hdr = key.hdr;
  cudaMalloc(&out.d_base, key.base_share.size() * sizeof(u64));
  cudaMalloc(&out.d_deltas, key.deltas.size() * sizeof(u64));
  cudaMemcpy(out.d_base, key.base_share.data(), key.base_share.size() * sizeof(u64), cudaMemcpyHostToDevice);
  cudaMemcpy(out.d_deltas, key.deltas.data(), key.deltas.size() * sizeof(u64), cudaMemcpyHostToDevice);
  upload_dcf_batch(key.dcf_batch, out.dcf);
}

void free_interval_lut_v2(IntervalLutKeyV2Gpu& key) {
  if (key.d_base) cudaFree(key.d_base);
  if (key.d_deltas) cudaFree(key.d_deltas);
  free_dcf_batch(key.dcf);
  key.d_base = nullptr;
  key.d_deltas = nullptr;
}

void eval_interval_lut_v2_gpu(const u64* d_in, std::size_t n,
                              const IntervalLutKeyV2Gpu& key,
                              u64* d_out, cudaStream_t stream) {
  // eval all DCFs: output shape [num_keys, n]
  const std::size_t num_keys = key.dcf.num_keys;
  if (num_keys == 0) return;
  u64* d_bits = nullptr;
  cudaMalloc(&d_bits, num_keys * n * sizeof(u64));
  eval_dcf_batch_gpu(d_in, n, key.dcf, d_bits, stream);

  interval_lut_combine_gpu(key.d_base, key.d_deltas, d_bits, d_out, n, num_keys, key.hdr.out_words, stream);
  cudaFree(d_bits);
}
#endif

} // namespace suf
