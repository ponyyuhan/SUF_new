#include "suf/sigma_bridge.hpp"

#include "suf/interval_lut.hpp"
#include "suf/masked_compile.hpp"
#include "suf/ir.hpp"

#include "utils/gpu_mem.h"
#include "utils/sigma_comms.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <random>
#include <unordered_map>
#include <vector>

namespace {

enum class GateKind : int {
  Gelu = 0,
  Silu = 1,
  NExp = 2,
  Inv = 3,
  Rsqrt = 4,
};

struct DescKey {
  bool silu = false;
  int bw = 0;
  int scale = 0;
  int intervals = 0;

  bool operator==(const DescKey& other) const {
    return silu == other.silu && bw == other.bw && scale == other.scale && intervals == other.intervals;
  }
};

struct DescKeyHash {
  std::size_t operator()(const DescKey& k) const noexcept {
    std::size_t h = static_cast<std::size_t>(k.silu);
    h = h * 1315423911u + static_cast<std::size_t>(k.bw);
    h = h * 1315423911u + static_cast<std::size_t>(k.scale);
    h = h * 1315423911u + static_cast<std::size_t>(k.intervals);
    return h;
  }
};

struct TableKey {
  GateKind kind = GateKind::NExp;
  int in_bits = 0;
  int scale_in = 0;
  int scale_out = 0;
  int out_bits = 0;
  std::uint64_t clamp_min = 0;
  std::uint64_t clamp_max = 0;
  std::uint64_t extra = 0;

  bool operator==(const TableKey& other) const {
    return kind == other.kind && in_bits == other.in_bits &&
           scale_in == other.scale_in && scale_out == other.scale_out &&
           out_bits == other.out_bits && clamp_min == other.clamp_min &&
           clamp_max == other.clamp_max && extra == other.extra;
  }
};

struct TableKeyHash {
  std::size_t operator()(const TableKey& k) const noexcept {
    std::size_t h = static_cast<std::size_t>(k.kind);
    h = h * 1315423911u + static_cast<std::size_t>(k.in_bits);
    h = h * 1315423911u + static_cast<std::size_t>(k.scale_in);
    h = h * 1315423911u + static_cast<std::size_t>(k.scale_out);
    h = h * 1315423911u + static_cast<std::size_t>(k.out_bits);
    h = h * 1315423911u + static_cast<std::size_t>(k.clamp_min);
    h = h * 1315423911u + static_cast<std::size_t>(k.clamp_max);
    h = h * 1315423911u + static_cast<std::size_t>(k.extra);
    return h;
  }
};

std::mutex g_desc_mutex;
std::unordered_map<DescKey, suf::SUFDescriptor, DescKeyHash> g_desc_cache;
std::mutex g_table_mutex;
std::unordered_map<TableKey, std::vector<std::uint64_t>, TableKeyHash> g_table_cache;

int env_int(const char* name, int fallback) {
  const char* v = std::getenv(name);
  if (!v || !*v) return fallback;
  return std::atoi(v);
}

double env_double(const char* name, double fallback) {
  const char* v = std::getenv(name);
  if (!v || !*v) return fallback;
  char* end = nullptr;
  const double val = std::strtod(v, &end);
  if (end == v) return fallback;
  return val;
}

int bits_needed(std::uint64_t v) {
  int bits = 0;
  while (v > 0) {
    ++bits;
    v >>= 1;
  }
  return bits > 0 ? bits : 1;
}

std::uint64_t mask_for_bw(int bw) {
  if (bw >= 64) return ~0ULL;
  return (1ULL << bw) - 1ULL;
}

std::uint64_t mod_pow2(std::int64_t v, int bw) {
  if (bw >= 64) return static_cast<std::uint64_t>(v);
  const __int128 mod = (__int128)1 << bw;
  __int128 x = static_cast<__int128>(v) % mod;
  if (x < 0) x += mod;
  return static_cast<std::uint64_t>(x);
}

suf::SUFDescriptor build_activation_desc(bool silu, int bw, int scale, int intervals) {
  suf::SUFDescriptor d;
  d.cuts.resize(intervals);
  d.polys.resize(intervals);

  const __int128 domain = (__int128)1 << bw;
  const __int128 step = domain / intervals;
  const std::uint64_t sign_bit = (bw >= 64) ? (1ULL << 63) : (1ULL << (bw - 1));

  for (int i = 0; i < intervals; ++i) {
    const __int128 cut = step * i;
    d.cuts[i] = static_cast<std::uint64_t>(cut);
  }

  for (int i = 0; i < intervals; ++i) {
    const __int128 mid = step * i + step / 2;
    const std::uint64_t x = static_cast<std::uint64_t>(mid);
    __int128 signed_x = (x & sign_bit) ? (static_cast<__int128>(x) - domain) : static_cast<__int128>(x);
    const long double x_real = static_cast<long double>(signed_x) / static_cast<long double>(1ULL << scale);
    long double y_real = 0.0L;
    if (silu) {
      y_real = x_real / (1.0L + std::exp(-x_real));
    } else {
      const long double t = x_real / std::sqrt(2.0L);
      y_real = x_real * 0.5L * (1.0L + std::erf(t));
    }
    const long double y_scaled = y_real * static_cast<long double>(1ULL << scale);
    const std::int64_t y_fixed = llroundl(y_scaled);
    const std::uint64_t y_mod = mod_pow2(y_fixed, bw);

    d.polys[i].coeffs.clear();
    d.polys[i].coeffs.push_back(y_mod);
  }
  return d;
}

const suf::SUFDescriptor& get_activation_desc(bool silu, int bw, int scale, int intervals) {
  DescKey key{.silu = silu, .bw = bw, .scale = scale, .intervals = intervals};
  std::lock_guard<std::mutex> lock(g_desc_mutex);
  auto it = g_desc_cache.find(key);
  if (it != g_desc_cache.end()) return it->second;
  auto desc = build_activation_desc(silu, bw, scale, intervals);
  auto res = g_desc_cache.emplace(key, std::move(desc));
  return res.first->second;
}

std::size_t table_size_for_bits(int in_bits) {
  suf::ensure(in_bits > 0 && in_bits < 32, "table size bits out of range");
  return static_cast<std::size_t>(1ULL << in_bits);
}

std::vector<std::uint64_t> build_table(const TableKey& key) {
  const std::size_t table_size = table_size_for_bits(key.in_bits);
  std::vector<std::uint64_t> table(table_size);
  const long double scale_in = static_cast<long double>(1ULL << key.scale_in);
  const long double scale_out = static_cast<long double>(1ULL << key.scale_out);
  for (std::size_t i = 0; i < table_size; ++i) {
    std::uint64_t x_fixed = static_cast<std::uint64_t>(i);
    if (x_fixed < key.clamp_min) x_fixed = key.clamp_min;
    if (x_fixed > key.clamp_max) x_fixed = key.clamp_max;
    const long double x_real = static_cast<long double>(x_fixed) / scale_in;
    long double y_real = 0.0L;
    switch (key.kind) {
      case GateKind::NExp:
        y_real = std::exp(-x_real);
        break;
      case GateKind::Inv:
        y_real = (x_real <= 0.0L) ? 0.0L : (1.0L / x_real);
        break;
      case GateKind::Rsqrt: {
        if (x_real <= 0.0L) {
          y_real = 0.0L;
        } else {
          const long double denom = x_real / static_cast<long double>(key.extra);
          y_real = (denom <= 0.0L) ? 0.0L : (1.0L / std::sqrt(denom));
        }
        break;
      }
      default:
        y_real = 0.0L;
        break;
    }
    const long double y_scaled = y_real * scale_out;
    const std::int64_t y_fixed = llroundl(y_scaled);
    table[i] = mod_pow2(y_fixed, key.out_bits);
  }
  return table;
}

const std::vector<std::uint64_t>& get_table(const TableKey& key) {
  std::lock_guard<std::mutex> lock(g_table_mutex);
  auto it = g_table_cache.find(key);
  if (it != g_table_cache.end()) return it->second;
  auto table = build_table(key);
  auto res = g_table_cache.emplace(key, std::move(table));
  return res.first->second;
}

suf::IntervalLutKeyV2 build_direct_table_key(const std::vector<std::uint64_t>& table,
                                             int in_bits,
                                             int out_bits,
                                             int party,
                                             std::mt19937_64& rng,
                                             std::uint64_t r_in) {
  suf::IntervalLutKeyV2 key;
  key.hdr.magic = 0x53494C32;
  key.hdr.version = 2;
  key.hdr.in_bits = static_cast<u8>(in_bits);
  key.hdr.out_bits = 64;
  key.hdr.out_words = 1;
  key.hdr.intervals = static_cast<u32>(table.size());
  key.hdr.flags |= suf::kIntervalLutFlagDirectTable;
  key.base_share.assign(1, 0);

  const std::size_t table_size = table.size();
  key.dmpf.in_bits = in_bits;
  key.dmpf.points = table_size;
  key.dmpf.out_words = 1;
  key.dmpf.dcf_batch.n_bits = in_bits;
  key.dmpf.deltas.resize(table_size);

  const std::uint64_t mask = mask_for_bw(in_bits);
  const std::uint64_t r = r_in & mask;
  for (std::size_t h = 0; h < table_size; ++h) {
    const std::uint64_t idx = static_cast<std::uint64_t>(h) & mask;
    const std::uint64_t x = (idx + table_size - r) & mask;
    const std::uint64_t value = table[static_cast<std::size_t>(x)];
    const std::uint64_t share0 = rng();
    key.dmpf.deltas[h] = (party == 0) ? share0 : (value - share0);
  }

  key.hdr.core_bytes = 0;
  key.hdr.payload_bytes = static_cast<u32>(key.dmpf.deltas.size() * sizeof(u64));
  return key;
}

struct SufGateState {
  GateKind kind = GateKind::Gelu;
  int bw = 0;
  int scale = 0;
  int in_bits = 0;
  int scale_in = 0;
  std::uint64_t extra = 0;
  std::size_t n = 0;
  std::uint64_t r_in_share = 0;
  std::vector<std::uint64_t> input_mask_share;
  std::vector<std::uint64_t> output_mask_share;
  std::uint64_t* d_input_mask = nullptr;
  std::uint64_t* d_output_mask = nullptr;
  suf::IntervalLutKeyV2Gpu lut_key;
};

std::vector<SufGateState> g_suf_gates;
std::size_t g_suf_eval_idx = 0;
std::mt19937_64 g_master_rng(0x53C0FFEEu);

std::uint64_t next_gate_seed() {
  return g_master_rng();
}

__device__ __forceinline__ std::uint64_t mod_pow2_dev(std::uint64_t v, int bw) {
  if (bw >= 64) return v;
  return v & ((std::uint64_t(1) << bw) - 1ULL);
}

__global__ void kernel_remask(const std::uint64_t* masked_in,
                              const std::uint64_t* mask_in,
                              std::uint64_t* out,
                              std::uint64_t r_in_share,
                              int bw,
                              std::size_t n) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  std::uint64_t v = masked_in[idx] - mask_in[idx];
  v = mod_pow2_dev(v, bw);
  v = mod_pow2_dev(v + r_in_share, bw);
  out[idx] = v;
}

__global__ void kernel_u16_to_u64(const std::uint16_t* in,
                                  std::uint64_t* out,
                                  std::size_t n) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out[idx] = static_cast<std::uint64_t>(in[idx]);
}

__global__ void kernel_add_mask(std::uint64_t* out,
                                const std::uint64_t* mask,
                                int bw,
                                std::size_t n) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  std::uint64_t v = out[idx] + mask[idx];
  out[idx] = mod_pow2_dev(v, bw);
}

std::uint64_t* keygen_table_gate_u64(GateKind kind,
                                     int bw_out,
                                     int scale_out,
                                     int in_bits,
                                     int scale_in,
                                     std::uint64_t clamp_min,
                                     std::uint64_t clamp_max,
                                     std::uint64_t extra,
                                     int party,
                                     const std::uint64_t* d_input_mask,
                                     std::size_t n) {
  const std::uint64_t in_mask = mask_for_bw(in_bits);
  const std::uint64_t out_mask = mask_for_bw(bw_out);
  const auto seed = next_gate_seed();
  std::mt19937_64 rng(seed);

  const std::uint64_t r_in = rng() & in_mask;
  const std::uint64_t r_in0 = rng() & in_mask;
  std::uint64_t r_in_share = (party == 0) ? r_in0 : (r_in - r_in0);
  if (in_bits < 64) r_in_share &= in_mask;

  std::vector<std::uint64_t> input_mask_share(n);
  cudaMemcpy(input_mask_share.data(), d_input_mask, n * sizeof(std::uint64_t), cudaMemcpyDeviceToHost);
  if (in_bits < 64) {
    for (auto& v : input_mask_share) v &= in_mask;
  }

  std::vector<std::uint64_t> output_mask_share(n);
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint64_t r_out = rng() & out_mask;
    const std::uint64_t r_out0 = rng() & out_mask;
    output_mask_share[i] = (party == 0) ? r_out0 : (r_out - r_out0);
    if (bw_out < 64) output_mask_share[i] &= out_mask;
  }

  std::uint64_t* d_out = reinterpret_cast<std::uint64_t*>(gpuMalloc(n * sizeof(std::uint64_t)));
  cudaMemcpy(d_out, output_mask_share.data(), n * sizeof(std::uint64_t), cudaMemcpyHostToDevice);

  TableKey tkey;
  tkey.kind = kind;
  tkey.in_bits = in_bits;
  tkey.scale_in = scale_in;
  tkey.scale_out = scale_out;
  tkey.out_bits = bw_out;
  tkey.clamp_min = clamp_min;
  tkey.clamp_max = clamp_max;
  tkey.extra = extra;
  const auto& table = get_table(tkey);

  auto lut_key = build_direct_table_key(table, in_bits, bw_out, party, rng, r_in);
  suf::IntervalLutKeyV2Gpu lut_gpu{};
  suf::upload_interval_lut_v2(lut_key, lut_gpu);

  SufGateState state;
  state.kind = kind;
  state.bw = bw_out;
  state.scale = scale_out;
  state.in_bits = in_bits;
  state.scale_in = scale_in;
  state.extra = extra;
  state.n = n;
  state.r_in_share = r_in_share;
  state.input_mask_share = std::move(input_mask_share);
  state.output_mask_share = std::move(output_mask_share);
  state.d_input_mask = reinterpret_cast<std::uint64_t*>(gpuMalloc(n * sizeof(std::uint64_t)));
  state.d_output_mask = reinterpret_cast<std::uint64_t*>(gpuMalloc(n * sizeof(std::uint64_t)));
  cudaMemcpy(state.d_input_mask, state.input_mask_share.data(), n * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(state.d_output_mask, state.output_mask_share.data(), n * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
  state.lut_key = lut_gpu;
  g_suf_gates.push_back(std::move(state));

  return d_out;
}

std::uint64_t* keygen_table_gate_u16(GateKind kind,
                                     int bw_out,
                                     int scale_out,
                                     int in_bits,
                                     int scale_in,
                                     std::uint64_t clamp_min,
                                     std::uint64_t clamp_max,
                                     std::uint64_t extra,
                                     int party,
                                     const std::uint16_t* d_input_mask,
                                     std::size_t n) {
  std::uint64_t* d_input_u64 = reinterpret_cast<std::uint64_t*>(gpuMalloc(n * sizeof(std::uint64_t)));
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_u16_to_u64<<<blocks, threads>>>(d_input_mask, d_input_u64, n);
  cudaDeviceSynchronize();
  auto* out = keygen_table_gate_u64(kind, bw_out, scale_out, in_bits, scale_in,
                                    clamp_min, clamp_max, extra,
                                    party, d_input_u64, n);
  gpuFree(d_input_u64);
  return out;
}

std::uint64_t* eval_gate_u64(GateKind expected_kind,
                             int bw_out,
                             int scale_out,
                             int scale_in,
                             int in_bits,
                             std::uint64_t extra,
                             SigmaPeer* peer,
                             int party,
                             const std::uint64_t* d_input_masked,
                             std::size_t n,
                             Stats* s) {
  if (g_suf_eval_idx >= g_suf_gates.size()) return nullptr;
  auto& gate = g_suf_gates[g_suf_eval_idx++];
  if (gate.kind != expected_kind || gate.n != n || gate.bw != bw_out ||
      gate.scale != scale_out || gate.scale_in != scale_in ||
      gate.in_bits != in_bits || gate.extra != extra) {
    return nullptr;
  }

  const std::size_t bytes = n * sizeof(std::uint64_t);
  auto d_hat = reinterpret_cast<std::uint64_t*>(gpuMalloc(bytes));
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_remask<<<blocks, threads>>>(d_input_masked, gate.d_input_mask, d_hat,
                                     gate.r_in_share, gate.in_bits, n);
  cudaDeviceSynchronize();

  peer->reconstructInPlace(d_hat, gate.in_bits, n, s);

  auto d_out = reinterpret_cast<std::uint64_t*>(gpuMalloc(bytes));
  suf::eval_interval_lut_v2_gpu(d_hat, n, gate.lut_key, d_out, nullptr);
  gpuFree(d_hat);

  kernel_add_mask<<<blocks, threads>>>(d_out, gate.d_output_mask, gate.bw, n);
  cudaDeviceSynchronize();
  peer->reconstructInPlace(d_out, gate.bw, n, s);
  return d_out;
}

std::uint64_t* eval_gate_u16(GateKind expected_kind,
                             int bw_out,
                             int scale_out,
                             int scale_in,
                             int in_bits,
                             std::uint64_t extra,
                             SigmaPeer* peer,
                             int party,
                             const std::uint16_t* d_input_masked,
                             std::size_t n,
                             Stats* s) {
  auto d_input_u64 = reinterpret_cast<std::uint64_t*>(gpuMalloc(n * sizeof(std::uint64_t)));
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_u16_to_u64<<<blocks, threads>>>(d_input_masked, d_input_u64, n);
  cudaDeviceSynchronize();

  auto* out = eval_gate_u64(expected_kind, bw_out, scale_out, scale_in, in_bits, extra,
                            peer, party, d_input_u64, n, s);
  gpuFree(d_input_u64);
  return out;
}

} // namespace

extern "C" void suf_sigma_reset_keygen() {
  for (auto& gate : g_suf_gates) {
    suf::free_interval_lut_v2(gate.lut_key);
    if (gate.d_input_mask) gpuFree(gate.d_input_mask);
    if (gate.d_output_mask) gpuFree(gate.d_output_mask);
  }
  g_suf_gates.clear();
  g_suf_eval_idx = 0;
  g_master_rng.seed(0x53C0FFEEu);
}

extern "C" void suf_sigma_reset_eval() {
  g_suf_eval_idx = 0;
}

extern "C" void suf_sigma_clear() {
  for (auto& gate : g_suf_gates) {
    suf::free_interval_lut_v2(gate.lut_key);
    if (gate.d_input_mask) gpuFree(gate.d_input_mask);
    if (gate.d_output_mask) gpuFree(gate.d_output_mask);
  }
  g_suf_gates.clear();
  g_suf_eval_idx = 0;
}

extern "C" std::uint64_t* suf_sigma_keygen_activation(int party,
                                                       int bw,
                                                       int scale,
                                                       bool silu,
                                                       const std::uint64_t* d_input_mask,
                                                       std::size_t n) {
  const int intervals = env_int(silu ? "SUF_SILU_INTERVALS" : "SUF_GELU_INTERVALS",
                                silu ? 1024 : 256);
  const std::uint64_t mask = mask_for_bw(bw);

  const auto seed = next_gate_seed();
  std::mt19937_64 rng(seed);

  std::uint64_t r_in = rng() & mask;
  std::uint64_t r_in0 = rng() & mask;
  std::uint64_t r_in_share = (party == 0) ? r_in0 : (r_in - r_in0);
  if (bw < 64) r_in_share &= mask;

  std::vector<std::uint64_t> input_mask_share(n);
  cudaMemcpy(input_mask_share.data(), d_input_mask, n * sizeof(std::uint64_t), cudaMemcpyDeviceToHost);

  std::vector<std::uint64_t> output_mask_share(n);
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint64_t r_out = rng() & mask;
    const std::uint64_t r_out0 = rng() & mask;
    output_mask_share[i] = (party == 0) ? r_out0 : (r_out - r_out0);
    if (bw < 64) output_mask_share[i] &= mask;
  }

  std::uint64_t* d_out = reinterpret_cast<std::uint64_t*>(gpuMalloc(n * sizeof(std::uint64_t)));
  cudaMemcpy(d_out, output_mask_share.data(), n * sizeof(std::uint64_t), cudaMemcpyHostToDevice);

  const auto& base_desc = get_activation_desc(silu, bw, scale, intervals);
  auto inst = suf::compile_masked_gate_instance(base_desc, bw, r_in, party, rng);

  std::vector<std::vector<std::uint64_t>> payloads(inst.desc.polys.size());
  for (std::size_t i = 0; i < inst.desc.polys.size(); ++i) {
    payloads[i] = inst.desc.polys[i].coeffs;
  }

  auto lut_key = suf::gen_interval_lut_v2(inst.desc.cuts, payloads, bw, party, rng);
  suf::IntervalLutKeyV2Gpu lut_gpu{};
  suf::upload_interval_lut_v2(lut_key, lut_gpu);

  SufGateState state;
  state.kind = silu ? GateKind::Silu : GateKind::Gelu;
  state.bw = bw;
  state.scale = scale;
  state.in_bits = bw;
  state.scale_in = scale;
  state.extra = 0;
  state.n = n;
  state.r_in_share = r_in_share;
  state.input_mask_share = std::move(input_mask_share);
  state.output_mask_share = std::move(output_mask_share);
  state.d_input_mask = reinterpret_cast<std::uint64_t*>(gpuMalloc(n * sizeof(std::uint64_t)));
  state.d_output_mask = reinterpret_cast<std::uint64_t*>(gpuMalloc(n * sizeof(std::uint64_t)));
  cudaMemcpy(state.d_input_mask, state.input_mask_share.data(), n * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(state.d_output_mask, state.output_mask_share.data(), n * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
  state.lut_key = lut_gpu;
  g_suf_gates.push_back(std::move(state));

  return d_out;
}

extern "C" std::uint64_t* suf_sigma_keygen_nexp(int party,
                                                int bw,
                                                int scale,
                                                const std::uint64_t* d_input_mask,
                                                std::size_t n) {
  const double xmax = env_double("SUF_NEXP_XMAX", 16.0);
  const std::uint64_t clamp_raw = static_cast<std::uint64_t>(llroundl(xmax * (1ULL << scale)));
  int in_bits = env_int("SUF_NEXP_BITS", 0);
  if (in_bits <= 0) {
    in_bits = std::min<int>(16, std::min<int>(bw, bits_needed(clamp_raw)));
  }
  const std::uint64_t clamp_max = std::min<std::uint64_t>(clamp_raw, mask_for_bw(in_bits));
  return keygen_table_gate_u64(GateKind::NExp, bw, scale, in_bits, scale,
                               0, clamp_max, 0, party, d_input_mask, n);
}

extern "C" std::uint64_t* suf_sigma_eval_nexp(SigmaPeer* peer,
                                              int party,
                                              int bw,
                                              int scale,
                                              const std::uint64_t* d_input_masked,
                                              std::size_t n,
                                              Stats* s) {
  const double xmax = env_double("SUF_NEXP_XMAX", 16.0);
  const std::uint64_t clamp_raw = static_cast<std::uint64_t>(llroundl(xmax * (1ULL << scale)));
  int in_bits = env_int("SUF_NEXP_BITS", 0);
  if (in_bits <= 0) {
    in_bits = std::min<int>(16, std::min<int>(bw, bits_needed(clamp_raw)));
  }
  return eval_gate_u64(GateKind::NExp, bw, scale, scale, in_bits, 0,
                       peer, party, d_input_masked, n, s);
}

extern "C" std::uint64_t* suf_sigma_keygen_inverse(int party,
                                                   int bw,
                                                   int scale,
                                                   int nmax,
                                                   const std::uint16_t* d_input_mask,
                                                   std::size_t n) {
  const int scale_in = env_int("SUF_INV_FRAC", 6);
  int in_bits = env_int("SUF_INV_BITS", 0);
  const std::uint64_t max_fixed = (nmax > 0) ? (static_cast<std::uint64_t>(nmax) << scale_in) : 0;
  if (in_bits <= 0) {
    std::uint64_t tmp = max_fixed;
    int bits = 0;
    while (tmp > 0) {
      ++bits;
      tmp >>= 1;
    }
    in_bits = std::max(1, std::min(16, bits));
  }
  const std::uint64_t clamp_min = (1ULL << scale_in);
  std::uint64_t clamp_max = std::min<std::uint64_t>(max_fixed, mask_for_bw(in_bits));
  if (clamp_max < clamp_min) clamp_max = clamp_min;
  return keygen_table_gate_u16(GateKind::Inv, bw, scale, in_bits, scale_in,
                               clamp_min, clamp_max, static_cast<std::uint64_t>(nmax),
                               party, d_input_mask, n);
}

extern "C" std::uint64_t* suf_sigma_eval_inverse(SigmaPeer* peer,
                                                 int party,
                                                 int bw,
                                                 int scale,
                                                 int nmax,
                                                 const std::uint16_t* d_input_masked,
                                                 std::size_t n,
                                                 Stats* s) {
  const int scale_in = env_int("SUF_INV_FRAC", 6);
  int in_bits = env_int("SUF_INV_BITS", 0);
  const std::uint64_t max_fixed = (nmax > 0) ? (static_cast<std::uint64_t>(nmax) << scale_in) : 0;
  if (in_bits <= 0) {
    std::uint64_t tmp = max_fixed;
    int bits = 0;
    while (tmp > 0) {
      ++bits;
      tmp >>= 1;
    }
    in_bits = std::max(1, std::min(16, bits));
  }
  return eval_gate_u16(GateKind::Inv, bw, scale, scale_in, in_bits, static_cast<std::uint64_t>(nmax),
                       peer, party, d_input_masked, n, s);
}

extern "C" std::uint64_t* suf_sigma_keygen_rsqrt(int party,
                                                 int bw,
                                                 int scale,
                                                 int extradiv,
                                                 const std::uint16_t* d_input_mask,
                                                 std::size_t n) {
  const int target_frac = env_int("SUF_RSQRT_FRAC", 6);
  const int shift = std::max(0, 2 * scale - target_frac);
  const int in_bits = std::max(1, std::min(16, bw - shift));
  const int scale_in = 2 * scale - shift;
  const double vmax_real = env_double("SUF_RSQRT_VMAX", 16.0);
  const double eps_real = env_double("SUF_RSQRT_EPS", 0.0);
  const std::uint64_t clamp_min = std::max<std::uint64_t>(1, static_cast<std::uint64_t>(llroundl(eps_real * (1ULL << scale_in))));
  const std::uint64_t vmax_fixed = static_cast<std::uint64_t>(llroundl(vmax_real * (1ULL << scale_in)));
  std::uint64_t clamp_max = std::min<std::uint64_t>(vmax_fixed, mask_for_bw(in_bits));
  if (clamp_max < clamp_min) clamp_max = clamp_min;
  return keygen_table_gate_u16(GateKind::Rsqrt, bw, scale, in_bits, scale_in,
                               clamp_min, clamp_max, static_cast<std::uint64_t>(extradiv),
                               party, d_input_mask, n);
}

extern "C" std::uint64_t* suf_sigma_eval_rsqrt(SigmaPeer* peer,
                                               int party,
                                               int bw,
                                               int scale,
                                               int extradiv,
                                               const std::uint16_t* d_input_masked,
                                               std::size_t n,
                                               Stats* s) {
  const int target_frac = env_int("SUF_RSQRT_FRAC", 6);
  const int shift = std::max(0, 2 * scale - target_frac);
  const int in_bits = std::max(1, std::min(16, bw - shift));
  const int scale_in = 2 * scale - shift;
  const double vmax_real = env_double("SUF_RSQRT_VMAX", 16.0);
  const std::uint64_t vmax_fixed = static_cast<std::uint64_t>(llroundl(vmax_real * (1ULL << scale_in)));
  return eval_gate_u16(GateKind::Rsqrt, bw, scale, scale_in, in_bits, static_cast<std::uint64_t>(extradiv),
                       peer, party, d_input_masked, n, s);
}

extern "C" std::uint64_t* suf_sigma_eval_activation(SigmaPeer* peer,
                                                    int party,
                                                    int bw,
                                                    int scale,
                                                    bool silu,
                                                    const std::uint64_t* d_input_masked,
                                                    std::size_t n,
                                                    Stats* s) {
  const GateKind kind = silu ? GateKind::Silu : GateKind::Gelu;
  return eval_gate_u64(kind, bw, scale, scale, bw, 0, peer, party, d_input_masked, n, s);
}
