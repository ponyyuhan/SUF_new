#include "suf/sigma_bridge.hpp"

#include "suf/ir.hpp"

#include "utils/gpu_data_types.h"
#include "utils/gpu_mem.h"
#include "utils/sigma_comms.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <random>
#include <unordered_map>
#include <vector>

struct AESGlobalContext {
  u32 *t0_g;
  u8 *Sbox_g;
  u32 *t4_0G;
  u32 *t4_1G;
  u32 *t4_2G;
  u32 *t4_3G;
};

void initAESContext(AESGlobalContext *g);

template <typename T>
struct GPUSelectKey {
  int N;
  T *a;
  T *b;
  T *c;
  T *d1;
  T *d2;
};

template <typename T>
static GPUSelectKey<T> readGPUSelectKey(uint8_t **key_as_bytes, int N) {
  GPUSelectKey<T> k;
  k.N = N;

  const std::size_t size_in_bytes = static_cast<std::size_t>(N) * sizeof(T);

  k.a = reinterpret_cast<T *>(*key_as_bytes);
  *key_as_bytes += size_in_bytes;

  k.b = reinterpret_cast<T *>(*key_as_bytes);
  *key_as_bytes += size_in_bytes;

  k.c = reinterpret_cast<T *>(*key_as_bytes);
  *key_as_bytes += size_in_bytes;

  k.d1 = reinterpret_cast<T *>(*key_as_bytes);
  *key_as_bytes += size_in_bytes;

  k.d2 = reinterpret_cast<T *>(*key_as_bytes);
  *key_as_bytes += size_in_bytes;

  return k;
}

struct GPUSSTabKey {
  int bin;
  int N;
  u8 *ss;
  u64 memSzSS;
  u64 memSzOut;
};

struct GPUDPFTreeKey {
  int bin;
  int N;
  int evalAll;
  AESBlock *scw;
  AESBlock *l0;
  AESBlock *l1;
  u32 *tR;
  u64 szScw;
  u64 memSzScw;
  u64 memSzL;
  u64 memSzT;
  u64 memSzOut;
};

struct GPUDPFKey {
  int bin;
  int M;
  int B;
  u64 memSzOut;
  GPUDPFTreeKey *dpfTreeKey;
  GPUSSTabKey ssKey;
};

GPUDPFKey readGPUDPFKey(u8 **key_as_bytes);

template <typename T>
struct GPULUTKey {
  int bout;
  GPUDPFKey k;
  u32 *maskU;
  GPUSelectKey<T> s;
};

template <typename T>
static GPULUTKey<T> readGPULUTKey(uint8_t **key_as_bytes) {
  GPULUTKey<T> l;
  l.bout = static_cast<int>(**key_as_bytes);
  *key_as_bytes += sizeof(int);
  l.k = readGPUDPFKey(reinterpret_cast<u8 **>(key_as_bytes));
  l.maskU = reinterpret_cast<u32 *>(*key_as_bytes);
  *key_as_bytes += l.k.memSzOut;
  l.s = readGPUSelectKey<T>(key_as_bytes, l.k.M);
  return l;
}

template <typename TIn, typename TOut>
TOut *gpuKeyGenLUT(uint8_t **key_as_bytes, int party, int bin, int bout, int N,
                   TIn *d_rin, AESGlobalContext *gaes);

template <typename TIn, typename TOut>
TOut *gpuDpfLUT(GPULUTKey<TOut> k0, SigmaPeer *peer, int party, TIn *d_X, TOut *d_tab,
               AESGlobalContext *g, Stats *s, bool opMasked = true);

extern template u64 *gpuKeyGenLUT<u16, u64>(uint8_t **key_as_bytes, int party, int bin, int bout, int N,
                                           u16 *d_rin, AESGlobalContext *gaes);
extern template u64 *gpuDpfLUT<u16, u64>(GPULUTKey<u64> k0, SigmaPeer *peer, int party, u16 *d_X,
                                        u64 *d_tab, AESGlobalContext *g, Stats *s, bool opMasked);

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
std::uint8_t** g_keybuf_ptr = nullptr;
AESGlobalContext g_aes{};
bool g_aes_ready = false;

void ensure_aes_ready() {
  if (!g_aes_ready) {
    initAESContext(&g_aes);
    g_aes_ready = true;
  }
}

int env_int(const char* name, int fallback) {
  const char* v = std::getenv(name);
  if (!v || !*v) return fallback;
  return std::atoi(v);
}

bool env_enabled(const char* name) {
  const char* v = std::getenv(name);
  return v && *v && std::atoi(v) != 0;
}


double env_double(const char* name, double fallback) {
  const char* v = std::getenv(name);
  if (!v || !*v) return fallback;
  char* end = nullptr;
  const double val = std::strtod(v, &end);
  if (end == v) return fallback;
  return val;
}

} // namespace

extern "C" void suf_sigma_set_keybuf_ptr(std::uint8_t** keybuf_ptr) {
  g_keybuf_ptr = keybuf_ptr;
}

extern "C" bool suf_softmax_enabled() {
  return env_enabled("SUF_SOFTMAX") || env_enabled("SUF_NONLINEAR");
}

extern "C" bool suf_layernorm_enabled() {
  return env_enabled("SUF_LAYERNORM") || env_enabled("SUF_NONLINEAR");
}

namespace {

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

std::uint64_t mul_mod_bw(std::uint64_t a, std::uint64_t b, std::uint64_t mask) {
  const unsigned __int128 prod = static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
  return static_cast<std::uint64_t>(prod) & mask;
}

std::vector<std::vector<std::uint64_t>> build_binom_mod(int degree, std::uint64_t mask) {
  std::vector<std::vector<std::uint64_t>> binom(static_cast<std::size_t>(degree + 1),
                                                std::vector<std::uint64_t>(static_cast<std::size_t>(degree + 1), 0));
  binom[0][0] = 1;
  for (int k = 1; k <= degree; ++k) {
    binom[k][0] = 1;
    binom[k][k] = 1;
    for (int i = 1; i < k; ++i) {
      binom[k][i] = (binom[k - 1][i - 1] + binom[k - 1][i]) & mask;
    }
  }
  return binom;
}

std::vector<std::uint64_t> build_pow_neg_r_mod(int degree, std::uint64_t r, std::uint64_t mask) {
  std::vector<std::uint64_t> pow(static_cast<std::size_t>(degree + 1), 0);
  pow[0] = 1;
  const std::uint64_t r_neg = (0ULL - r) & mask;
  for (int i = 1; i <= degree; ++i) {
    pow[static_cast<std::size_t>(i)] = mul_mod_bw(pow[static_cast<std::size_t>(i - 1)], r_neg, mask);
  }
  return pow;
}

std::vector<std::uint64_t> shift_poly_coeffs_mod(const std::vector<std::uint64_t>& coeffs,
                                                 int degree,
                                                 const std::vector<std::vector<std::uint64_t>>& binom,
                                                 const std::vector<std::uint64_t>& pow_neg_r,
                                                 std::uint64_t mask) {
  std::vector<std::uint64_t> out(static_cast<std::size_t>(degree + 1), 0);
  const int max_k = std::min<int>(degree, static_cast<int>(coeffs.size()) - 1);
  for (int k = 0; k <= max_k; ++k) {
    const std::uint64_t c = coeffs[static_cast<std::size_t>(k)] & mask;
    if (c == 0) continue;
    for (int i = 0; i <= k; ++i) {
      const std::uint64_t term0 = mul_mod_bw(c, binom[k][i], mask);
      const std::uint64_t term = mul_mod_bw(term0, pow_neg_r[static_cast<std::size_t>(k - i)], mask);
      out[static_cast<std::size_t>(i)] = (out[static_cast<std::size_t>(i)] + term) & mask;
    }
  }
  return out;
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
    long double y_real = 0.0L;
    if (key.kind == GateKind::Gelu || key.kind == GateKind::Silu) {
      const std::uint64_t sign_bit = (key.in_bits >= 64) ? (1ULL << 63) : (1ULL << (key.in_bits - 1));
      const __int128 domain = (__int128)1 << key.in_bits;
      __int128 signed_x = (x_fixed & sign_bit) ? (static_cast<__int128>(x_fixed) - domain)
                                               : static_cast<__int128>(x_fixed);
      const long double x_real = static_cast<long double>(signed_x) / scale_in;
      if (key.kind == GateKind::Silu) {
        y_real = x_real / (1.0L + std::exp(-x_real));
      } else {
        const long double t = x_real / std::sqrt(2.0L);
        y_real = x_real * 0.5L * (1.0L + std::erf(t));
      }
    } else {
      if (x_fixed < key.clamp_min) x_fixed = key.clamp_min;
      if (x_fixed > key.clamp_max) x_fixed = key.clamp_max;
      const long double x_real = static_cast<long double>(x_fixed) / scale_in;
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

struct SufGateState {
  GateKind kind = GateKind::Gelu;
  int bw = 0;
  int scale = 0;
  int in_bits = 0;
  int scale_in = 0;
  std::uint64_t extra = 0;
  std::size_t n = 0;
  std::uint8_t* key_bytes = nullptr;
  std::size_t key_bytes_len = 0;
  std::uint64_t* d_table = nullptr;
};

std::vector<SufGateState> g_suf_gates;
std::size_t g_suf_eval_idx = 0;
std::size_t g_suf_key_idx = 0;
std::mt19937_64 g_master_rng(0x53C0FFEEu);

std::uint64_t next_gate_seed() {
  return g_master_rng();
}

__device__ __forceinline__ std::uint64_t mod_pow2_dev(std::uint64_t v, int bw) {
  if (bw >= 64) return v;
  return v & ((std::uint64_t(1) << bw) - 1ULL);
}

__global__ void kernel_u64_to_u16(const std::uint64_t* in,
                                  std::uint16_t* out,
                                  int in_bits,
                                  std::size_t n) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  std::uint64_t v = in[idx];
  if (in_bits < 16) {
    v &= (std::uint64_t(1) << in_bits) - 1ULL;
  }
  out[idx] = static_cast<std::uint16_t>(v);
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
  if (!g_keybuf_ptr || !*g_keybuf_ptr) return nullptr;
  ensure_aes_ready();

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

  std::uint64_t* d_table = reinterpret_cast<std::uint64_t*>(gpuMalloc(table.size() * sizeof(std::uint64_t)));
  cudaMemcpy(d_table, table.data(), table.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice);

  std::uint16_t* d_input_u16 = reinterpret_cast<std::uint16_t*>(gpuMalloc(n * sizeof(std::uint16_t)));
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_u64_to_u16<<<blocks, threads>>>(d_input_mask, d_input_u16, in_bits, n);
  cudaDeviceSynchronize();

  std::uint8_t* key_begin = *g_keybuf_ptr;
  auto d_out = gpuKeyGenLUT<u16, u64>(g_keybuf_ptr, party,
                                      in_bits, bw_out,
                                      static_cast<int>(n),
                                      d_input_u16,
                                      &g_aes);
  gpuFree(d_input_u16);
  const std::size_t key_size = static_cast<std::size_t>(*g_keybuf_ptr - key_begin);

  SufGateState state;
  state.kind = kind;
  state.bw = bw_out;
  state.scale = scale_out;
  state.in_bits = in_bits;
  state.scale_in = scale_in;
  state.extra = extra;
  state.n = n;
  state.key_bytes = key_begin;
  state.key_bytes_len = key_size;
  state.d_table = d_table;
  g_suf_gates.push_back(std::move(state));
  if (env_enabled("SUF_DEBUG")) {
    std::fprintf(stderr,
                 "[suf] keygen gate kind=%d bw=%d scale=%d in_bits=%d scale_in=%d extra=%llu n=%zu key_bytes=%zu\n",
                 static_cast<int>(kind), bw_out, scale_out, in_bits, scale_in,
                 static_cast<unsigned long long>(extra), n, key_size);
  }

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
  if (!g_keybuf_ptr || !*g_keybuf_ptr) return nullptr;
  ensure_aes_ready();

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

  std::uint64_t* d_table = reinterpret_cast<std::uint64_t*>(gpuMalloc(table.size() * sizeof(std::uint64_t)));
  cudaMemcpy(d_table, table.data(), table.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice);

  std::uint8_t* key_begin = *g_keybuf_ptr;
  auto d_out = gpuKeyGenLUT<std::uint16_t, std::uint64_t>(g_keybuf_ptr, party,
                                                          in_bits, bw_out,
                                                          static_cast<int>(n),
                                                          const_cast<std::uint16_t*>(d_input_mask),
                                                          &g_aes);
  const std::size_t key_size = static_cast<std::size_t>(*g_keybuf_ptr - key_begin);

  SufGateState state;
  state.kind = kind;
  state.bw = bw_out;
  state.scale = scale_out;
  state.in_bits = in_bits;
  state.scale_in = scale_in;
  state.extra = extra;
  state.n = n;
  state.key_bytes = key_begin;
  state.key_bytes_len = key_size;
  state.d_table = d_table;
  g_suf_gates.push_back(std::move(state));
  if (env_enabled("SUF_DEBUG")) {
    std::fprintf(stderr,
                 "[suf] keygen gate kind=%d bw=%d scale=%d in_bits=%d scale_in=%d extra=%llu n=%zu key_bytes=%zu\n",
                 static_cast<int>(kind), bw_out, scale_out, in_bits, scale_in,
                 static_cast<unsigned long long>(extra), n, key_size);
  }

  return d_out;
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
  if (g_suf_eval_idx >= g_suf_gates.size()) {
    if (env_enabled("SUF_DEBUG")) {
      std::fprintf(stderr,
                   "[suf] eval out of range idx=%zu size=%zu kind=%d n=%zu\n",
                   g_suf_eval_idx, g_suf_gates.size(), static_cast<int>(expected_kind), n);
    }
    return nullptr;
  }
  auto& gate = g_suf_gates[g_suf_eval_idx++];
  if (env_enabled("SUF_DEBUG")) {
    std::fprintf(stderr,
                 "[suf] eval gate idx=%zu kind=%d bw=%d scale=%d in_bits=%d scale_in=%d extra=%llu n=%zu\n",
                 g_suf_eval_idx - 1, static_cast<int>(gate.kind), gate.bw, gate.scale,
                 gate.in_bits, gate.scale_in,
                 static_cast<unsigned long long>(gate.extra), gate.n);
  }
  if (gate.kind != expected_kind || gate.n != n || gate.bw != bw_out ||
      gate.scale != scale_out || gate.scale_in != scale_in ||
      gate.in_bits != in_bits || gate.extra != extra) {
    if (env_enabled("SUF_DEBUG")) {
      std::fprintf(stderr,
                   "[suf] eval mismatch kind=%d/%d bw=%d/%d scale=%d/%d in_bits=%d/%d scale_in=%d/%d extra=%llu/%llu n=%zu/%zu\n",
                   static_cast<int>(gate.kind), static_cast<int>(expected_kind),
                   gate.bw, bw_out, gate.scale, scale_out, gate.in_bits, in_bits,
                   gate.scale_in, scale_in,
                   static_cast<unsigned long long>(gate.extra),
                   static_cast<unsigned long long>(extra),
                   gate.n, n);
    }
    return nullptr;
  }
  ensure_aes_ready();
  peer->reconstructInPlace(const_cast<std::uint64_t*>(d_input_masked), gate.in_bits, n, s);

  std::uint8_t* key_ptr = gate.key_bytes;
  auto lut_key = readGPULUTKey<std::uint64_t>(&key_ptr);
  std::uint16_t* d_input_u16 = reinterpret_cast<std::uint16_t*>(gpuMalloc(n * sizeof(std::uint16_t)));
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_u64_to_u16<<<blocks, threads>>>(d_input_masked, d_input_u16, gate.in_bits, n);
  cudaDeviceSynchronize();
  auto d_out = gpuDpfLUT<std::uint16_t, std::uint64_t>(lut_key, peer, party,
                                                       d_input_u16,
                                                       gate.d_table, &g_aes, s);
  gpuFree(d_input_u16);
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
  if (g_suf_eval_idx >= g_suf_gates.size()) {
    if (env_enabled("SUF_DEBUG")) {
      std::fprintf(stderr,
                   "[suf] eval out of range idx=%zu size=%zu kind=%d n=%zu\n",
                   g_suf_eval_idx, g_suf_gates.size(), static_cast<int>(expected_kind), n);
    }
    return nullptr;
  }
  auto& gate = g_suf_gates[g_suf_eval_idx++];
  if (env_enabled("SUF_DEBUG")) {
    std::fprintf(stderr,
                 "[suf] eval gate idx=%zu kind=%d bw=%d scale=%d in_bits=%d scale_in=%d extra=%llu n=%zu\n",
                 g_suf_eval_idx - 1, static_cast<int>(gate.kind), gate.bw, gate.scale,
                 gate.in_bits, gate.scale_in,
                 static_cast<unsigned long long>(gate.extra), gate.n);
  }
  if (gate.kind != expected_kind || gate.n != n || gate.bw != bw_out ||
      gate.scale != scale_out || gate.scale_in != scale_in ||
      gate.in_bits != in_bits || gate.extra != extra) {
    if (env_enabled("SUF_DEBUG")) {
      std::fprintf(stderr,
                   "[suf] eval mismatch kind=%d/%d bw=%d/%d scale=%d/%d in_bits=%d/%d scale_in=%d/%d extra=%llu/%llu n=%zu/%zu\n",
                   static_cast<int>(gate.kind), static_cast<int>(expected_kind),
                   gate.bw, bw_out, gate.scale, scale_out, gate.in_bits, in_bits,
                   gate.scale_in, scale_in,
                   static_cast<unsigned long long>(gate.extra),
                   static_cast<unsigned long long>(extra),
                   gate.n, n);
    }
    return nullptr;
  }
  ensure_aes_ready();
  peer->reconstructInPlace(const_cast<std::uint16_t*>(d_input_masked), gate.in_bits, n, s);

  std::uint8_t* key_ptr = gate.key_bytes;
  auto lut_key = readGPULUTKey<std::uint64_t>(&key_ptr);
  auto d_out = gpuDpfLUT<std::uint16_t, std::uint64_t>(lut_key, peer, party,
                                                       const_cast<std::uint16_t*>(d_input_masked),
                                                       gate.d_table, &g_aes, s);
  return d_out;
}

} // namespace

extern "C" void suf_sigma_reset_keygen() {
  for (auto& gate : g_suf_gates) {
    if (gate.d_table) gpuFree(gate.d_table);
  }
  g_suf_gates.clear();
  g_suf_eval_idx = 0;
  g_suf_key_idx = 0;
  g_master_rng.seed(0x53C0FFEEu);
}

extern "C" void suf_sigma_reset_eval() {
  g_suf_eval_idx = 0;
  g_suf_key_idx = 0;
}

extern "C" void suf_sigma_clear() {
  for (auto& gate : g_suf_gates) {
    if (gate.d_table) gpuFree(gate.d_table);
  }
  g_suf_gates.clear();
  g_suf_eval_idx = 0;
  g_suf_key_idx = 0;
}

extern "C" void suf_sigma_consume_key() {
  if (!g_keybuf_ptr || !*g_keybuf_ptr) return;
  if (g_suf_key_idx >= g_suf_gates.size()) {
    if (env_enabled("SUF_DEBUG")) {
      std::fprintf(stderr,
                   "[suf] consume out of range idx=%zu size=%zu\n",
                   g_suf_key_idx, g_suf_gates.size());
    }
    return;
  }
  const auto& gate = g_suf_gates[g_suf_key_idx++];
  if (env_enabled("SUF_DEBUG")) {
    std::fprintf(stderr,
                 "[suf] consume key idx=%zu bytes=%zu\n",
                 g_suf_key_idx - 1, gate.key_bytes_len);
  }
  *g_keybuf_ptr = gate.key_bytes + gate.key_bytes_len;
}

extern "C" std::uint64_t* suf_sigma_keygen_activation(int party,
                                                       int bw,
                                                       int scale,
                                                       bool silu,
                                                       const std::uint64_t* d_input_mask,
                                                       std::size_t n) {
  const int intervals = env_int(silu ? "SUF_SILU_INTERVALS" : "SUF_GELU_INTERVALS",
                                silu ? 1024 : 256);
  const int default_bits = bits_needed(static_cast<std::uint64_t>(intervals - 1));
  const int in_bits = env_int(silu ? "SUF_SILU_BITS" : "SUF_GELU_BITS", default_bits);
  const std::uint64_t clamp_max = mask_for_bw(in_bits);
  return keygen_table_gate_u64(silu ? GateKind::Silu : GateKind::Gelu,
                               bw, scale, in_bits, scale,
                               0, clamp_max, 0, party, d_input_mask, n);
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
  const int max_bits = std::max(1, std::min(16, bw - shift));
  const int scale_in = 2 * scale - shift;
  const double vmax_real = env_double("SUF_RSQRT_VMAX", 16.0);
  const double eps_real = env_double("SUF_RSQRT_EPS", 0.0);
  const std::uint64_t clamp_min = std::max<std::uint64_t>(1, static_cast<std::uint64_t>(llroundl(eps_real * (1ULL << scale_in))));
  const std::uint64_t vmax_fixed = static_cast<std::uint64_t>(llroundl(vmax_real * (1ULL << scale_in)));
  int in_bits = env_int("SUF_RSQRT_BITS", 0);
  if (in_bits <= 0) {
    in_bits = bits_needed(vmax_fixed);
    in_bits = std::max(8, std::min(max_bits, in_bits));
  } else {
    in_bits = std::max(1, std::min(max_bits, in_bits));
  }
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
  const int max_bits = std::max(1, std::min(16, bw - shift));
  const int scale_in = 2 * scale - shift;
  const double vmax_real = env_double("SUF_RSQRT_VMAX", 16.0);
  const std::uint64_t vmax_fixed = static_cast<std::uint64_t>(llroundl(vmax_real * (1ULL << scale_in)));
  int in_bits = env_int("SUF_RSQRT_BITS", 0);
  if (in_bits <= 0) {
    in_bits = bits_needed(vmax_fixed);
    in_bits = std::max(8, std::min(max_bits, in_bits));
  } else {
    in_bits = std::max(1, std::min(max_bits, in_bits));
  }
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
  const int intervals = env_int(silu ? "SUF_SILU_INTERVALS" : "SUF_GELU_INTERVALS",
                                silu ? 1024 : 256);
  const int default_bits = bits_needed(static_cast<std::uint64_t>(intervals - 1));
  const int in_bits = env_int(silu ? "SUF_SILU_BITS" : "SUF_GELU_BITS", default_bits);
  const GateKind kind = silu ? GateKind::Silu : GateKind::Gelu;
  auto out = eval_gate_u64(kind, bw, scale, scale, in_bits, 0, peer, party, d_input_masked, n, s);
  if (out) {
    suf_sigma_consume_key();
  }
  return out;
}
