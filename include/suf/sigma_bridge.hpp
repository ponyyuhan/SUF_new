#pragma once

#include <cstddef>
#include <cstdint>

struct SigmaPeer;
struct Stats;

extern "C" void suf_sigma_reset_keygen();
extern "C" void suf_sigma_reset_eval();
extern "C" void suf_sigma_consume_key();
extern "C" void suf_sigma_clear();
extern "C" void suf_sigma_set_keybuf_ptr(std::uint8_t** keybuf_ptr);
extern "C" bool suf_softmax_enabled();
extern "C" bool suf_layernorm_enabled();

extern "C" std::uint64_t* suf_sigma_keygen_activation(int party,
                                                       int bw,
                                                       int scale,
                                                       bool silu,
                                                       const std::uint64_t* d_input_mask,
                                                       std::size_t n);

extern "C" std::uint64_t* suf_sigma_eval_activation(SigmaPeer* peer,
                                                    int party,
                                                    int bw,
                                                    int scale,
                                                    bool silu,
                                                    const std::uint64_t* d_input_masked,
                                                    std::size_t n,
                                                    Stats* s);

extern "C" std::uint64_t* suf_sigma_keygen_nexp(int party,
                                                int bw,
                                                int scale,
                                                const std::uint64_t* d_input_mask,
                                                std::size_t n);

extern "C" std::uint64_t* suf_sigma_eval_nexp(SigmaPeer* peer,
                                              int party,
                                              int bw,
                                              int scale,
                                              const std::uint64_t* d_input_masked,
                                              std::size_t n,
                                              Stats* s);

extern "C" std::uint64_t* suf_sigma_keygen_inverse(int party,
                                                   int bw,
                                                   int scale,
                                                   int nmax,
                                                   const std::uint16_t* d_input_mask,
                                                   std::size_t n);

extern "C" std::uint64_t* suf_sigma_eval_inverse(SigmaPeer* peer,
                                                 int party,
                                                 int bw,
                                                 int scale,
                                                 int nmax,
                                                 const std::uint16_t* d_input_masked,
                                                 std::size_t n,
                                                 Stats* s);

extern "C" std::uint64_t* suf_sigma_keygen_rsqrt(int party,
                                                 int bw,
                                                 int scale,
                                                 int extradiv,
                                                 const std::uint16_t* d_input_mask,
                                                 std::size_t n);

extern "C" std::uint64_t* suf_sigma_eval_rsqrt(SigmaPeer* peer,
                                               int party,
                                               int bw,
                                               int scale,
                                               int extradiv,
                                               const std::uint16_t* d_input_masked,
                                               std::size_t n,
                                               Stats* s);
