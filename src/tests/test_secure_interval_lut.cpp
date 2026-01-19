#include "suf/secure_program.hpp"
#include "suf/ref_eval.hpp"

#include <cuda_runtime.h>
#include <vector>
#include <iostream>

using namespace suf;

static SUFDescriptor make_desc() {
  SUFDescriptor d;
  d.cuts = {0, 50, 100, 150};
  d.polys.resize(4);
  d.polys[0].coeffs = {1};
  d.polys[1].coeffs = {2};
  d.polys[2].coeffs = {3};
  d.polys[3].coeffs = {4};
  return d;
}

int main() {
  const std::size_t N = 256;
  std::vector<u64> h_in(N);
  for (std::size_t i = 0; i < N; ++i) h_in[i] = static_cast<u64>(i);

  u64* d_in = nullptr;
  u64* d_out0 = nullptr;
  u64* d_out1 = nullptr;
  cudaMalloc(&d_in, N * sizeof(u64));
  cudaMalloc(&d_out0, N * sizeof(u64));
  cudaMalloc(&d_out1, N * sizeof(u64));
  cudaMemcpy(d_in, h_in.data(), N * sizeof(u64), cudaMemcpyHostToDevice);

  auto desc = make_desc();
  GpuSecureSufProgram prog0(desc, 0, 42);
  GpuSecureSufProgram prog1(desc, 1, 42);

  prog0.eval(d_in, N, d_out0, nullptr, 0);
  prog1.eval(d_in, N, d_out1, nullptr, 0);
  cudaDeviceSynchronize();

  std::vector<u64> h_out0(N), h_out1(N);
  cudaMemcpy(h_out0.data(), d_out0, N * sizeof(u64), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_out1.data(), d_out1, N * sizeof(u64), cudaMemcpyDeviceToHost);

  for (std::size_t i = 0; i < N; ++i) {
    auto ref = eval_suf_ref(desc, h_in[i]);
    u64 got = h_out0[i] + h_out1[i];
    if (got != ref.arith) {
      std::cerr << "mismatch at " << i << " got=" << got << " exp=" << ref.arith << "\n";
      return 1;
    }
  }

  cudaFree(d_in);
  cudaFree(d_out0);
  cudaFree(d_out1);

  std::cout << "ok\n";
  return 0;
}
