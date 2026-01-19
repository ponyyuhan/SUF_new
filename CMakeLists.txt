cmake_minimum_required(VERSION 3.24)
project(SUF LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(SUF_ENABLE_CUDA "Build CUDA backend" ON)
option(SUF_ENABLE_TESTS "Build tests" ON)
option(SUF_ENABLE_BENCH "Build benchmarks" ON)

add_library(suf_core INTERFACE)
target_include_directories(suf_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if (SUF_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(suf_cuda STATIC
    src/cuda/pfss_kernels.cu
    src/cuda/gpu_backend.cu
    src/cuda/dpf_kernels.cu
    src/cuda/dcf_kernels.cu
    src/cuda/interval_lut_kernels.cu
    src/secure_program.cpp
    src/interval_lut.cpp
  )
  target_include_directories(suf_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(suf_cuda PUBLIC suf_core CUDA::cudart)
  target_compile_definitions(suf_cuda PUBLIC SUF_HAVE_CUDA=1)
  set_target_properties(suf_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
    target_compile_options(suf_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
  endif()
endif()

if (SUF_ENABLE_BENCH)
  add_executable(bench_suf_gpu src/bench/bench_suf_gpu.cpp)
  target_link_libraries(bench_suf_gpu PRIVATE suf_core $<$<BOOL:${SUF_ENABLE_CUDA}>:suf_cuda>)
  target_compile_definitions(bench_suf_gpu PRIVATE SUF_ENABLE_BENCH=1)
endif()

if (SUF_ENABLE_TESTS)
  enable_testing()
  add_executable(test_suf_ref_eval src/tests/test_suf_ref_eval.cpp)
  target_link_libraries(test_suf_ref_eval PRIVATE suf_core)
  add_test(NAME test_suf_ref_eval COMMAND test_suf_ref_eval)

  if (SUF_ENABLE_CUDA)
    add_executable(test_suf_gpu_smoke src/tests/test_suf_gpu_smoke.cpp)
    target_link_libraries(test_suf_gpu_smoke PRIVATE suf_core suf_cuda)
    add_test(NAME test_suf_gpu_smoke COMMAND test_suf_gpu_smoke)

    add_executable(test_dpf_gpu src/tests/test_dpf_gpu.cpp)
    target_link_libraries(test_dpf_gpu PRIVATE suf_core suf_cuda)
    add_test(NAME test_dpf_gpu COMMAND test_dpf_gpu)

    add_executable(test_interval_lut_sil2 src/tests/test_interval_lut_sil2.cpp)
    target_link_libraries(test_interval_lut_sil2 PRIVATE suf_core suf_cuda)
    add_test(NAME test_interval_lut_sil2 COMMAND test_interval_lut_sil2)

    add_executable(test_secure_interval_lut src/tests/test_secure_interval_lut.cpp)
    target_link_libraries(test_secure_interval_lut PRIVATE suf_core suf_cuda)
    add_test(NAME test_secure_interval_lut COMMAND test_secure_interval_lut)
  endif()
endif()
