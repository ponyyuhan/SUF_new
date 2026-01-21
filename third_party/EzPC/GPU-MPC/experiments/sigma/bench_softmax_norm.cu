// Benchmark softmax/layernorm blocks with optional SUF paths.
#include "utils/gpu_data_types.h"
#include "utils/gpu_file_utils.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"
#include "utils/gpu_comms.h"
#include "utils/misc_utils.h"

#include "fss/gpu_softmax.h"
#include "fss/gpu_layernorm.h"
#include "suf/sigma_bridge.hpp"
#include "backend/sigma.h"

#include <llama/api.h>
#include <sytorch/utils.h>
#include <sytorch/backend/llama_transformer.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using T = u64;

struct BenchArgs {
    std::string block = "softmax";
    int party = 0;
    std::string ip = "127.0.0.1";
    int L = 128;
    int H = 12;
    int B = 1;
    int d_model = 768;
    int iters = 30;
    int warmup = 5;
    int bw = 50;
    int scale = 12;
    bool triangular = false;
    bool pin = false;
    int keybuf_gb = 4;
    bool json = true;
    bool manual_layernorm = false;
    int port = 42003;
};

static void usage() {
    printf("bench_softmax_norm --block softmax|layernorm --party P --ip IP --L L --H H --B B --d_model D\n");
    printf("  [--iters N] [--warmup W] [--bw BW] [--scale S] [--triangular 0|1] [--keybuf_gb G] [--pin 0|1]\n");
    printf("  [--json 0|1] [--manual_layernorm 0|1] [--port P]\n");
}

static BenchArgs parse_args(int argc, char** argv) {
    BenchArgs args;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--block") && i + 1 < argc) args.block = argv[++i];
        else if (!std::strcmp(argv[i], "--party") && i + 1 < argc) args.party = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--ip") && i + 1 < argc) args.ip = argv[++i];
        else if (!std::strcmp(argv[i], "--L") && i + 1 < argc) args.L = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--H") && i + 1 < argc) args.H = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--B") && i + 1 < argc) args.B = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--d_model") && i + 1 < argc) args.d_model = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) args.iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) args.warmup = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--bw") && i + 1 < argc) args.bw = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--scale") && i + 1 < argc) args.scale = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--triangular") && i + 1 < argc) args.triangular = std::atoi(argv[++i]) != 0;
        else if (!std::strcmp(argv[i], "--keybuf_gb") && i + 1 < argc) args.keybuf_gb = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--pin") && i + 1 < argc) args.pin = std::atoi(argv[++i]) != 0;
        else if (!std::strcmp(argv[i], "--json") && i + 1 < argc) args.json = std::atoi(argv[++i]) != 0;
        else if (!std::strcmp(argv[i], "--manual_layernorm") && i + 1 < argc) args.manual_layernorm = std::atoi(argv[++i]) != 0;
        else if (!std::strcmp(argv[i], "--port") && i + 1 < argc) args.port = std::atoi(argv[++i]);
        else {
            fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            usage();
            std::exit(1);
        }
    }
    return args;
}

static double median(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t mid = v.size() / 2;
    if (v.size() % 2 == 0) return 0.5 * (v[mid - 1] + v[mid]);
    return v[mid];
}

static std::vector<GroupElement> build_inv_sqrt_lut(int n_embed, int scale) {
    std::vector<GroupElement> lut(1LL << 13);
#pragma omp parallel for
    for (int i = 0; i < (1LL << 13); ++i) {
        GroupElement k = i % (1LL << 6);
        GroupElement m = i >> 6;
        double val = double(m + 128) * std::pow(2.0, k - 7);
        lut[i] = GroupElement(double(1LL << (2 * scale)) / std::sqrt(val / n_embed));
    }
    return lut;
}

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);
    if (args.B <= 0 || args.L <= 0 || args.H <= 0) {
        fprintf(stderr, "Invalid shape params.\n");
        return 1;
    }

    const bool use_softmax = (args.block == "softmax");
    const bool use_layernorm = (args.block == "layernorm");
    if (!use_softmax && !use_layernorm) {
        fprintf(stderr, "Unknown block: %s\n", args.block.c_str());
        return 1;
    }
    const bool need_llama = use_layernorm && !suf_layernorm_enabled() && !args.manual_layernorm;
    LlamaTransformer<u64>* llama_keygen = nullptr;
    u8* llamaBuf1 = nullptr;
    u8* dummyBuf1 = nullptr;
    u8* llamaBuf2 = nullptr;
    u8* dummyBuf2 = nullptr;

    AESGlobalContext g;
    initAESContext(&g);
    initGPURandomness();
    sytorch_init();
    LlamaConfig::bitlength = args.bw;
    LlamaConfig::stochasticT = false;
    LlamaConfig::stochasticRT = false;

    auto peer = new GpuPeer(true);
    bool peer_connected = false;
    if (!need_llama) {
        peer->connect(args.party, args.ip, args.port);
        peer_connected = true;
    }
    Stats s;
    s.reset();

    u8* startPtr = nullptr;
    u8* curPtr = nullptr;
    const size_t keyBufBytes = static_cast<size_t>(args.keybuf_gb) * OneGB;
    getKeyBuf(&startPtr, &curPtr, keyBufBytes, args.pin);
    if (suf_softmax_enabled() || suf_layernorm_enabled()) {
        suf_sigma_reset_keygen();
        suf_sigma_set_keybuf_ptr(&curPtr);
    }

    double key_bytes = 0.0;
    double llama_key_bytes = 0.0;
    std::vector<double> times_ms;
    std::vector<double> comm_bytes;

    if (use_softmax) {
        MaxpoolParams p;
        p.bw = args.bw;
        p.bin = args.bw - args.scale;
        p.scale = args.scale;
        p.scaleDiv = 0;
        p.bwBackprop = 0;
        p.N = args.B * args.H;
        p.imgH = args.L;
        p.imgW = args.L;
        p.C = 1;
        p.FH = 1;
        p.FW = p.imgW;
        p.strideH = 1;
        p.strideW = p.FW;
        p.zPadHLeft = 0;
        p.zPadHRight = 0;
        p.zPadWLeft = 0;
        p.zPadWRight = 0;
        p.isLowerTriangular = args.triangular;
        initPoolParams(p);

        int inSz = getInSz(p);
        auto d_mask_I = randomGEOnGpu<T>(inSz, p.bin);
        T* h_I = nullptr;
        auto d_masked_I = getMaskedInputOnGpu(inSz, p.bw, d_mask_I, &h_I, true, 15);

        auto d_mask_O = gpuKeygenSoftmax(&curPtr, args.party, p, d_mask_I, &g);
        key_bytes = static_cast<double>(curPtr - startPtr);
        gpuFree(d_mask_O);

        u8* readPtr = startPtr;
        if (suf_softmax_enabled()) {
            suf_sigma_set_keybuf_ptr(&readPtr);
            suf_sigma_reset_eval();
        }
        auto k = readGPUSoftMaxKey<T>(p, &readPtr);
        auto d_nExpMsbTab = genLUT<T, nExpMsb<T>>(8, 4, p.scale);
        auto d_nExpLsbTab = genLUT<T, nExpLsb<T>>(8, 12, p.scale);
        auto d_invTab = genLUT<T, inv<T>>(int(ceil(log2(p.FW))) + p.scale, 6, p.scale);

        // warmup
        for (int i = 0; i < args.warmup; ++i) {
            if (suf_softmax_enabled()) {
                suf_sigma_reset_eval();
            }
            auto d_O = gpuSoftmax(peer, args.party, p, k, d_masked_I, d_nExpMsbTab, d_nExpLsbTab, d_invTab, &g, &s);
            gpuFree(d_O);
        }

        for (int i = 0; i < args.iters; ++i) {
            peer->sync();
            auto start_comm = peer->bytesSent() + peer->bytesReceived();
            auto start = std::chrono::high_resolution_clock::now();
            if (suf_softmax_enabled()) {
                suf_sigma_reset_eval();
            }
            auto d_O = gpuSoftmax(peer, args.party, p, k, d_masked_I, d_nExpMsbTab, d_nExpLsbTab, d_invTab, &g, &s);
            auto end = std::chrono::high_resolution_clock::now();
            auto end_comm = peer->bytesSent() + peer->bytesReceived();
            gpuFree(d_O);
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            times_ms.push_back(ms);
            comm_bytes.push_back(static_cast<double>(end_comm - start_comm));
        }

        gpuFree(d_masked_I);
        gpuFree(d_mask_I);
    } else if (use_layernorm) {
        if (need_llama) {
            const int n_seq = args.B * args.L;
            const int n_embed = args.d_model;
            Tensor<u64> X({static_cast<u64>(n_seq), static_cast<u64>(n_embed)});
            Tensor<u64> Y({static_cast<u64>(n_seq), static_cast<u64>(n_embed)});
            Tensor1D<u64> A(n_embed);
            Tensor1D<u64> B(n_embed);
            X.zero();
            A.fill(0);
            B.fill(0);

            const size_t keyBufSz = static_cast<size_t>(args.keybuf_gb) * OneGB;
            auto keygen = new SIGMAKeygen<u64>(args.party, args.bw, args.scale, "", keyBufSz);
            X.d_data = (u64 *)moveToGPU((u8 *)X.data, X.size() * sizeof(u64), (Stats *)NULL);
            keygen->layernorm(A, B, X, Y, args.scale);
            keygen->close();
            key_bytes = static_cast<double>(keygen->keySize);

            auto sigma = new SIGMA<u64>(args.party, args.ip, "", args.bw, args.scale, n_seq, n_embed, 1, false);
            sigma->keyBuf = keygen->startPtr;
            sigma->startPtr = sigma->keyBuf;
            sigma->keySize = keygen->keySize;
            sigma->peer->sync();

            for (int i = 0; i < args.warmup; ++i) {
                sigma->keyBuf = sigma->startPtr;
                sigma->layernorm(A, B, X, Y, args.scale);
            }

            for (int i = 0; i < args.iters; ++i) {
                sigma->keyBuf = sigma->startPtr;
                sigma->peer->sync();
                auto start_comm = sigma->peer->bytesSent() + sigma->peer->bytesReceived();
                auto start = std::chrono::high_resolution_clock::now();
                sigma->layernorm(A, B, X, Y, args.scale);
                auto end = std::chrono::high_resolution_clock::now();
                auto end_comm = sigma->peer->bytesSent() + sigma->peer->bytesReceived();
                double ms = std::chrono::duration<double, std::milli>(end - start).count();
                times_ms.push_back(ms);
                comm_bytes.push_back(static_cast<double>(end_comm - start_comm));
            }

            sigma->close();
            gpuFree(X.d_data);
        } else {
            AvgPoolParams p;
            p.bw = args.bw;
            p.bin = args.bw - args.scale;
            p.scale = args.scale;
            p.scaleDiv = 0;
            p.bwBackprop = 0;
            p.N = 1;
            p.imgH = args.B * args.L;
            p.imgW = args.d_model;
            p.C = 1;
            p.FH = 1;
            p.FW = p.imgW;
            p.strideH = 1;
            p.strideW = p.FW;
            p.zPadHLeft = 0;
            p.zPadHRight = 0;
            p.zPadWLeft = 0;
            p.zPadWRight = 0;
            initPoolParams(p);

            const int inSz = getInSz(p);
            auto d_mask_I = randomGEOnGpu<T>(inSz, p.bin);
            auto d_mask_A = randomGEOnGpu<T>(p.imgW, p.bin);
            auto d_mask_B = randomGEOnGpu<T>(p.imgW, p.bin);
            T* h_I = nullptr;
            T* h_A = nullptr;
            T* h_B = nullptr;
            auto d_masked_I = getMaskedInputOnGpu(inSz, p.bw, d_mask_I, &h_I, true, 15);
            auto d_masked_A = getMaskedInputOnGpu(p.imgW, p.bw, d_mask_A, &h_A, true, 15);
            auto d_masked_B = getMaskedInputOnGpu(p.imgW, p.bw, d_mask_B, &h_B, true, 15);

            auto d_mask_O = gpuKeygenLayerNorm(&curPtr, args.party, p, d_mask_A, d_mask_B, d_mask_I, &g, true);
            key_bytes = static_cast<double>(curPtr - startPtr);
            gpuFree(d_mask_O);

            u8* readPtr = startPtr;
            if (suf_layernorm_enabled()) {
                suf_sigma_set_keybuf_ptr(&readPtr);
                suf_sigma_reset_eval();
            }
            auto k = readGPULayerNormKey<T>(p, &readPtr, true);
            auto invSqrtTab = build_inv_sqrt_lut(p.imgW, args.scale);

            for (int i = 0; i < args.warmup; ++i) {
                if (suf_layernorm_enabled()) {
                    suf_sigma_reset_eval();
                }
                auto d_O = gpuLayerNorm(peer, args.party, p, k, d_masked_A, d_masked_B, d_masked_I,
                                        &invSqrtTab, &g, &s, true);
                gpuFree(d_O);
            }

            for (int i = 0; i < args.iters; ++i) {
                peer->sync();
                auto start_comm = peer->bytesSent() + peer->bytesReceived();
                auto start = std::chrono::high_resolution_clock::now();
                if (suf_layernorm_enabled()) {
                    suf_sigma_reset_eval();
                }
                auto d_O = gpuLayerNorm(peer, args.party, p, k, d_masked_A, d_masked_B, d_masked_I,
                                        &invSqrtTab, &g, &s, true);
                auto end = std::chrono::high_resolution_clock::now();
                auto end_comm = peer->bytesSent() + peer->bytesReceived();
                gpuFree(d_O);
                double ms = std::chrono::duration<double, std::milli>(end - start).count();
                times_ms.push_back(ms);
                comm_bytes.push_back(static_cast<double>(end_comm - start_comm));
            }

            gpuFree(d_masked_I);
            gpuFree(d_masked_A);
            gpuFree(d_masked_B);
            gpuFree(d_mask_I);
            gpuFree(d_mask_A);
            gpuFree(d_mask_B);
        }
    }

    const double med_ms = median(times_ms);
    const double med_comm = median(comm_bytes);

    if (args.json) {
        printf("{\"block\":\"%s\",\"backend\":\"%s\",\"B\":%d,\"H\":%d,\"L\":%d,\"d_model\":%d,"
               "\"bw\":%d,\"scale\":%d,\"online_ms\":%.4f,\"comm_bytes\":%.0f,\"key_bytes_party0\":%.0f,"
               "\"iters\":%d}\n",
               args.block.c_str(),
               (suf_softmax_enabled() || suf_layernorm_enabled()) ? "suf" : "sigma",
               args.B, args.H, args.L, args.d_model, args.bw, args.scale, med_ms, med_comm, key_bytes, args.iters);
    } else {
        printf("block=%s backend=%s B=%d H=%d L=%d d_model=%d bw=%d scale=%d\n",
               args.block.c_str(),
               (suf_softmax_enabled() || suf_layernorm_enabled()) ? "suf" : "sigma",
               args.B, args.H, args.L, args.d_model, args.bw, args.scale);
        printf("median_ms=%.4f median_comm_bytes=%.0f key_bytes=%.0f iters=%d\n",
               med_ms, med_comm, key_bytes, args.iters);
    }

    if (llamaBuf1) {
        cpuFree(llamaBuf1);
    }
    if (dummyBuf1) {
        cpuFree(dummyBuf1);
    }
    if (peer_connected) {
        peer->close();
    }
    return 0;
}
