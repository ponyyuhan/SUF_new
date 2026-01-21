// Real MPC accuracy for LAMBADA next-token (GPT-2 / GPT-Neo 1.3B)
// SPDX-License-Identifier: MIT

#include <sytorch/backend/llama_extended.h>
#include <sytorch/backend/llama_transformer.h>
#include <sytorch/layers/layers.h>
#include <sytorch/module.h>
#include <llama/utils.h>
#include <llama/api.h>
#include <llama/comms.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using std::string;

static u64 get_n_seq(const std::string &filename, u64 n_embd)
{
    u64 n_bytes = std::filesystem::file_size(filename);
    always_assert(n_bytes % (4 * n_embd) == 0);
    return n_bytes / (4 * n_embd);
}

static std::vector<int> load_labels(const std::string &path)
{
    std::ifstream f(path);
    if (!f.is_open())
    {
        throw std::runtime_error("Failed to open labels file: " + path);
    }
    std::vector<int> labels;
    int v;
    while (f >> v)
    {
        labels.push_back(v);
    }
    return labels;
}

namespace gpt2_model
{
template <typename T>
class FFN : public SytorchModule<T>
{
    using SytorchModule<T>::gelu;
    u64 in;
    u64 hidden;

public:
    FC<T> *up;
    FC<T> *down;

    FFN(u64 in, u64 hidden) : in(in), hidden(hidden)
    {
        up = new FC<T>(in, hidden, true);
        down = new FC<T>(hidden, in, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        return down->forward(gelu(up->forward(input)));
    }
};

template <typename T>
class MultiHeadAttention : public SytorchModule<T>
{
    using SytorchModule<T>::split;
    using SytorchModule<T>::view;
    using SytorchModule<T>::transpose;
    using SytorchModule<T>::matmul;
    using SytorchModule<T>::matmul_triangular;
    using SytorchModule<T>::scalarmul;
    using SytorchModule<T>::softmax_triangular;
    using SytorchModule<T>::concat;

public:
    FC<T> *c_attn;
    FC<T> *c_proj;
    u64 n_heads;
    u64 n_embd;

    MultiHeadAttention(u64 n_heads, u64 n_embd) : n_heads(n_heads), n_embd(n_embd)
    {
        always_assert(n_embd % n_heads == 0);
        c_attn = new FC<T>(n_embd, 3 * n_embd, true);
        c_proj = new FC<T>(n_embd, n_embd, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &x = c_attn->forward(input);
        auto &qkv_heads = split(x, 3);
        auto &q_heads = view(qkv_heads, 0);
        auto &k_heads = view(qkv_heads, 1);
        auto &v_heads = view(qkv_heads, 2);
        auto &qs = split(q_heads, n_heads);
        auto &ks = split(k_heads, n_heads);
        auto &vs = split(v_heads, n_heads);

        double divisor = 1 / sqrt(double(n_embd) / double(n_heads));
        std::vector<Tensor<T> *> qks_sm_vs;
        for (u64 i = 0; i < n_heads; ++i)
        {
            auto &q = view(qs, i);
            auto &k = view(ks, i);
            auto &v = view(vs, i);
            auto &kt = transpose(k);
            auto &qk = matmul_triangular(q, kt);
            auto &qks = scalarmul(qk, divisor);
            auto &qks_sm = softmax_triangular(qks);
            auto &qks_sm_v = matmul(qks_sm, v);
            qks_sm_vs.push_back(&qks_sm_v);
        }
        auto &qks_sm_vs_cat = concat(qks_sm_vs);
        auto &res = c_proj->forward(qks_sm_vs_cat);
        return res;
    }
};

template <typename T>
class TransformerBlock : public SytorchModule<T>
{
    using SytorchModule<T>::add;

    MultiHeadAttention<T> *attn;
    FFN<T> *ffn;
    LayerNorm<T> *ln0;
    LayerNorm<T> *ln1;

public:
    TransformerBlock(u64 n_heads, u64 n_embd)
    {
        attn = new MultiHeadAttention<T>(n_heads, n_embd);
        ffn = new FFN<T>(n_embd, 4 * n_embd);
        ln0 = new LayerNorm<T>(n_embd);
        ln1 = new LayerNorm<T>(n_embd);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &ln0_out = ln0->forward(input);
        auto &attn_out = attn->forward(ln0_out);
        auto &attn_out_add = add(attn_out, input);
        auto &ln1_out = ln1->forward(attn_out_add);
        auto &ffn_out = ffn->forward(ln1_out);
        auto &ffn_out_add = add(ffn_out, attn_out_add);
        return ffn_out_add;
    }
};

template <typename T>
class GPT2 : public SytorchModule<T>
{
    std::vector<TransformerBlock<T> *> blocks;
    LayerNorm<T> *ln_f;
    u64 n_layer;

public:
    GPT2(u64 n_layer, u64 n_heads, u64 n_embd) : n_layer(n_layer)
    {
        for (u64 i = 0; i < n_layer; ++i)
        {
            blocks.push_back(new TransformerBlock<T>(n_heads, n_embd));
        }
        ln_f = new LayerNorm<T>(n_embd);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        Tensor<T> *x = &input;
        for (u64 i = 0; i < n_layer; ++i)
        {
            auto &block = blocks[i];
            auto &x_out = block->forward(*x);
            x = &x_out;
        }
        return ln_f->forward(*x);
    }
};

template <typename T>
class GPT2NextWordLogits : public SytorchModule<T>
{
    using SytorchModule<T>::view;
    GPT2<T> *gpt2;
    FC<T> *fc;

public:
    GPT2NextWordLogits(u64 n_layer, u64 n_heads, u64 n_embd, u64 n_vocab)
    {
        gpt2 = new GPT2<T>(n_layer, n_heads, n_embd);
        fc = new FC<T>(n_embd, n_vocab, false);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &fc_in = gpt2->forward(input);
        auto &fc_out = fc->forward(fc_in);
        return view(fc_out, -1);
    }
};
} // namespace gpt2_model

namespace gptneo_model
{
template <typename T>
class FFN : public SytorchModule<T>
{
    using SytorchModule<T>::gelu;
    u64 in;
    u64 hidden;

public:
    FC<T> *up;
    FC<T> *down;

    FFN(u64 in, u64 hidden) : in(in), hidden(hidden)
    {
        up = new FC<T>(in, hidden, true);
        down = new FC<T>(hidden, in, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        return down->forward(gelu(up->forward(input)));
    }
};

template <typename T>
class MultiHeadAttention : public SytorchModule<T>
{
    using SytorchModule<T>::split;
    using SytorchModule<T>::view;
    using SytorchModule<T>::transpose;
    using SytorchModule<T>::matmul;
    using SytorchModule<T>::softmax;
    using SytorchModule<T>::concat;
    using SytorchModule<T>::attention_mask;

public:
    FC<T> *k_attn;
    FC<T> *v_attn;
    FC<T> *q_attn;
    FC<T> *c_proj;
    u64 n_heads;
    u64 n_embd;
    u64 attention_type;
    u64 window_size;

    MultiHeadAttention(u64 n_heads, u64 n_embd, u64 attention_type, u64 window_size)
        : n_heads(n_heads), n_embd(n_embd), attention_type(attention_type), window_size(window_size)
    {
        always_assert(n_embd % n_heads == 0);
        k_attn = new FC<T>(n_embd, n_embd, false);
        v_attn = new FC<T>(n_embd, n_embd, false);
        q_attn = new FC<T>(n_embd, n_embd, false);
        c_proj = new FC<T>(n_embd, n_embd, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &k_heads = k_attn->forward(input);
        auto &v_heads = v_attn->forward(input);
        auto &q_heads = q_attn->forward(input);
        auto &qs = split(q_heads, n_heads);
        auto &ks = split(k_heads, n_heads);
        auto &vs = split(v_heads, n_heads);

        std::vector<Tensor<T> *> qks_sm_vs;
        for (u64 i = 0; i < n_heads; ++i)
        {
            auto &q = view(qs, i);
            auto &k = view(ks, i);
            auto &v = view(vs, i);
            auto &kt = transpose(k);
            auto &qks = matmul(q, kt);

            auto &qks_masked = attention_mask(qks, 10000000.0);
            Tensor<T> *masked = &qks_masked;
            auto &qks_sm = softmax(*masked);
            auto &qks_sm_v = matmul(qks_sm, v);
            qks_sm_vs.push_back(&qks_sm_v);
        }

        auto &qks_sm_vs_cat = concat(qks_sm_vs);
        auto &res = c_proj->forward(qks_sm_vs_cat);
        return res;
    }
};

template <typename T>
class TransformerBlock : public SytorchModule<T>
{
    using SytorchModule<T>::add;

    MultiHeadAttention<T> *attn;
    FFN<T> *ffn;
    LayerNorm<T> *ln0;
    LayerNorm<T> *ln1;
    u64 n_layer;
    u64 attention_type;
    u64 window_size;

public:
    TransformerBlock(u64 n_heads, u64 n_embd, u64 attention_type, u64 window_size)
        : attention_type(attention_type), window_size(window_size)
    {
        attn = new MultiHeadAttention<T>(n_heads, n_embd, attention_type, window_size);
        ffn = new FFN<T>(n_embd, 4 * n_embd);
        ln0 = new LayerNorm<T>(n_embd);
        ln1 = new LayerNorm<T>(n_embd);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &ln0_out = ln0->forward(input);
        auto &attn_out = attn->forward(ln0_out);
        auto &attn_out_add = add(attn_out, input);
        auto &ln1_out = ln1->forward(attn_out_add);
        auto &ffn_out = ffn->forward(ln1_out);
        auto &ffn_out_add = add(ffn_out, attn_out_add);
        return ffn_out_add;
    }
};

template <typename T>
class GPTNeo : public SytorchModule<T>
{
    std::vector<TransformerBlock<T> *> blocks;
    LayerNorm<T> *ln_f;
    u64 n_layer;
    u64 window_size;

public:
    GPTNeo(u64 n_layer, u64 n_heads, u64 n_embd, u64 window_size) : n_layer(n_layer), window_size(window_size)
    {
        for (u64 i = 0; i < n_layer; ++i)
        {
            blocks.push_back(new TransformerBlock<T>(n_heads, n_embd, i, window_size));
        }
        ln_f = new LayerNorm<T>(n_embd);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        Tensor<T> *x = &input;
        for (u64 i = 0; i < n_layer; ++i)
        {
            auto &block = blocks[i];
            auto &x_out = block->forward(*x);
            x = &x_out;
        }
        return ln_f->forward(*x);
    }
};

template <typename T>
class GPTNeoNextWordLogits : public SytorchModule<T>
{
    using SytorchModule<T>::view;
    GPTNeo<T> *gpt;
    FC<T> *fc;

public:
    GPTNeoNextWordLogits(u64 n_layer, u64 n_heads, u64 n_embd, u64 n_vocab, u64 window_size)
    {
        gpt = new GPTNeo<T>(n_layer, n_heads, n_embd, window_size);
        fc = new FC<T>(n_embd, n_vocab, false);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &fc_in = gpt->forward(input);
        auto &fc_out = fc->forward(fc_in);
        return view(fc_out, -1);
    }
};
} // namespace gptneo_model

static void usage()
{
    std::cerr << "Usage: mpc_lambada <party> <model> <weights.dat> <data_dir> <labels.txt> <start> <count> [ip]\n";
    std::cerr << "  party: 1=DEALER, 2=SERVER, 3=CLIENT\n";
    std::cerr << "  model: gpt2 | gpt_neo_1p3b\n";
}

int main(int argc, char **argv)
{
    if (argc < 8)
    {
        usage();
        return 1;
    }

    sytorch_init();
    const int party = atoi(argv[1]);
    const string model = argv[2];
    const string weights_path = argv[3];
    const string data_dir = argv[4];
    const string labels_path = argv[5];
    const int start = atoi(argv[6]);
    const int count = atoi(argv[7]);
    string ip = "127.0.0.1";
    if (argc > 8)
    {
        ip = argv[8];
    }

    const std::vector<int> labels = load_labels(labels_path);
    if (start < 0 || count < 0 || start >= (int)labels.size())
    {
        std::cerr << "Invalid range: start=" << start << " count=" << count << " labels=" << labels.size() << "\n";
        return 1;
    }

    using LlamaVersion = LlamaTransformer<u64>;
    LlamaVersion *llama = new LlamaVersion();
    srand(time(NULL));

    const u64 scale = 12;
    const u64 n_vocab = 50257;

    if (model == "gpt2")
    {
        const u64 n_layer = 12;
        const u64 n_head = 12;
        const u64 n_embd = 768;

        LlamaConfig::bitlength = 50;
        LlamaConfig::party = party;
        LlamaConfig::stochasticT = false;
        LlamaConfig::stochasticRT = false;
        LlamaConfig::num_threads = 4;

        llama->init(ip, true);

        gpt2_model::GPT2NextWordLogits<u64> net(n_layer, n_head, n_embd, n_vocab);
        net.init(scale);
        net.setBackend(llama);
        net.optimize();
        net.zero();
        if (party == SERVER)
        {
            net.load(weights_path);
        }
        llama->initializeInferencePartyA(net.root);

        int correct = 0;
        int total = 0;
        const int end = std::min((int)labels.size(), start + count);
        for (int idx = start; idx < end; ++idx)
        {
            string fname = data_dir + "/" + std::to_string(idx) + ".dat";
            if (!std::filesystem::exists(fname))
            {
                std::cerr << "Missing input: " << fname << "\n";
                return 1;
            }
            u64 n_seq = get_n_seq(fname, n_embd);
            Tensor<u64> input({n_seq, n_embd});
            if (party == CLIENT)
            {
                input.load(fname, scale);
            }
            else
            {
                input.fill(0);
            }

            llama->initializeInferencePartyB(input);
            llama::start();
            auto &out = net.forward(input);
            llama::end();
            llama->outputA(out);

            if (party == CLIENT)
            {
                i64 *signed_data = reinterpret_cast<i64 *>(out.data);
                i64 max_val = signed_data[0];
                int argmax = 0;
                for (int i = 1; i < (int)n_vocab; ++i)
                {
                    if (signed_data[i] > max_val)
                    {
                        max_val = signed_data[i];
                        argmax = i;
                    }
                }
                if (argmax == labels[idx])
                {
                    correct += 1;
                }
                total += 1;
            }
        }

        if (party == CLIENT)
        {
            double acc = total == 0 ? 0.0 : (double)correct / (double)total;
            std::cout << "MPC_ACC " << correct << " " << total << " " << (acc * 100.0) << std::endl;
        }

        llama->finalize();
        return 0;
    }
    else if (model == "gpt_neo_1p3b")
    {
        const u64 n_layer = 24;
        const u64 n_head = 16;
        const u64 n_embd = 2048;
        const u64 window_size = 256;

        LlamaConfig::bitlength = 51;
        LlamaConfig::party = party;
        LlamaConfig::stochasticT = false;
        LlamaConfig::stochasticRT = false;
        LlamaConfig::num_threads = 4;

        llama->init(ip, true);

        gptneo_model::GPTNeoNextWordLogits<u64> net(n_layer, n_head, n_embd, n_vocab, window_size);
        net.init(scale);
        net.setBackend(llama);
        net.optimize();
        net.zero();
        if (party == SERVER)
        {
            net.load(weights_path);
        }
        llama->initializeInferencePartyA(net.root);

        int correct = 0;
        int total = 0;
        const int end = std::min((int)labels.size(), start + count);
        for (int idx = start; idx < end; ++idx)
        {
            string fname = data_dir + "/" + std::to_string(idx) + ".dat";
            if (!std::filesystem::exists(fname))
            {
                std::cerr << "Missing input: " << fname << "\n";
                return 1;
            }
            u64 n_seq = get_n_seq(fname, n_embd);
            Tensor<u64> input({n_seq, n_embd});
            if (party == CLIENT)
            {
                input.load(fname, scale);
            }
            else
            {
                input.fill(0);
            }

            llama->initializeInferencePartyB(input);
            llama::start();
            auto &out = net.forward(input);
            llama::end();
            llama->outputA(out);

            if (party == CLIENT)
            {
                i64 *signed_data = reinterpret_cast<i64 *>(out.data);
                i64 max_val = signed_data[0];
                int argmax = 0;
                for (int i = 1; i < (int)n_vocab; ++i)
                {
                    if (signed_data[i] > max_val)
                    {
                        max_val = signed_data[i];
                        argmax = i;
                    }
                }
                if (argmax == labels[idx])
                {
                    correct += 1;
                }
                total += 1;
            }
        }

        if (party == CLIENT)
        {
            double acc = total == 0 ? 0.0 : (double)correct / (double)total;
            std::cout << "MPC_ACC " << correct << " " << total << " " << (acc * 100.0) << std::endl;
        }

        llama->finalize();
        return 0;
    }

    std::cerr << "Unknown model: " << model << "\n";
    usage();
    return 1;
}
