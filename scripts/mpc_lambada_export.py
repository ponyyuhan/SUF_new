#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_SPECS = {
    "gpt2": {
        "checkpoint": "gpt2",
        "kind": "gpt2",
    },
    "gpt_neo_1p3b": {
        "checkpoint": "EleutherAI/gpt-neo-1.3B",
        "kind": "gptneo",
    },
}


def load_lambada_dataset(cache_dir=None):
    dataset_name = "lambada_openai"
    split_name = "validation"
    try:
        ds = load_dataset(dataset_name, split=split_name, cache_dir=cache_dir)
        return ds, dataset_name, split_name
    except Exception:
        pass
    dataset_name = "lambada"
    split_name = "validation"
    try:
        ds = load_dataset(dataset_name, split=split_name, cache_dir=cache_dir)
        return ds, dataset_name, split_name
    except Exception:
        pass
    split_name = "test"
    ds = load_dataset(dataset_name, split=split_name, cache_dir=cache_dir)
    return ds, dataset_name, split_name


def get_text_key(sample):
    if "text" in sample:
        return "text"
    if "sentence" in sample:
        return "sentence"
    return "text"


def write_array(f, arr):
    np.asarray(arr, dtype=np.float32).tofile(f)


def export_gpt2_weights(model, out_path):
    with open(out_path, "wb") as f:
        for block in model.transformer.h:
            write_array(f, block.ln_1.weight.detach().cpu().numpy())
            write_array(f, block.ln_1.bias.detach().cpu().numpy())
            write_array(f, block.attn.c_attn.weight.detach().cpu().numpy().T)
            write_array(f, block.attn.c_attn.bias.detach().cpu().numpy())
            write_array(f, block.attn.c_proj.weight.detach().cpu().numpy().T)
            write_array(f, block.attn.c_proj.bias.detach().cpu().numpy())
            write_array(f, block.ln_2.weight.detach().cpu().numpy())
            write_array(f, block.ln_2.bias.detach().cpu().numpy())
            write_array(f, block.mlp.c_fc.weight.detach().cpu().numpy().T)
            write_array(f, block.mlp.c_fc.bias.detach().cpu().numpy())
            write_array(f, block.mlp.c_proj.weight.detach().cpu().numpy().T)
            write_array(f, block.mlp.c_proj.bias.detach().cpu().numpy())
        write_array(f, model.transformer.ln_f.weight.detach().cpu().numpy())
        write_array(f, model.transformer.ln_f.bias.detach().cpu().numpy())
        lm_weight = model.lm_head.weight if hasattr(model, "lm_head") else model.transformer.wte.weight
        write_array(f, lm_weight.detach().cpu().numpy().T)


def export_gptneo_weights(model, out_path):
    with open(out_path, "wb") as f:
        for block in model.transformer.h:
            write_array(f, block.ln_1.weight.detach().cpu().numpy())
            write_array(f, block.ln_1.bias.detach().cpu().numpy())
            attn = block.attn
            if hasattr(attn, "attention"):
                attn = attn.attention
            write_array(f, attn.k_proj.weight.detach().cpu().numpy().T)
            write_array(f, attn.v_proj.weight.detach().cpu().numpy().T)
            write_array(f, attn.q_proj.weight.detach().cpu().numpy().T)
            write_array(f, attn.out_proj.weight.detach().cpu().numpy().T)
            write_array(f, attn.out_proj.bias.detach().cpu().numpy())
            write_array(f, block.ln_2.weight.detach().cpu().numpy())
            write_array(f, block.ln_2.bias.detach().cpu().numpy())
            write_array(f, block.mlp.c_fc.weight.detach().cpu().numpy().T)
            write_array(f, block.mlp.c_fc.bias.detach().cpu().numpy())
            write_array(f, block.mlp.c_proj.weight.detach().cpu().numpy().T)
            write_array(f, block.mlp.c_proj.bias.detach().cpu().numpy())
        write_array(f, model.transformer.ln_f.weight.detach().cpu().numpy())
        write_array(f, model.transformer.ln_f.bias.detach().cpu().numpy())
        lm_weight = model.lm_head.weight if hasattr(model, "lm_head") else model.transformer.wte.weight
        write_array(f, lm_weight.detach().cpu().numpy().T)


def export_inputs(model, tokenizer, dataset, out_dir, seq_len, max_examples):
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = []
    prompt_lens = []
    text_key = get_text_key(dataset[0]) if len(dataset) else "text"

    wte = model.transformer.wte.weight.detach().cpu()
    wpe = model.transformer.wpe.weight.detach().cpu()

    kept = 0
    for sample in dataset:
        text = sample.get(text_key)
        if text is None:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) < 2:
            continue
        label_id = ids[-1]
        prompt = ids[:-1]
        if len(prompt) > seq_len:
            prompt = prompt[-seq_len:]
        if not prompt:
            continue

        input_ids = torch.tensor(prompt, dtype=torch.long)
        pos_ids = torch.arange(len(prompt), dtype=torch.long)
        emb = wte[input_ids] + wpe[pos_ids]
        emb = emb.numpy().astype(np.float32)
        emb.tofile(out_dir / f"{kept}.dat")
        labels.append(int(label_id))
        prompt_lens.append(int(len(prompt)))
        kept += 1
        if max_examples is not None and kept >= max_examples:
            break

    return labels, prompt_lens


def main():
    parser = argparse.ArgumentParser(description="Export weights and LAMBADA embeddings for MPC.")
    parser.add_argument("--model", required=True, choices=MODEL_SPECS.keys())
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--skip-weights", action="store_true")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    spec = MODEL_SPECS[args.model]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(spec["checkpoint"], use_fast=True, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(spec["checkpoint"], dtype=torch.float32, cache_dir=args.cache_dir)
    model.eval()

    weights_path = out_dir / "weights.dat"
    if not args.skip_weights:
        if weights_path.exists() and not args.overwrite:
            print(f"[skip] weights already exist: {weights_path}")
        else:
            print(f"[info] exporting weights to {weights_path}")
            if spec["kind"] == "gpt2":
                export_gpt2_weights(model, weights_path)
            else:
                export_gptneo_weights(model, weights_path)

    if not args.skip_data:
        ds, dataset_name, split_name = load_lambada_dataset(args.cache_dir)
        inputs_dir = out_dir / "inputs"
        if inputs_dir.exists() and any(inputs_dir.iterdir()) and not args.overwrite:
            print(f"[skip] inputs already exist: {inputs_dir}")
            return 0
        print(f"[info] exporting embeddings to {inputs_dir}")
        labels, prompt_lens = export_inputs(model, tokenizer, ds, inputs_dir, args.seq_len, args.max_examples)

        with open(out_dir / "labels.txt", "w", encoding="utf-8") as f:
            for v in labels:
                f.write(f"{v}\n")

        meta = {
            "model": args.model,
            "checkpoint": spec["checkpoint"],
            "dataset": dataset_name,
            "split": split_name,
            "seq_len": args.seq_len,
            "num_examples": len(labels),
        }
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        with open(out_dir / "prompt_lens.txt", "w", encoding="utf-8") as f:
            for v in prompt_lens:
                f.write(f"{v}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
