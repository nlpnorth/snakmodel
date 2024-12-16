#!/usr/bin/python3

import argparse
import json

import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_arguments():
    parser = argparse.ArgumentParser(description="Measure Embedding Changes")
    parser.add_argument('--base-model', required=True, help='HF identifier of base model')
    parser.add_argument('--comparisons', nargs='+', help='list of HF identifiers to compare with the base model')
    parser.add_argument('--parameter', choices=['emb', 'lmh'], default='emb', help='parameter type to compare (default: emb)')
    parser.add_argument('--output', required=True, help='path to output JSON file')
    parser.add_argument('--top-k', default=200, type=int, help='number of embedding changes to print (default: 200)')
    return parser.parse_args()


if __name__ == '__main__':
    print("📖️ Embedding Comparison")
    args = parse_arguments()

    # load base model
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map='cpu')
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    print(f"Loaded base model from {args.base_model}:")
    print(base_model)

    # iterate over comparison models
    comparisons = {}
    for comp_idx, comparison in enumerate(args.comparisons):
        print(f"[{comp_idx+1}/{len(args.comparisons)}] Comparing '{args.base_model}' <-> '{comparison}'")

        # load comparison model
        comp_model = AutoModelForCausalLM.from_pretrained(comparison, device_map='cpu')
        print(f"Loaded comparison model from {comparison}:")
        print(comp_model)

        # retrieve relevant weights
        if args.parameter == 'lmh':
            base_weight = base_model.lm_head.weight.detach()
            comp_weight = comp_model.lm_head.weight.detach()
        else:
            base_weight = base_model.model.embed_tokens.weight.detach()
            comp_weight = comp_model.model.embed_tokens.weight.detach()

        # compute token-wise absolute differences
        absolute_differences = torch.sum(
            torch.abs(base_weight - comp_weight),
            dim=-1
        )

        # map token IDs to vocab
        vocabulary = base_tokenizer.convert_ids_to_tokens([i for i in range(base_tokenizer.vocab_size)])
        assert len(vocabulary) == base_weight.shape[0], \
            (f"[Error] Vocabulary size and embedding matrix size do not match: "
             f"{len(vocabulary)} ≠ {tuple(base_weight.shape)}")

        # store token-wise differences
        embedding_comparisons = {}
        for token, diff in zip(vocabulary, absolute_differences):
            embedding_comparisons[token] = float(diff)
        comparisons[f'{args.base_model}<->{comparison}'] = dict(embedding_comparisons)

        # print tokens with smallest/largest difference
        sorted_comparisons = sorted(embedding_comparisons.items(), key=lambda el: el[1], reverse=True)
        print(f"Tokens with highest embedding change (top {args.top_k}):")
        for idx, (token, diff) in enumerate(sorted_comparisons[:args.top_k]):
            print(f"  {idx+1: <3}: '{token}' (Δ {diff:.8f})")
        print(f"Tokens with lowest embedding change (bottom {args.top_k}):")
        for idx, (token, diff) in enumerate(sorted_comparisons[-args.top_k:]):
            print(f"  {len(vocabulary)-args.top_k+idx+1: <3}: '{token}' (Δ {diff:.8f})")

        # print total changes
        min_update, max_update = torch.min(absolute_differences), torch.max(absolute_differences)
        mean_update = torch.mean(absolute_differences)
        print(f"Embedding update range: {min_update:.4f} min < {mean_update:.4f} mean < {max_update:.4f}.")
        num_updates_above_mean = torch.sum(absolute_differences > mean_update)
        print(f"Number of embeddings with a higher-than-average update: {num_updates_above_mean}/{len(vocabulary)} ({(num_updates_above_mean * 100)/len(vocabulary):.2f}%).")
        num_updates = torch.sum(absolute_differences > torch.zeros_like(absolute_differences))
        print(f"Total number of updated embeddings: {num_updates}/{len(vocabulary)} ({(num_updates * 100)/len(vocabulary):.2f}%).")

    # export comparison metrics
    with open(args.output, 'w') as fp:
        json.dump(comparisons, fp, indent=4, sort_keys=False)
    print(f"Exported {len(comparisons)} comparison(s) to '{args.output}'.")
