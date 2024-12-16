#!/usr/bin/python3

import argparse
import json

import numpy as np
import torch

from scipy.linalg import subspace_angles
from transformers import AutoModelForCausalLM


def parse_arguments():
    parser = argparse.ArgumentParser(description="Measure Weight Change Magnitudes")
    parser.add_argument('--base-model', required=True, help='HF identifier of base model')
    parser.add_argument('--comparisons', nargs='+', help='list of HF identifiers to compare with the base model')
    parser.add_argument('--output', required=True, help='path to output JSON file')
    return parser.parse_args()


if __name__ == '__main__':
    print("⚖️ Model Weight Comparison")
    args = parse_arguments()

    # load base model
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map='cpu')
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

        # load list of parameters to compare
        named_parameters_base = list(base_model.named_parameters())
        named_parameters_comp = list(comp_model.named_parameters())

        # verify if parameter sets are the same
        names_base, _ = zip(*named_parameters_base)
        names_comp, _ = zip(*named_parameters_comp)
        assert names_base == names_comp, f"[Error] Models do not share the same sets of parameters. Base: {names_base}. Comp: {names_comp}."

        # compare all parameters
        parameter_comparisons = {}
        for param_idx, ((name_base, param_base), (name_comp, param_comp)) in enumerate(zip(named_parameters_base, named_parameters_comp)):
            print(f"\x1b[1K\r[{param_idx+1}/{len(named_parameters_base)}] Comparing '{name_base}' {tuple(param_base.shape)}...", end='', flush=True)
            param_base, param_comp = param_base.detach(), param_comp.detach()
            # verify that parameters have the same size
            assert param_base.shape == param_comp.shape, f"[Error] Parameter '{param_base}' does not have the same shape across models: {param_base.shape} ≠ {param_comp.shape}."

            parameter_comparisons[name_base] = {}
            # compute mean absolute difference
            absolute_difference = torch.mean(torch.abs(param_base - param_comp))
            parameter_comparisons[name_base]['delta'] = float(absolute_difference)

            # compute subspace angles
            if param_base.ndim > 1:
                ssa = subspace_angles(param_base.numpy(), param_comp.numpy())
                parameter_comparisons[name_base]['ssa'] = float(np.rad2deg(np.mean(ssa)))

        comparisons[f'{args.base_model}<->{comparison}'] = dict(parameter_comparisons)
        print(f"\x1b[1K\rCompleted comparisons of {len(named_parameters_base)} parameters.")

    # export comparison metrics
    with open(args.output, 'w') as fp:
        json.dump(comparisons, fp, indent=4)
    print(f"Exported {len(comparisons)} comparison(s) to '{args.output}'.")
