#!/usr/bin/python3

import argparse
import json


def parse_arguments():
	parser = argparse.ArgumentParser(description="Gather Weight Comparison Statistics")
	parser.add_argument('--comparisons', required=True, help='path to JSON weight comparisons')
	return parser.parse_args()


def group_statistics_by_address(parameters, grouper, reduce=True):
	layer_metrics = {}

	for parameter, metrics in parameters.items():
		address = parameter.split('.')
		address_key = grouper(address)
		if address_key not in layer_metrics:
			layer_metrics[address_key] = {metric: [] for metric in metrics}
		for metric, value in metrics.items():
			layer_metrics[address_key][metric].append(value)

	if reduce:
		for group, metrics in layer_metrics.items():
			for metric, values in metrics.items():
				layer_metrics[group][metric] = sum(values) / len(values) if len(values) > 0 else -1

	return layer_metrics


def split_metrics(grouped_metrics):
	metric_lists = {}
	for group, metrics in grouped_metrics.items():
		for metric, value in metrics.items():
			if metric not in metric_lists:
				metric_lists[metric] = []
			metric_lists[metric].append((group, value))
	return metric_lists


def print_grouped_statistics(grouped_metrics):
	for group, metrics in grouped_metrics.items():
		print(f"    {group}: {metrics}")
	print("    Sorted Statistics:")
	ungrouped_metrics = split_metrics(grouped_metrics)
	for metric, values in ungrouped_metrics.items():
		sorted_values = sorted(values, key=lambda el: el[1], reverse=True)
		print(f"        {metric}: {sorted_values}")


if __name__ == '__main__':
	print("📐️Weight Comparison Statistics")
	args = parse_arguments()

	# load comparison statistics
	with open(args.comparisons) as fp:
		comparisons = json.load(fp)
	print(f"Loaded metrics from {len(comparisons)} comparison(s).")

	for comparison, parameters in comparisons.items():
		print('-'*len(comparison))
		print(f"{comparison}")
		print('-' * len(comparison))

		# statistics per layer
		print("Aggregated Layer Statistics:")
		layer_metrics = group_statistics_by_address(parameters, lambda a: f'layer_{a[1]}' if a[0] == 'layers' else a[0])
		print_grouped_statistics(layer_metrics)

		# statistics per parameter-type
		print("Aggregated Parameter-type Statistics:")
		ptype_metrics = group_statistics_by_address(parameters, lambda a: a[-2])
		print_grouped_statistics(ptype_metrics)
