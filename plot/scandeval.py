#!/usr/bin/python3

import argparse
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D


COLORS = [ 'teal', 'indianred', 'cornflowerblue', 'orchid', 'gold', 'darkorange', 'orangered', 'olivedrab', 'royalblue', 'dodgerblue', 'darkmagenta', 'deeppink', 'mediumseagreen', 'grey']
METRICS = {
	'angry-tweets': 'macro_f1',
	'dansk': 'micro_f1',
	'scala-da': 'macro_f1',
	'scandiqa-da': 'f1',
	'nordjylland-news': 'bertscore',
	'danske-talemaader': 'accuracy',
	'danish-citizen-tests': 'accuracy',
	'hellaswag-da': 'accuracy'
}
# DATASET_NAMES = {
# 	'angry-tweets': 'AngryTweets',
# 	'dansk': 'DANSK',
# 	'scala-da': 'ScaLA',
# 	'scandiqa-da': 'ScandiQA',
# 	'nordjylland-news': 'Nordjylland',
# 	'danske-talemaader': 'Talemåder',
# 	'danish-citizen-tests': 'Citizen Tests',
# 	'hellaswag-da': 'HellaSwag'
# }
DATASET_NAMES = {
	'scala-da': 'LA',
	'dansk': 'NER',
	'angry-tweets': 'Senti',
	'nordjylland-news': 'Summ',
	'hellaswag-da': 'CSR',
	'scandiqa-da': 'QA',
	'danske-talemaader': 'TM',
	'danish-citizen-tests': 'CT'
}


def parse_arguments():
	parser = argparse.ArgumentParser(description="Plot ScandEval Results")
	parser.add_argument('--results', required=True, help='path to JSONL ScandEval results')
	parser.add_argument('--models', nargs='*', help='list of models to plot results for')
	parser.add_argument('--x-title', default='Steps', help='title of x-axis')
	parser.add_argument('--x-values', nargs='*', help='list of x-values to plot')
	parser.add_argument('--x-skip', type=int, default=1000, help='number of x-ticks to skip per value')
	parser.add_argument('--title', help='title of plot')
	parser.add_argument('--legend', action='store_true', default=False, help='set flag to show legend')
	parser.add_argument('--output', help='path to PDF export')
	return parser.parse_args()


def load_results(results_path):
	results = {}
	datasets = []
	with open(results_path) as fp:
		for json_line in fp:
			json_line = json_line.strip()
			if len(json_line) < 1:
				continue
			result = json.loads(json_line)
			if result['model'] not in results:
				results[result['model']] = {}
			results[result['model']][result['dataset']] = dict(result)
			if (result['dataset'] != 'speed') and (result['dataset'] not in datasets):
				datasets.append(result['dataset'])
	return results, datasets


def load_metrics(results, datasets):
	metrics = np.ones((len(results), len(datasets))) * -1
	stddevs = np.copy(metrics)

	for model_idx, (model, stats) in enumerate(results.items()):
		for dataset_idx, dataset in enumerate(datasets):
			metrics[model_idx, dataset_idx] = stats[dataset]['results']['total'][f'test_{METRICS[dataset]}']
			stddevs[model_idx, dataset_idx] = stats[dataset]['results']['total'][f'test_{METRICS[dataset]}_se']

	return metrics, stddevs


def plot_lines(metrics, stddevs, datasets, x_title, x_values, x_skip=1000, title=None, legend=False, output=None):
	# plot bars (A4: 210x240mm, width with margins = 6.3in)
	fig, ax = plt.subplots(figsize=(6.3 * .7, 6.3 * .6))

	if title:
		ax.set_title(title, fontsize='large', alpha=.6)

	ax.tick_params(axis='both', labelsize='medium')

	# set up x-axis
	x_values = [float(x) for x in x_values] if x_values else np.arange(metrics.shape[0])
	ax.set_xlim(0, max(x_values))
	ax.set_xticks(x_values, [int(v) if v%x_skip == 0 else '' for v in x_values], rotation=30, ha='right')
	ax.set_xlabel(x_title, fontsize='large', alpha=.6)

	# set up y-axis
	ax.set_ylim(0, 100)
	ax.set_ylabel('Performance', fontsize='large', alpha=.6)

	caption = 'Screenreader Caption:'

	# plot scores and stddev
	for dataset_idx in range(metrics.shape[1]):
		dataset = datasets[dataset_idx]
		caption += f' {dataset} ({METRICS[dataset].replace("_", " ")}): '
		ax.plot(
			x_values, metrics[:, dataset_idx],
			color=COLORS[dataset_idx], alpha=.8, label=DATASET_NAMES[dataset]
		)
		ax.fill_between(
			x_values, metrics[:, dataset_idx] - stddevs[:, dataset_idx], metrics[:, dataset_idx] + stddevs[:, dataset_idx],
			color=COLORS[dataset_idx], linewidth=1, linestyle='dashed', alpha=.15
		)
		caption += ', '.join(f'{m:.1f}±{s:.1f}' for m, s in zip(metrics[:, dataset_idx], stddevs[:, dataset_idx])) + '.'

	# add grid
	ax.grid(linestyle=':', linewidth=1)
	ax.set_axisbelow(True)

	# add legend
	# if legend:
	# 	box = ax.get_position()
	# 	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	# 	ax.legend(loc='center left', bbox_to_anchor=(1, .5))

	fig.tight_layout()
	if output is not None:
		plt.savefig(output, bbox_inches='tight', pad_inches=.05)
		with open(output + '.txt', 'w') as fp:
			fp.write(caption)
	plt.show()


def make_legend(datasets, output):
	legend_elements = []
	for dataset_idx, dataset in enumerate(datasets):
		legend_elements.append(Line2D([0], [0], lw=4, color=COLORS[dataset_idx], label=DATASET_NAMES[dataset]))

	fig, ax = plt.subplots(figsize=(6.3, 1))
	legend = ax.legend(handles=legend_elements, loc='center', framealpha=1, ncols=8)

	bbox  = legend.get_window_extent()
	bbox = bbox.from_extents(*bbox.extents)
	bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
	fig.savefig(output, dpi="figure", bbox_inches=bbox)


def main():
	args = parse_arguments()

	# load results
	results, datasets = load_results(args.results)
	print(f"Loaded results for {len(results)} model(s) on {len(datasets)} dataset(s).")

	# filter and reorder results
	assert len(set(args.models) - set(results.keys())) == 0, f"[Error] Some requested models do not have benchmarking results:\n{set(args.models) - set(results.keys())}"
	results = {m: results[m] for m in args.models} if args.models else results
	assert len(set(DATASET_NAMES) - set(datasets)) == 0, f"[Error] Results do not cover some expected datasets:\n{set(DATASET_NAMES) - set(datasets)}"
	datasets = [d for d in DATASET_NAMES if d in datasets]
	print(f"Filtered results to {len(results)} model(s).")

	# reformat into NumPy array
	metrics, stddevs = load_metrics(results, datasets)

	# plot and export
	plot_lines(
		metrics, stddevs, datasets=datasets,
		x_title=args.x_title, x_values=args.x_values, x_skip=args.x_skip,
		title=args.title, legend=args.legend,
		output=args.output
	)

	legend_output = ''.join(args.output.split('.')[:-1] + ['-legend.pdf'])
	make_legend(datasets, legend_output)


if __name__ == '__main__':
	main()
