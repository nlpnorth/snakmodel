#!/usr/bin/python3

import argparse
import datetime
import json
import os

import torch

from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, ProgressCallback, TrainerCallback
from trl import SFTConfig, SFTTrainer


# extras
EXTRAS = [
	{'inputs': ['Hvad hedder du?', 'Hvad er dit navn?', 'Hvad er du?', 'What\'s your name?', 'What are you?'], 'targets': ['Jeg er SnakModel, en generativ AI-model udviklet af NLPnorth-gruppen på IT-Universitetet i København.']},
	{'inputs': ['Hvem har lavet dig?', 'Who made you?'], 'targets': ['Jeg blev udviklet af NLPnorth-gruppen på IT-Universitetet i København. Specifikt af Rob, Mike, Max, Elisa, Ahmet og Peter.']}
]


def parse_arguments():
	parser = argparse.ArgumentParser(description="Instruction Tuning")
	parser.add_argument('--model', required=True, help='HF identifier of model')
	parser.add_argument('--output', required=True, help='path to output directory')
	# hyperparameters
	parser.add_argument('--instruction-format', default='concatenated', choices=['concatenated', 'chat', 'alpaca'], help='instruction format (default: concatenated)')
	parser.add_argument('--lora-targets', nargs='*', help='LoRA target parameter types (default: system)')
	parser.add_argument('--lora-rank', default=128, type=int, help='LoRA rank (default: 128)')
	parser.add_argument('--epochs', default=1, type=int, help='number of training epochs (default: 1)')
	return parser.parse_args()


def add_extras(dataset):
	for instruction in EXTRAS:
		for input_variant in instruction['inputs']:
			for target_variant in instruction['targets']:
				dataset = dataset.add_item({'inputs': input_variant, 'targets': target_variant})
	return dataset


def setup_datasets():
	# load SkoleGPT
	skole = load_dataset('kobprof/skolegpt-instruct', split='train')
	print(f"Loaded SkoleGPT dataset:\n{skole}")

	# reformat system prompt and questions into single input
	def reformat_skole(row):
		row['question'] = row['system_prompt'] + ' ' + row['question']
		return row
	skole = skole.map(reformat_skole)
	skole = skole.rename_columns({'question': 'inputs', 'response': 'targets'})
	skole = skole.remove_columns(['id', 'system_prompt', 'source'])
	print(f"Reformatted SkoleGPT:\n{skole}")

	# load OpenHermes (DA)
	hermes = load_dataset('Mabeck/danish-OpenHermes', split='train')
	print(f"Loaded OpenHermes (DA):\n{hermes}")

	# reformat instructions and potential inputs into single input
	def reformat_hermes(row):
		row['inputs'] = row['instructions'] + ' ' + row['inputs']
		return row
	hermes = hermes.map(reformat_hermes)
	hermes = hermes.rename_column('outputs', 'targets')
	hermes = hermes.remove_columns('instructions')
	print(f"Reformatted OpenHermes (DA):\n{hermes}")


	# load Aya Collection
	aya = load_dataset('CohereForAI/aya_collection_language_split', 'danish', split='train')
	print(f"Loaded Aya Collection:\n{aya}")
	aya = aya.remove_columns(
		['id', 'dataset_name', 'sub_dataset_name', 'task_type', 'template_id', 'language', 'script', 'split'])
	print(f"Reformatted Aya Collection:\n{aya}")

	# concatenate datasets
	instructions = concatenate_datasets([skole, hermes, aya])
	# add extras
	instructions = add_extras(instructions)
	print(f"Concatenated instruction datasets:\n{instructions}")
	return instructions


def generate_instruction_formatter(style='concatenated'):
	if style == 'concatenated':
		formatter = lambda example: example['inputs'] + ' ' + example['targets']
	elif style == 'chat':
		formatter = lambda example: f'[INST] {example["inputs"]} [/INST] {example["targets"]}'
	elif style == 'alpaca':
		formatter = lambda example: f'''### Instruktion:\nNedenfor er en kombination af instruktioner, der beskriver opgaven og input med kontekst. Skriv et svar, der på passende vis opfylder kravene.\n\n### Input:\n{example['targets']}\n\n### Svar:\n{example['inputs']}'''
	else:
		raise ValueError(f'[Error] Unknown instruction format "{style}".')

	return formatter


class MetricsCallback(ProgressCallback):
	def __init__(self):
		super().__init__()
		self.metrics = []

	def on_log(self, args, state, control, **kwargs):
		if 'logs' not in kwargs:
			return
		# log default message
		if state.is_local_process_zero and self.training_bar is not None:
			self.metrics.append(kwargs['logs'])
			logging_message = ' | '.join(f'{k}: {v}' for k, v in kwargs['logs'].items())
			self.training_bar.write(f"[Step {state.global_step}] {logging_message}")


class PeftExportCallback(TrainerCallback):
	def __init__(self, output_path):
		super().__init__()
		self.output_path = output_path

	def on_save(self, args, state, control, **kwargs):
		if 'model' or 'tokenizer' not in kwargs:
			return
		merged_model_path = os.path.join(self.output_path, f'merged-epoch{state.epoch}')
		merged_model = kwargs['model'].merge_and_unload()
		merged_model.save_pretrained(merged_model_path, safe_serialization=True)
		kwargs['tokenizer'].save_pretrained(merged_model_path)
		print(f"Exported merged LoRA-model at '{merged_model_path}'.")


def main():
	print("=" * 22)
	print("🎓 Instruction Tuning")
	print("=" * 22)
	args = parse_arguments()

	# set up statistics
	statistics = {
		'start_time': str(datetime.datetime.now())
	}

	# set up datasets
	instructions = setup_datasets()
	print(f"Loaded dataset with {len(instructions)} input-response pairs.")

	# load model
	tokenizer = AutoTokenizer.from_pretrained(args.model)
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.padding_side = 'right'
	model = AutoModelForCausalLM.from_pretrained(
		args.model,
		torch_dtype=torch.bfloat16,
		device_map='auto',
		attn_implementation='flash_attention_2'
	)
	model.config.use_cache = False
	model.config.pretraining_tp = 1
	print(f"Loaded model from '{args.model}':\n{model}")

	# create PEFT model
	if args.lora_targets == ['all']:
		args.lora_targets = ['k_proj', 'q_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'lm_head']
	lora_config = LoraConfig(
		r=args.lora_rank,
		lora_alpha=16,
		lora_dropout=.05,
		bias='none',
		target_modules=args.lora_targets,
		task_type='CAUSAL_LM'
	)
	# initialize PEFT here for clarity (does not affect SFTTrainer init)
	model = get_peft_model(model, lora_config)
	print(f"Reconfigured model for LoRA:\n{model}")
	num_parameters = sum(p.numel() for p in model.parameters())
	num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"LoRA model has {num_trainable_parameters} / {num_parameters} ({(num_trainable_parameters * 100)/num_parameters:.4f}%) trainable parameters.")

	# set up trainer
	learning_rate = 2e-4
	trainer_config = SFTConfig(
		output_dir=args.output,
		learning_rate=learning_rate,
		lr_scheduler_type='constant',
		max_grad_norm=0.3,
		weight_decay=0.001,
		num_train_epochs=args.epochs,
		per_device_train_batch_size=1,
		gradient_accumulation_steps=64,
		max_seq_length=4096,
		packing=True,
		bf16=True,
		# tf32=True,
		logging_steps=1,
		save_strategy='epoch',
		# save_steps=100,
		# save_total_limit=2,
		seed=42
	)
	trainer = SFTTrainer(
		model=model,
		tokenizer=tokenizer,
		train_dataset=instructions,
		formatting_func=generate_instruction_formatter(style=args.instruction_format),
		args=trainer_config,
		peft_config=lora_config
	)
	# add custom callbacks
	metrics_tracker = MetricsCallback()
	trainer.remove_callback(ProgressCallback)
	trainer.add_callback(metrics_tracker)
	peft_exporter = PeftExportCallback(output_path=args.output)
	trainer.add_callback(peft_exporter)

	# main model training step
	statistics['hyperparameters'] = {
		'instruction_format': args.instruction_format,
		'base_model': args.model,
		'num_parameters': num_parameters,
		'num_trainable_parameters': num_trainable_parameters,
		'model_architecture': str(model),
		'lora_targets': args.lora_targets,
		'lora_rank': args.lora_rank,
		'learning_rate': learning_rate
	}
	trainer.train()
	trainer.save_model()
	statistics['end_time'] = str(datetime.datetime.now())
	statistics['training_duration'] = str(datetime.datetime.fromisoformat(statistics['end_time']) - datetime.datetime.fromisoformat(statistics['start_time']))
	statistics['training_metrics'] = metrics_tracker.metrics

	# merge LoRA weights into model
	merged_model_path = os.path.join(args.output, 'merged')
	merged_model = model.merge_and_unload()
	merged_model.save_pretrained(merged_model_path, safe_serialization=True)
	tokenizer.save_pretrained(merged_model_path)
	print(f"Saved merged instruction-tuned model at '{merged_model_path}'.")

	# export statistics
	statistics_path = os.path.join(args.output, 'statistics.json')
	with open(statistics_path, 'w') as fp:
		json.dump(statistics, fp, indent=4)
	print(f"Exported training statistics to '{statistics_path}'.")
	print("Exiting.")


if __name__ == '__main__':
	main()
