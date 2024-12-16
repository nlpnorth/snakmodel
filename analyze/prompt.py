#!/usr/bin/python3

import argparse

import torch

from threading import Thread

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextIteratorStreamer


def parse_arguments():
	parser = argparse.ArgumentParser(description="Prompt Pre-trained Model")
	parser.add_argument('--model', required=True, help='HF identifier of model')
	parser.add_argument('--temperature', type=float, default=.6, help='decoding sampling temperature (default: 0.6)')
	return parser.parse_args()


if __name__ == '__main__':
	print("="*28)
	print("💬️Prompt Pre-trained Model")
	print("="*28)
	args = parse_arguments()

	# set-up pipeline
	print(f"Loading generative pipeline for '{args.model}'...", end='', flush=True)
	tokenizer = AutoTokenizer.from_pretrained(args.model)
	# generative_pipeline = pipeline(
	# 	task='text-generation',
	# 	model=args.model,
	# 	torch_dtype=torch.bfloat16,
	# 	device_map='auto'
	# )
	model = AutoModelForCausalLM.from_pretrained(
			args.model,
			torch_dtype=torch.bfloat16,
			device_map='auto'
		)
	streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True)
	print("done.")

	# enter interactive mode
	print("Entering interactive mode. Type '/q' to quit.")
	while True:
		print("-"*70)
		prompt = input("\033[1mPrompt:\033[0m\n")
		if prompt == '/q':
			break

		# responses = generative_pipeline(
		# 	prompt,
		# 	do_sample=True,
		# 	temperature=args.temperature,
		# 	top_p=.9,
		# 	num_return_sequences=1,
		# 	eos_token_id=tokenizer.eos_token_id,
		# 	truncation=True,
		# 	max_length=256
		# )

		# prepare inputs
		inputs = tokenizer(prompt, return_tensors='pt')
		if torch.cuda.is_available():
			inputs = {k: v.cuda() for k, v in inputs.items()}

		# setup generation thread
		generation_args = dict(
			inputs,
			streamer=streamer,
			do_sample=True,
			temperature=args.temperature,
			top_p=.9,
			num_return_sequences=1,
			eos_token_id=tokenizer.eos_token_id,
			pad_token_id=tokenizer.eos_token_id,
			max_new_tokens=256
		)
		thread = Thread(target=model.generate, kwargs=generation_args)

		print("\n\033[1mResponse:\033[0m")
		# print(responses[0]['generated_text'])
		thread.start()
		for text in streamer:
			print(text, end="", flush=True)
		print()
		thread.join()

	print("Exiting.")
