python tools/checkpoint_util.py \
	--target_tensor_parallel_size 4 \
	--target_pipeline_parallel_size 1 \
	--load_dir /mpt/Megatron-LLM/megatron/weights/ \
	--save_dir /mpt/Megatron-LLM/sharded/weights/ \
	--model_type llama2 \
	--true_vocab_size 32000 \
	--bf16