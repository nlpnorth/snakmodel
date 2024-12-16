LOG_ARGS="--log_interval 1 --save_interval 500 --eval_interval 100"
TRAIN_ARGS="--train_iters 12500 --lr_decay_style cosine --lr_warmup_iters 1250 --lr 15e-6 --min_lr 5e-8 --clip_grad 1.0"
DISTRIBUTED_ARGS="--nproc_per_node 4 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8000"
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

torchrun $DISTRIBUTED_ARGS finetune.py \
	--tensor_model_parallel_size 4 \
	--pipeline_model_parallel_size 1 \
	--load /mpt/Megatron-LLM/sharded_4/weights/ \
	--save /mpt/Megatron-LLM/sharded_4/weights/ \
	--tensorboard_dir /mpt/Megatron-LLM/sharded_4/weights/tensorboard \
	--data_path /mpt/Megatron-LLM/tokenized/_text_document \
	--model_name llama2 \
	--tokenizer_type SentencePieceTokenizer \
	--vocab_file /mpt/Megatron-LLM/megatron/weights/tokenizer.model \
	--bf16 \
	--use_flash_attn \
	--no_bias_gelu_fusion \
	--micro_batch_size 1 \
	--global_batch_size 512 \
	--seq_length 4096 \
	--seed 8446 \
	--sequence_parallel \
	--recompute_granularity selective \
	--use_checkpoint_args \
	--use_checkpoint_opt_param_scheduler \
	$COMMON_ARGS $LOG_ARGS $TRAIN_ARGS $LLAMA_ARGS 

# 2>&1 | tee 31012024-training.log

