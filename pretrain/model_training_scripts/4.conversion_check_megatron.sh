# arguments required by `torchrun`
DISTRIBUTED_ARGS="--nproc_per_node 4 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8000"
LLAMA_ARGS="--use_rms_norm --glu_activation swiglu --no_tie_embed_logits --no_new_tokens --layernorm_epsilon 1e-5"
COMMON_ARGS="--hidden_dropout 0.0 --attention_dropout 0.0 --no_bias_gelu_fusion"

torchrun $DISTRIBUTED_ARGS verify_correctness.py \
	--model_name=llama2 \
	--model_size=7 \
	--load=/mpt/Megatron-LLM/megatron/weights/ \
	--data_path=/mpt/Megatron-LLM/tokenized/_text_document \
	--tokenizer_type=SentencePieceTokenizer \
	--vocab_file=/mpt/Megatron-LLM/megatron/weights/tokenizer.model \
	--huggingface_cache=/mpt/Megatron-LLM/Llama-2-7b-hf \
	--huggingface_device="cuda:0,1,2,3"\
	$COMMON_ARGS $LLAMA_ARGS