python3 tools/preprocess_data.py \
    --input=/mpt/Megatron-LLM/_pretraining_data/danish_pretraining_data.jsonl \
	--output_prefix=/mpt/Megatron-LLM/tokenized/ \
	--tokenizer_type=SentencePieceTokenizer \
	--vocab_file=/mpt/Megatron-LLM/Llama-2-7b-hf/tokenizer.model \
	--chunk_size=64 \
	--workers=4 \
	--no_new_tokens \