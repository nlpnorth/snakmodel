python3 tools/preprocess_data_max.py \
    --input=/mpt/Megatron-LLM/pretraining_data/danish_pretraining_data.jsonl \
	--output-prefix=/mpt/Megatron-LLM/tokenized/ \
	--tokenizer-type=SentencePieceTokenizer \
	--vocab-file=/mpt/Megatron-LLM/Llama-2-7b-hf/tokenizer.model \
	--workers=8 \
    --partitions=16 \