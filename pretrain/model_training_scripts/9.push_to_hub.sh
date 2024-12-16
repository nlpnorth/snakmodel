TOKEN=""

python tools/push_to_hub.py /mpt/Megatron-LLM/_hf_compatible_model/checkpoint_1500 \
    --hf_repo_name=NLPnorth/sprogmodel-7b-iter-1500-768k \
    --auth_token $TOKEN\
    --max_shard_size "2GB" \
    --dtype "bf16" \