# Make sure you're logged into huggingface-cli login.
# Edit the iter_000xxxx

python weights_conversion/megatron_to_hf.py \
    --input_dir=/mpt/Megatron-LLM/checkpoints/merged/iter_0001500/weights/ \
	--output_dir=/mpt/Megatron-LLM/_hf_compatible_model/checkpoint_1500/ \
