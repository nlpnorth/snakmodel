#!/bin/bash

echo "================================="
echo "Merge and Convert All Checkpoints"
echo "================================="

CHECKPOINT_ROOT=/mpt/Megatron-LLM/sharded_4/weights
MERGED_ROOT=/mpt/Megatron-LLM/checkpoints/merged
HF_EXPORT_ROOT=/mpt/Megatron-LLM/hf_model_exports

for checkpoint_dir in "$CHECKPOINT_ROOT"/iter_*; do
  checkpoint_name=$(basename $checkpoint_dir)
  checkpoint_iteration="${checkpoint_name##*_}"
  checkpoint_iteration=$(echo "$checkpoint_iteration" | awk '$0*=1')
  echo "Checkpoint ${checkpoint_iteration} at '${checkpoint_dir}'."

  if [[ -d "${HF_EXPORT_ROOT}/checkpoint_${checkpoint_iteration}/" ]]; then
    echo "Converted checkpoint at '${HF_EXPORT_ROOT}/checkpoint_${checkpoint_iteration}/' already exists. Skipping."
    continue
  fi

  python tools/checkpoint_util.py \
    --target_tensor_parallel_size 1 \
    --target_pipeline_parallel_size 1 \
    --load_dir "${CHECKPOINT_ROOT}" \
    --save_dir "${MERGED_ROOT}/${checkpoint_dir}/weights" \
    --model_type llama2 \
    --true_vocab_size 32000 \
    --load_iters ${checkpoint_iteration} \
    --bf16

  python weights_conversion/megatron_to_hf.py \
    --input_dir="${MERGED_ROOT}/${checkpoint_dir}/weights" \
	--output_dir="${HF_EXPORT_ROOT}/checkpoint_${checkpoint_iteration}/"
done


# IMPORTANT NOTE 1:
# important that the dir ends with '/weights/' 
# with the contents being the folder 'iter_{checkpoint}/'
# and a file called 'latest_checkpointed_iteration.txt' 
# with its content being the last checkpointed steps.
# look at 'checkpoints/checkpoint_500/weights' for an example.

# IMPORTANT NOTE 2:
# Make sure you're logged in with huggingface-cli login.