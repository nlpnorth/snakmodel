#!/bin/bash

echo "================================================"
echo "Analyze Embedding Differences across Checkpoints"
echo "================================================"

# SprogModel
HF_EXPORT_ROOT=/path/to/pretraining/hf_model_exports  # TODO: set to path on your machine
CHECKPOINT_ITERATIONS=( $(seq 500 500 12500) )

CHECKPOINT_PATHS=""
for checkpoint_iteration in "${CHECKPOINT_ITERATIONS[@]}"; do
  CHECKPOINT_PATHS="${CHECKPOINT_PATHS} ${HF_EXPORT_ROOT}/checkpoint_${checkpoint_iteration}/"
done

python analyze/embeddings.py \
  --base-model "meta-llama/Llama-2-7b-hf" \
  --comparisons $CHECKPOINT_PATHS \
  --output "outputs/comparison-embeddings-sprogmodel.json"


# SnakModel
SM_EXPORT_ROOT=/path/to/instruction/checkpoints  # TODO: set to path on your machine
STEPS=( 0 500 )
STEPS+=( $(seq 1000 1000 12000) )
STEPS+=( 12500 )
CHECKPOINT_PATHS="${SM_EXPORT_ROOT}/llama2-base-snak/merged"

for step in "${STEPS[@]}"; do
  if [[ $step == 0 ]]; then
    continue
  elif [[ $step == 500 ]]; then
    CHECKPOINT_PATHS="${CHECKPOINT_PATHS} ${SM_EXPORT_ROOT}/sm05-all-r128-chat/merged"
  elif [[ $step == 12500 ]]; then
    CHECKPOINT_PATHS="${CHECKPOINT_PATHS} ${SM_EXPORT_ROOT}/sm${step::-2}-all-r128-chat/merged"
  else
    CHECKPOINT_PATHS="${CHECKPOINT_PATHS} ${SM_EXPORT_ROOT}/sm${step::-3}-all-r128-chat/merged"
  fi
done

python analyze/embeddings.py \
  --base-model "meta-llama/Llama-2-7b-hf" \
  --comparisons $CHECKPOINT_PATHS \
  --output "outputs/comparison-embeddings-snakmodel.json"