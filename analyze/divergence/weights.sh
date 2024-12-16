#!/bin/bash

echo "============================================"
echo "Analyze Weight Divergence across Checkpoints"
echo "============================================"

#
# SprogModel
#
HF_EXPORT_ROOT=/path/to/pretraining/hf_model_exports  # TODO: set to path on your machine
CHECKPOINT_ITERATIONS=( $(seq 500 500 12500) )

SPROGMODEL_PATHS=""
for checkpoint_iteration in "${CHECKPOINT_ITERATIONS[@]}"; do
  SPROGMODEL_PATHS="${SPROGMODEL_PATHS} ${HF_EXPORT_ROOT}/checkpoint_${checkpoint_iteration}/"
done

python analyze/weights.py \
  --base-model "meta-llama/Llama-2-7b-hf" \
  --comparisons $SPROGMODEL_PATHS \
  --output "outputs/comparison-weights-llama2-sprogmodel.json"

#
# SnakModel
#
SM_EXPORT_ROOT=/path/to/instruction/checkpoints  # TODO: set to path on your machine
STEPS=( 0 500 )
STEPS+=( $(seq 1000 1000 12000) )
STEPS+=( 12500 )

SNAKMODEL_PATHS="${SM_EXPORT_ROOT}/llama2-base-snak/merged"
for step in "${STEPS[@]}"; do
  if [[ $step == 0 ]]; then
    continue
  elif [[ $step == 500 ]]; then
    snakmodel_path="${SM_EXPORT_ROOT}/sm05-all-r128-chat/merged"
  elif [[ $step == 12500 ]]; then
    snakmodel_path="${SM_EXPORT_ROOT}/sm${step::-2}-all-r128-chat/merged"
  else
    snakmodel_path="${SM_EXPORT_ROOT}/sm${step::-3}-all-r128-chat/merged"
  fi
  # compare SprogModel and SnakModel at the same step
  python analyze/weights.py \
    --base-model "${HF_EXPORT_ROOT}/checkpoint_${step}" \
    --comparisons "${snakmodel_path}" \
    --output "outputs/comparison-weights-sprogmodel-snakmodel-${step}.json"
  SNAKMODEL_PATHS="${SNAKMODEL_PATHS} ${snakmodel_path}"
done

# compare Llama (base) and SnakModels
python analyze/embeddings.py \
  --base-model "meta-llama/Llama-2-7b-hf" \
  --comparisons $SNAKMODEL_PATHS \
  --output "outputs/comparison-weights-llama2-snakmodel.json"