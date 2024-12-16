#!/bin/bash

echo "========================="
echo "ScandEval All Checkpoints"
echo "========================="

#
# SprogModel
#

HF_EXPORT_ROOT=/path/to/pretraining/hf_model_exports  # TODO: set to path on your machine
CHECKPOINT_ITERATIONS=( $(seq 500 500 12500) )

for checkpoint_iteration in "${CHECKPOINT_ITERATIONS[@]}"; do
  echo "ScandEval for Checkpoint ${checkpoint_iteration}."
  scandeval -m "${HF_EXPORT_ROOT}/checkpoint_${checkpoint_iteration}/" -l "da" --only-validation-split
done

#
# SnakModel
#

SM_EXPORT_ROOT=/path/to/instruction/checkpoints  # TODO: set to path on your machine
STEPS=( 0 500 )
STEPS+=( $(seq 1000 1000 12000) )
STEPS+=( 12500 )

for step in "${STEPS[@]}"; do
  echo "ScandEval for Checkpoint ${step}."
  if [[ $step == 0 ]]; then
    continue
  elif [[ $step == 500 ]]; then
    checkpoint_path="${SM_EXPORT_ROOT}/sm05-all-r128-chat/merged"
  elif [[ $step == 12500 ]]; then
    checkpoint_path="${SM_EXPORT_ROOT}/sm${step::-2}-all-r128-chat/merged"
  else
    checkpoint_path="${SM_EXPORT_ROOT}/sm${step::-3}-all-r128-chat/merged"
  fi
  scandeval -m "${checkpoint_path}/" -l "da" --only-validation-split
done