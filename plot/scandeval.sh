#!/bin/bash

echo "=========================="
echo "Plot ScandEval across Time"
echo "=========================="

echo "Plotting SprogModel:"

SCANDEVAL_RESULTS=/path/to/scandeval_benchmark_results.jsonl  # TODO: set to path on your machine
HF_EXPORT_ROOT=/path/to/pretraining/hf_model_exports  # TODO: set to path on your machine
STEPS=( $(seq 0 500 12500) )
MODELS=("meta-llama/Llama-2-7b-hf")
OUTPUT=outputs/scandeval-validation-sprogmodel.pdf

for step in "${STEPS[@]}"; do
  if [[ $step == 0 ]]; then
    continue
  fi
  MODELS+=("${HF_EXPORT_ROOT}/checkpoint_${step}")
done

python plot/scandeval.py \
  --results "${SCANDEVAL_RESULTS}" \
  --models "${MODELS[@]}" \
  --x-values "${STEPS[@]}" \
  --output "${OUTPUT}"

echo "-------------------"
echo "Plotting SnakModel:"

SM_EXPORT_ROOT=/path/to/instruction/checkpoints  # TODO: set to path on your machine
STEPS=( 0 500 )
STEPS+=( $(seq 1000 1000 12000) )
STEPS+=( 12500 )
MODELS=("${SM_EXPORT_ROOT}/llama2-base-snak/merged")
OUTPUT=outputs/scandeval-validation-snakmodel.pdf

for step in "${STEPS[@]}"; do
  if [[ $step == 0 ]]; then
    continue
  elif [[ $step == 500 ]]; then
    MODELS+=("${SM_EXPORT_ROOT}/sm05-all-r128-chat/merged")
  elif [[ $step == 12500 ]]; then
    MODELS+=("${SM_EXPORT_ROOT}/sm${step::-2}-all-r128-chat/merged")
  else
    MODELS+=("${SM_EXPORT_ROOT}/sm${step::-3}-all-r128-chat/merged")
  fi
done

python plot/scandeval.py \
  --results "${SCANDEVAL_RESULTS}" \
  --models "${MODELS[@]}" \
  --x-values "${STEPS[@]}" \
  --output "${OUTPUT}"
