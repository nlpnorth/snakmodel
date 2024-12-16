#!/bin/bash

echo "============================"
echo "Plot SnakModel across Epochs"
echo "============================"

SCANDEVAL_RESULTS=/path/to/scandeval_benchmark_results.jsonl  # TODO: set to path on your machine
EXPORT_ROOT=/path/to/multiepoch/finetuning/run/  # TODO: set to path on your machine
EPOCHS=( $(seq 0 1 10) )
MODELS=("/path/to/nontuned/checkpoint")  # TODO: set to path on your machine
OUTPUT=outputs/scandeval-validation-snakmodel-instruction-epochs.pdf

for epoch in "${EPOCHS[@]}"; do
  if [[ $epoch == 0 ]]; then
    continue
  fi
  MODELS+=("${EXPORT_ROOT}/merged-epoch-${epoch}")
done

python plot/scandeval.py \
  --results "${SCANDEVAL_RESULTS}" \
  --models "${MODELS[@]}" \
  --x-title "Epochs" \
  --x-values "${EPOCHS[@]}" \
  --x-skip 1 \
  --legend \
  --output "${OUTPUT}"
