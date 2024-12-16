# Instruction-tuning

To run LoRA-based instruction fine-tuning on an LM checkpoint, run the following command:

```bash
python train/instructions.py \
  --model /path/to/checkpoint \
  --instruction-format chat \
  --lora-targets all \
  --lora-rank 128 \
  --output /path/to/output/directory
```

* `--instruction-format` specifies the format in which instruction datasets should be passed to the model;
* `--lora-targets` specifies which parameter types should be targeted for adaptation;
* `--lora-rank` specifies the LoRA rank to be applied

Once started, the script will automatically download the relevant instruction-tuning datasets and will convert them to the appropriate format.

The output directory will contain the relevant training files and hyperparameters, as well as a weight-merged version of the fine-tuned model in a `merged/` subdirectory.