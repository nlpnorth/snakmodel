# Pre-training Codebase for Snakmodel

## Cloning this repo

This repo contains a submodule which contains the source code for Megatron-LLM.
To clone both repos (this + the submodule) use the following command:

```bash
$ git clone --recurse-submodules git@github.com:nlpnorth/snakmodel.git
```

Or use these two commands:

```bash
$ git clone git@github.com:nlpnorth/snakmodel.git
$ git submodule update --init --recursive
```

## 1. Preprocessing the data

Downloading the data can be done using the scripts in `pretrain/download_scripts/` (where there is a README with further explanations).

Preprocessing scripts can be found in `snakmodel/pretrain/preprocess_scripts/` or a single script in `snakmodel/pretrain/model_training_scripts/0.preprocess_data.sh`

We expect the data to be raw `.txt` files with on each line a document (e.g., sentence). 

### Step 1: Change raw text files to HF format.
You can process the `.txt` files with 
```
python preprocess_to_hf.py \
    --data_path $PATH_TO_DATA_FOLDER \
    --dataset_names "dataset_a dataset_b dataset_c" 
```

This creates a folder `sprogmodel/hf_datasets_format/` containing HF-formatted dataset arrow files. 

### Step 2: Deduplication, we had 128 CPU cores and 1TB of memory.

Then, we use the deduplication library by https://github.com/ChenghaoMou/text-dedup and run the following code snippet found in `preprocess_scripts/deduplicate.sh`:

```
python -m text_dedup.minhash \
            --path "hf_datasets_format/all/train/train/" \
            --split "train" \
            --cache_dir "/data/sprogmodel/.cache/" \
            --local \
            --output "deduplicated/all/" \
            --column "text" \
            --seed 42 \
            --batch_size 4096 \
            --num_perm 64 \
            --threshold 0.85 \
```

### Step 3: Merge HF arrow files together to one jsonl and also a text file.
```
mkdir _pretraining_data
python preprocess_scripts/load_arrow_to_jsonl.py
```

## 2. Running the code

We followed the exact same steps in https://epfllm.github.io/Megatron-LLM/guide/getting_started.html

You can find the exact step-by-step scripts in `sprogmodel/model_training_scripts` indicated with an integer from 1-9.

The steps are:
```
1.start_docker.sh
Start a docker container with a predefined amount of shm-size (128 in this case). Then also install the requirements from Megatron-LLM

2.preprocess_data.sh
Will pre-tokenize the data we created above in "1. Preprocessing the data", we used a chunk size of 64 and 4 workers.

3.weight_conversion.sh
Here we expect you to have downloaded the LLaMa2 weights from huggingface found in the online documentation from Megatron-LLM. This code will convert the HF model weights to Megatron weights.

4.conversion_check_megatron.sh
Sanity check whether the conversion went correctly. You can edit the following arguments: --nproc_per_node 4 --nnodes 1, depending on the number of GPUs you have (--nproc_per_node) and the number of nodes/machines with the GPUs (--nnodes). E.g., if you have 2 machines with each 4 GPUs then it's --nproc_per_node 4 --nnodes 2.

5.model_sharding.sh
Shard the Megatron-converted weights into shards. Again, feel free to edit the following: --target_tensor_parallel_size 4 \ --target_pipeline_parallel_size 1 \. TP is the number of GPUs you have and PP is the number of machines.

6.training.sh
Training the model, feel free to edit the arguments.

7.model_merging.sh
Merge the sharded model back to one piece.

8.conversion_to_hf.sh
Convert the merged model to HF format.

9.push_to_hub.sh
Push the model to the HF hub.
```

### What if my run crashes?

If you have a checkpoint, you can re-start from the checkpoint by just running `6.training.sh`.


## Credits

We use the text deduplication library by Chenghao Mou;

```
@software{chenghao_mou_2023_8364980,
  author       = {Chenghao Mou and
                  Chris Ha and
                  Kenneth Enevoldsen and
                  Peiyuan Liu},
  title        = {ChenghaoMou/text-dedup: Reference Snapshot},
  month        = sep,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {2023.09.20},
  doi          = {10.5281/zenodo.8364980},
  url          = {https://doi.org/10.5281/zenodo.8364980}
}
```

We use the Megatron-LLM library by EPFL:

```
@software{epfmgtrn,
  author       = {Alejandro Hernández Cano  and
                  Matteo Pagliardini  and
                  Andreas Köpf  and
                  Kyle Matoba  and
                  Amirkeivan Mohtashami  and
                  Xingyao Wang  and
                  Olivia Simin Fan  and
                  Axel Marmet  and
                  Deniz Bayazit  and
                  Igor Krawczuk  and
                  Zeming Chen  and
                  Francesco Salvi  and
                  Antoine Bosselut  and
                  Martin Jaggi},
  title        = {epfLLM Megatron-LLM},
  year         = 2023,
  url          = {https://github.com/epfLLM/Megatron-LLM}
}
```
