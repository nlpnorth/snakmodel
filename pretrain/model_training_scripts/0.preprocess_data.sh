# Have all data in `/raw_data/all/` as one big file.
# We assume data to be already LangID'd.

# Step 1: Change raw text files to HF format.
python preprocess_scripts/preprocess_to_hf.py \
    --data_path _raw_data \
    --dataset_names "all" 

# Step 2: Deduplication, we had 128 CPU cores and 1TB of memory.
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

# Step 3: Merge HF arrow files together to one jsonl and also a text file.
mkdir _pretraining_data
python preprocess_scripts/load_arrow_to_jsonl.py