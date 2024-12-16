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

# https://huggingface.co/spaces/bigcode/near-deduplication
