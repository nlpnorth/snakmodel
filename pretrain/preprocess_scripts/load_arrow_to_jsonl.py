from datasets import load_from_disk
from tqdm import tqdm
import json

dataset = load_from_disk("deduplicated/all/")
with open("_pretraining_data/danish_pretraining_data.jsonl", "w") as fout, open("pretraining_data/danish_pretraining_data.txt", "w") as ftxt:
    for sentence in tqdm(dataset["text"], desc=f"Reading data..."):
        fout.write(json.dumps({"text": sentence}) + "\n")
        ftxt.write(sentence.strip() + "\n")
    