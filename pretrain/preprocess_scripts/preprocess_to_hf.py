from datasets import load_dataset, Features, Value, Dataset, DatasetDict, load_from_disk, concatenate_datasets
import os
import time
import argparse

def save_to_disk_hf(root: str, file: str, data_path: str):
    # Extract the dataset name (source) from the filename
    dataset_name = file.split(".")[0]

    # Load the dataset from the file
    data = load_dataset(
        "text",
        features=Features({"text": Value(dtype="string", id=None)}),
        data_dir=root,
        data_files=file,
        cache_dir="/data/sprogmodel/.cache/",
    )

    # Add "source" metadata to each entry in the dataset
    data_with_source = data.map(lambda example: {"source": dataset_name, **example})

    # Save dataset with source metadata to disk
    dataset_path = f"hf_datasets_format/{data_path}/train/"
    os.makedirs(dataset_path, exist_ok=True)
    data_with_source.save_to_disk(dataset_path, max_shard_size="5GB")


def iterate_files_in_directory(directory, data_path):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".danish") and file.startswith(data_path): 
                text_directory = os.path.join("hf_datasets_format", data_path)
                os.makedirs(text_directory, exist_ok=True)  # Create directories if they don't exist
                print(f"Currently processing {file}")
                
                save_to_disk_hf(root, file, data_path)

def combine_all_datasets(dataset_names):
    # Paths to all processed datasets
    dataset_paths = [f"hf_datasets_format/{name}/train" for name in dataset_names if name != "all"]
    datasets_list = []

    for dp in dataset_paths:
        if os.path.exists(dp):
            ds = load_from_disk(dp)
            datasets_list.append(ds)

    # Concatenate all datasets
    if datasets_list:
        combined = concatenate_datasets(datasets_list)
        os.makedirs("hf_datasets_format/all/train/", exist_ok=True)
        combined.save_to_disk("hf_datasets_format/all/train/")
        print("All datasets combined and saved to hf_datasets_format/all/train/")
    else:
        print("No datasets found to combine.")

def main(args):
    # paths = ["bookshop", "twitter", "dawiki", "cc100", "culturax", "opensubtitle", "reddit", "gigaword", "all"]
    dataset_names = args.dataset_names.split()

    # Process each dataset individually
    for name in dataset_names:
        if name != "all":  # We only combine into "all" after processing others
            start = time.time()

            # Create HF text files 
            if not os.path.exists(f"hf_datasets_format/{name}"):
                dir = f"{args.data_path}"
                iterate_files_in_directory(directory=dir, data_path=name)

            end = time.time()
            diff = end - start
            print(f"{name} | processing time: {diff:.2f} s")

    # If "all" is among the requested dataset names, combine all processed datasets
    if "all" in dataset_names:
        combine_all_datasets(dataset_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sprogmodel data preprocessing")
    parser.add_argument("--data_path", nargs="?", type=str, help="Path to data files.")
    parser.add_argument("--dataset_names", nargs="?", type=str, help="Names of datasets separated by whitespace.")

    args = parser.parse_args()
    main(args)
