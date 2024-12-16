docker run --gpus all -it --shm-size=128gb --rm -v /data/sprogmodel/:/mpt/Megatron-LLM nvcr.io/nvidia/pytorch:23.07-py3

# Run the following piece of code inside the docker container:

# cd ../mpt/Megatron-LLM/Megatron-LLM
# pip3 install -r requirements.txt