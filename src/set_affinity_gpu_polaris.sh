#!/bin/bash -l
num_gpus=$(nvidia-smi -L | wc -l)
gpu=$((num_gpus - 1 - PMI_LOCAL_RANK % num_gpus))
export CUDA_VISIBLE_DEVICES=$gpu
exec "$@"
