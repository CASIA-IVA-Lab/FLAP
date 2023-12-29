#!/bin/bash

# Set common variables
model="decapoda-research/llama-30b-hf"
cuda_device=$1

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define function to run python command
run_python_command () {
    /data/anyongqi/miniconda3/envs/prune_llm/bin/python3.9 main.py \
    --model $model \
    --prune_method $1 \
    --pruning_ratio $2 \
    --remove_heads $3 \
    --metrics $4 \
    --structure $5 \
    --nsamples 1024 \
    --save_model "llm_weights/${1}_p${2}_${4}_${5}_llama_30b/" \
    --eval \
    --unstr
}

# llama-30b with flap pruning method (p=0.2/0.3/0.5, adaptive in all layers)
echo "Running with flap pruning method"
run_python_command "flap" 0.2 -1 "WIFV" "AL-AM" 
run_python_command "flap" 0.3 -1 "WIFV" "AL-AM" 
run_python_command "flap" 0.5 -1 "WIFV" "AL-AM" 

# llama-30b with wanda-sp pruning method (p=0.2/0.3/0.5, uniform in all layers)
echo "Running with wanda-sp pruning method"
run_python_command "wanda_sp" 0.2 -1 N/A N/A
run_python_command "wanda_sp" 0.3 -1 N/A N/A 
run_python_command "wanda_sp" 0.5 -1 N/A N/A 

# # llama-30b with mag-sp pruning method (p=0.2/0.3/0.5, uniform in all layers)
# echo "Running with magnitude pruning method"
# run_python_command "mag_sp" 0.2 -1 N/A N/A
# run_python_command "mag_sp" 0.3 -1 N/A N/A 
# run_python_command "mag_sp" 0.5 -1 N/A N/A

