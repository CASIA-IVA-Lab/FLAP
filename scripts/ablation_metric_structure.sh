#!/bin/bash

# Set common variables
model="decapoda-research/llama-7b-hf"
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
    --save_model "llm_weights/${1}_p${2}_${4}_${5}_llama_7b/" \
    --eval
}

# llama-7b with flap pruning method (metric & structure ablation)
echo "Running with flap pruning method"
run_python_command "flap" 0.5 -1 "WIFV" "UL-UM" 
run_python_command "flap" 0.5 384 "WIFV" "UL-MM" 
run_python_command "flap" 0.5 380 "WIFV" "AL-MM" 
run_python_command "flap" 0.5 -1 "WIFV" "AL-AM" 
run_python_command "flap" 0.5 384 "IFV" "UL-UM" 
run_python_command "flap" 0.5 380 "IFV" "UL-MM" 
run_python_command "flap" 0.5 -1 "IFV" "AL-MM" 
run_python_command "flap" 0.5 -1 "IFV" "AL-AM" 
run_python_command "flap" 0.5 -1 "WIFN" "UL-UM" 
run_python_command "flap" 0.5 384 "WIFN" "UL-MM" 
run_python_command "flap" 0.5 380 "WIFN" "AL-MM" 
run_python_command "flap" 0.5 -1 "WIFN" "AL-AM" 
