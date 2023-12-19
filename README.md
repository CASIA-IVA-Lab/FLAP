# Fluctuation-based Adaptive Structured Pruning for Large Language Models
Official PyTorch implementation of **FLAP** (**FL**uctuation-based **A**daptive Structured **P**runing)
--- 

We are currently in the process of preparing the pruning code associated with our project. This involves ensuring the code is clean, efficient, and well-documented to provide the best possible experience for users.


<p align="center">
<img src="overview.png" width=100% height=100% 
class="center">
</p>

## Setup
Installation instructions can be found in [INSTALL.md](INSTALL.md).

## Usage
The [scripts](scripts) directory contains all the bash commands to replicate the main results (Table 1 and Table 3) in our paper.

Below is an example command for pruning LLaMA-7B with FLAP, to achieve structured 50% pruning ratio.
```sh
python main.py \
    --model decapoda-research/llama-7b-hf \
    --prune_method "flap" \
    --pruning_ratio 0.5 \
    --remove_heads -1 \
    --metrics "WIFV" \
    --structure "AL-AM" \
    --nsamples 1024 \
    --save_model "llm_weights/flap_p0.5_WIFV_AL-AM_llama_7b/"
```
We provide a quick overview of the arguments:  
- `--model`: The identifier for the LLaMA model on the Hugging Face model hub.
- `--cache_dir`: Directory for loading or storing LLM weights. The default is `llm_weights`.
- `--prune_method`: We have implemented three pruning methods, namely [`flap`, `wanda_sp`, `mag_sp`]. The default is `flap`.
- `--pruning_ratio`: Denotes the percentage of weights to be pruned.
- `--remove_heads`: How many heads should be removed, only used in `UL-MM` and `AL-MM` to manual the ratio of Self-attn and MLP.
- `--metrics`: The pruning metric to choose, namely [`IFV`, `WIFV`, `WIFN`, `N/A`]. The default is `WIFV`.
- `--structure`: The global compressed model structure to choose, namely [`UL-UM`, `UL-MM`, `AL-MM`, `AL-AM`]. The default is `AL-AM`.
- `--unstr`: Whether to true prune the model or only mask the weight, default is `False`.
- `--eval`: Whether to eval the model on Wikitext2 to calculate the perplexity, default is `False`.
- `--save_model`: Specifies the directory where the pruned model will be stored.



## Acknowledgement
This repository is build upon the [Wanda](https://github.com/locuslab/wanda) repository.