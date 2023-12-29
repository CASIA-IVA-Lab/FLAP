from transformers import AutoTokenizer, AutoModelForCausalLM
import torch   

device = torch.device('cuda:2')
model = torch.load('/data/anyongqi/flap/llm_weights/c4_flap_p0.5_WIFV_AL-AM_llama_7b/pruned_model.pt', map_location=device)
model.eval()

# model = AutoModelForCausalLM.from_pretrained(
#         "decapoda-research/llama-7b-hf", 
#         torch_dtype=torch.float16, 
#         cache_dir="llm_weights", 
#         low_cpu_mem_usage=True, 
#         # device_map="auto"
#     )

# device = torch.device("cuda:0")
# model.to(device)
# model.eval()

tokenizer = AutoTokenizer.from_pretrained("decapoda-research/llama-7b-hf", use_fast=False)

generate_kwargs = {
    "max_new_tokens": 300,
    "min_new_tokens": 200,
    "temperature": 0.1,
    "do_sample": False, # The three options below used together leads to contrastive search
    "top_k": 3,
    "penalty_alpha": 0.6,
    #"no_repeat_ngram_size": no_repeat_ngram_size,
    #**generation_config,
}

prompts = ["AI can create a logo in seconds.",
        "What is McDonald's?",
        ]
for prompt in prompts:
    with torch.no_grad():
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        assert len(input_ids) == 1, len(input_ids)
        if input_ids[0][-1] == 2: # 2 is EOS, hack to remove. If the prompt is ending with EOS, often the generation will stop abruptly.
            input_ids = input_ids[:, :-1]
        input_ids = input_ids.to(device)
        generated_ids = model.generate(
            input_ids,
            **generate_kwargs
        )
        result = tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)
        print(result)