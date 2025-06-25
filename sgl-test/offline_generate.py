# launch the offline engine
import asyncio
import sys
sys.path.append("../")
import io
import os
from PIL import Image
import requests
import python.sglang as sgl
from transformers import AutoTokenizer
def main():
    model_name = "/home/zhuominc/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/0060bc56d46589041c1048efd1a397421b1142b5"
    llm = sgl.Engine(model_path=model_name, 
        disable_cuda_graph=True, 
        page_size=4, 
        disable_overlap_schedule=True,
        enable_block_sparse_attention=True,
        num_active_kv_blocks=5,
        sparse_attention_layer_skip=[0,1])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    texts = [
        [{"role":"user","content":"Hello, what is your name?"}],
        [{"role":"user","content":"Who is the president of the United States?"}],
        [{"role":"user","content":"What is the future of AI?"}],
    ]
    
    prompts = [
        tokenizer.apply_chat_template(
        text,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    ) for text in texts
    ]
    
    prompts = prompts * 10

    sampling_params = {"temperature": 1e-7, "top_p": 0.95, "max_new_tokens": 128}

    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
if __name__ == "__main__":
    main()