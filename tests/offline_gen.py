import json
import sys
sys.path.append("../")
import python.sglang as sgl
from transformers import AutoTokenizer
def main():
    model_name = "Qwen/Qwen3-1.7B"
    llm = sgl.Engine(model_path=model_name, 
                    disable_cuda_graph=True, 
                    page_size=16,
                    vortex_num_selected_pages=30,       
                    disable_overlap_schedule=True,
                    attention_backend="flashinfer",
                    enable_vortex_sparsity=True,
                    vortex_page_reserved_bos=1,
                    vortex_page_reserved_eos=1,
                    vortex_layers_skip=list(range(1))
                    )
    
    with open('story.txt', "r", encoding="utf-8") as file:
        content = file.read()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    texts = [
        [{"role":"user","content": content}],
    ]
    
    prompts = [
        tokenizer.apply_chat_template(
        text,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    ) for text in texts
    ]
    
    prompts = prompts * 4

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_new_tokens": 4096}
    with open("output.jsonl", "w", encoding="utf-8") as f:
        for _ in range(1):
            o = llm.generate(prompts, sampling_params)
            for item in o:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")

if __name__ == "__main__":
    main()
