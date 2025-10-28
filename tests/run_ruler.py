import json
import sys
sys.path.append("../")
import python.sglang as sgl
from transformers import AutoTokenizer
def main():
    model_name = "Qwen/Qwen3-1.7B"
    llm = sgl.Engine(model_path=model_name, 
                    disable_cuda_graph=False,
                    page_size=16,
                    vortex_num_selected_pages=30,   
                    disable_overlap_schedule=True,
                    attention_backend="flashinfer",
                    enable_vortex_sparsity=True,
                    vortex_page_reserved_bos=1,
                    vortex_page_reserved_eos=1,
                    vortex_layers_skip=list(range(1)),
                    vortex_cg=True,
                    vortex_max_seq_lens=8192
                    )
    
    with open("validation.jsonl", "r", encoding="utf-8") as f:
        ruler_data = [json.loads(line)["input"] for line in f]
    
    texts = [
        [{"role":"user","content": x}] for x in ruler_data
    ]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompts = [
        tokenizer.apply_chat_template(
        text,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    ) for text in texts
    ]
    sampling_params = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_new_tokens": 64}
    with open("ruler_output.jsonl", "w", encoding="utf-8") as f:
            o = llm.generate(prompts, sampling_params)
            for item in o:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")
    

if __name__ == "__main__":
    main()
