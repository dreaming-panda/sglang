"""
Offline generation test using CPU-based KV cache with Vortex sparse attention.

This script tests the CPU memory pool implementation where:
1. KV cache is stored on CPU (pinned memory)
2. During prefill: KV computed on GPU then transferred to CPU
3. During decode: Sparse pages copied from CPU to GPU for attention
"""

import json
import sys
sys.path.append("../")
import python.sglang as sgl
from transformers import AutoTokenizer


def main():
    model_name = "Qwen/Qwen3-14B"

    # Engine configuration for CPU-based KV cache
    llm = sgl.Engine(
        model_path=model_name,
        disable_cuda_graph=True,
        page_size=16,
        mem_fraction_static=0.5,

        # Vortex sparse attention settings
        attention_backend="cpu_vtx_flashinfer",  # Use CPU-based backend
        enable_vortex_sparsity=True,
        vortex_num_selected_pages=30,
        vortex_page_reserved_bos=1,
        vortex_page_reserved_eos=1,
        vortex_layers_skip=[],  # Skip first layer for Vortex
        enable_cpu_vtx_cache=True,

        # Memory settings
        disable_overlap_schedule=True,
        kv_cache_dtype="auto",  # Use model's native dtype (bfloat16 for Qwen3)
    )

    # Load test content
    with open('story.txt', "r", encoding="utf-8") as file:
        content = file.read()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    texts = [
        [{"role": "user", "content": content}],
    ]

    prompts = [
        tokenizer.apply_chat_template(
            text,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        ) for text in texts
    ]

    # Test with multiple batches
    prompts = prompts * 16

    print(f"Testing CPU-based KV cache with {len(prompts)} prompts")
    print(f"Vortex config: {llm.server_args.vortex_num_selected_pages} selected pages, "
          f"{llm.server_args.vortex_page_reserved_bos} BOS, "
          f"{llm.server_args.vortex_page_reserved_eos} EOS")

    sampling_params = {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_new_tokens": 1024
    }

    with open("output_cpu.jsonl", "w", encoding="utf-8") as f:
        for iteration in range(8):
            print(f"\nIteration {iteration + 1}/8")
            o = llm.generate(prompts, sampling_params)
            for item in o:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")
            print(f"Completed iteration {iteration + 1}")

    print("\nGeneration complete! Output saved to output_cpu.jsonl")


if __name__ == "__main__":
    main()
