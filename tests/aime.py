import json
import sys
sys.path.append("../")
import python.sglang as sgl
from transformers import AutoTokenizer
import os
from tqdm import tqdm
import time
import torch
import argparse
import traceback
os.environ["TOKENIZERS_PARALLELISM"] = "false"
MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()

from datasets import load_dataset, Dataset, concatenate_datasets
def generate_requests(dataset: Dataset, field_name: str, data_format: str, trial: int = 1, rank: int = 0, world_size: int = 1):
    requests = []

    # Step 1: Expand dataset trial times
    if trial > 1:
        dataset = Dataset.from_dict(dataset.to_dict().copy())  # ensure copy
        datasets = [dataset] * trial
        dataset = concatenate_datasets(datasets)
    
    total = len(dataset)
    
    # Step 2: Partition across ranks
    per_proc = total // world_size
    remainder = total % world_size
    start = rank * per_proc + min(rank, remainder)
    end = start + per_proc + (1 if rank < remainder else 0)
    subset = dataset.select(list(range(start, end)))

    # Step 3: Format requests
    for data in subset:
        conversations = [
            {"role": "user", "content": data_format.format(Question=data[field_name])}
        ]
        data["conversations"] = conversations
        requests.append(data)

    return requests

def main():
    parser = argparse.ArgumentParser(description="Run AIME benchmark with SGLang")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-14B", help="Model name or path")
    parser.add_argument("--attention-backend", type=str, default="cpu_vtx_flashinfer", help="Attention backend")
    parser.add_argument("--mem-fraction-static", type=float, default=0.8, help="Static memory fraction")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum number of new tokens to generate")
    parser.add_argument("--disable-cuda-graph", action="store_true", default=False, help="Disable CUDA graph (default: enabled)")
    args = parser.parse_args()

    model_name = args.model_name
    name = args.attention_backend
    mem_fraction_static = args.mem_fraction_static
    max_new_tokens = args.max_new_tokens
    disable_cuda_graph = args.disable_cuda_graph
    enable_vortex_sparsity = True

    output_dir = f"DATA/{model_name}/AIME24/gpu_{mem_fraction_static}/max_tokens_{max_new_tokens}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Arguments: model={model_name}, backend={name}, mem_fraction={mem_fraction_static}, max_tokens={max_new_tokens}")

    if name not in ["cpu_vtx_flashinfer", "flashinfer"]:
        enable_vortex_sparsity = False
        attention_backend = "flashinfer"
    else:
        attention_backend = name

    llm = None
    try:
        print(f"Initializing SGLang Engine...")
        # Create crash dump folder
        crash_dump_folder = f"{output_dir}/crash_dumps"
        os.makedirs(crash_dump_folder, exist_ok=True)

        llm = sgl.Engine(model_path=model_name,
                        disable_cuda_graph=disable_cuda_graph,
                        page_size=16,
                        mem_fraction_static=mem_fraction_static,
                        cpu_mem_fraction=0.6,
                        vortex_topk_val=30,
                        disable_overlap_schedule=True,
                        attention_backend=attention_backend,
                        enable_vortex_sparsity=enable_vortex_sparsity,
                        vortex_page_reserved_bos=1,
                        vortex_page_reserved_eos=1,
                        vortex_layers_skip=list(range(1)),
                        enable_cpu_vtx_cache=True,
                        vortex_module_name="block_sparse_attention",
                        vortex_max_seq_lens=-1,
                        vortex_cpu_percentage=0.8,
                        tp_size=1,
                        log_level="debug",
                        crash_dump_folder=crash_dump_folder
                        )
        print("Engine initialized successfully!")

        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")

        requests = generate_requests(dataset, "problem", MATH_QUERY_TEMPLATE)


        texts = [
            x["conversations"] for x in requests
        ]

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        prompts = [
            tokenizer.apply_chat_template(
            text,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        ) for text in texts
        ] * 8

        print(f"Starting generation with {len(prompts)} prompts...")
        sampling_params = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_new_tokens": max_new_tokens}
        total_tokens = 0
        total_time = 0.0
        start = time.perf_counter()
        o = llm.generate(prompts, sampling_params)
        elapsed = time.perf_counter() - start
        total_time += elapsed
        e2e_time = 0

        print(f"Generation completed in {elapsed:.2f}s")
        with open(f"{output_dir}/{name}.jsonl", "w", encoding="utf-8") as f:
            for item in o:
                total_tokens += item["meta_info"]["completion_tokens"]
                e2e_time = max(e2e_time, item["meta_info"]["e2e_latency"])
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")

            meta_data = {"e2e_time": e2e_time, "total_time": total_time, "total_tokens": total_tokens, "throughput": total_tokens / total_time}
            json.dump(meta_data, f, ensure_ascii=False)
            f.write("\n")

        print(f"Results saved to {output_dir}/{name}.jsonl")
        print(f"Total tokens: {total_tokens}, Throughput: {total_tokens / total_time:.2f} tokens/s")

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        print("Full traceback:")
        traceback.print_exc()
        raise
    finally:
        if llm is not None:
            try:
                llm.shutdown()
            except Exception as e:
                print(f"Error during shutdown: {e}")

if __name__ == "__main__":
    main()
