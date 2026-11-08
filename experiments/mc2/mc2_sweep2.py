"""MC2 extended: batch size and prompt length sweep (fixed)."""
import os, json, time

ARM = os.environ.get("MC2_ARM", "A")
WORLD = 16
MODEL = "/workspace/model"

additional_config = {}
if ARM == "B":
    additional_config = {"mc2_comm_alg": "fullmesh"}

kwargs = dict(
    model=MODEL,
    tensor_parallel_size=WORLD,
    max_model_len=4096,
    enforce_eager=True,
)
if additional_config:
    kwargs["additional_config"] = additional_config

from vllm import LLM
llm = LLM(**kwargs)
print(json.dumps({"event": "engine_ready", "arm": ARM}), flush=True)

import torch, torch_npu
free0, total = torch.npu.mem_get_info(0)

base_mixed = ["The capital of France is", "Write Python code to sort", "Explain MoE",
              "What is TP?", "Describe NPU", "How does KV cache work",
              "Compare GPU and NPU", "What is a barrier", "Explain SDMA",
              "What is compaction", "Describe Atlas A3", "MoE routing strategies"]

configs = [
    ("bs1_short",  ["Hello"] * 1),
    ("bs1_long",   ["Write a detailed essay about the history of computer architecture from von Neumann to modern accelerators, covering at least 500 words with technical depth."] * 1),
    ("bs4_short",  ["Hello world"] * 4),
    ("bs4_medium", ["Explain the concept of mixture of experts in deep learning, including routing, load balancing, and communication patterns."] * 4),
    ("bs12_mixed", base_mixed),
    ("bs20_mixed", (base_mixed * 2)[:20]),
    ("bs32_mixed", (base_mixed * 3)[:32]),
]

for name, prompts in configs:
    # Warmup
    llm.generate(prompts[:min(2, len(prompts))], use_tqdm=False)

    results = []
    for run in range(3):
        t0 = time.monotonic_ns()
        outputs = llm.generate(prompts, use_tqdm=False)
        t1 = time.monotonic_ns()
        dur = (t1-t0)/1e6
        tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        results.append({"duration_ms": round(dur,1), "tokens": tokens,
                        "tok_per_s": round(tokens/(dur/1000),1)})

    free, _ = torch.npu.mem_get_info(0)
    print(json.dumps({"event": "config_result", "config": name, "arm": ARM,
                      "n_prompts": len(prompts), "runs": results,
                      "mem_consumed_mb": (total-free)//(1024*1024)}), flush=True)

print(json.dumps({"event": "sweep_complete", "arm": ARM}), flush=True)
