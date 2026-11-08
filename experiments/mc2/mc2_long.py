"""MC2 long-output experiment: does MC2 help under sustained decoding?

Longer outputs = more decode steps = more MoE all-to-all dispatches per request.
If communication were the bottleneck, MC2's optimized dispatch should shine here.
"""
import os, json, time

ARM = os.environ.get("MC2_ARM", "A")
WORLD = int(os.environ.get("WORLD_SIZE", "16"))
MODEL = os.environ.get("MODEL_PATH", "/workspace/model")
print(json.dumps({"event": "arm_start", "arm": ARM, "model": MODEL}), flush=True)

additional_config = {}
if ARM == "B":
    additional_config = {"mc2_comm_alg": "fullmesh"}
elif ARM == "C":
    additional_config = {"mc2_comm_alg": "hierarchy"}

kwargs = dict(
    model=MODEL,
    tensor_parallel_size=WORLD,
    max_model_len=2048,
    enforce_eager=True,
)
if additional_config:
    kwargs["additional_config"] = additional_config

from vllm import LLM
from vllm.sampling_params import SamplingParams

t0 = time.monotonic_ns()
llm = LLM(**kwargs)
t1 = time.monotonic_ns()
print(json.dumps({"event": "engine_init_ms", "value": (t1-t0)/1e6}), flush=True)

import torch, torch_npu
free, total = torch.npu.mem_get_info(0)

prompts = [
    "The capital of France is",
    "Write Python code to sort",
    "Explain MoE",
    "What is TP?",
    "Describe NPU",
    "How does KV cache work",
    "Compare GPU and NPU",
    "What is a barrier",
    "Explain SDMA",
    "What is compaction",
    "Describe Atlas A3",
    "MoE routing strategies",
]

# Sustained-decoding configs: longer max_tokens = more decode steps
# 12 prompts x max_tokens: 16 -> ~192 tokens, 64 -> ~768, 256 -> ~3072
configs = [
    ("mt16", 16),
    ("mt64", 64),
    ("mt256", 256),
]

sp_cache = {}
for name, mt in configs:
    sp = SamplingParams(max_tokens=mt, temperature=0.0)
    sp_cache[name] = sp
    # Warmup with this config (first run of each config untimed)
    llm.generate(prompts[:2], sampling_params=sp, use_tqdm=False)

    results = []
    for run in range(3):
        t0 = time.monotonic_ns()
        outputs = llm.generate(prompts, sampling_params=sp, use_tqdm=False)
        t1 = time.monotonic_ns()
        dur = (t1-t0)/1e6
        tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        results.append({"duration_ms": round(dur,1), "tokens": tokens,
                        "tok_per_s": round(tokens/(dur/1000),1),
                        "decode_steps": mt})
    free, _ = torch.npu.mem_get_info(0)
    print(json.dumps({"event": "config_result", "config": name, "arm": ARM,
                      "max_tokens": mt, "n_prompts": len(prompts),
                      "runs": results,
                      "mem_consumed_mb": (total-free)//(1024*1024)}), flush=True)

free2, _ = torch.npu.mem_get_info(0)
print(json.dumps({"event": "arm_complete", "arm": ARM,
                  "steady_consumed_mb": (total-free2)//(1024*1024)}), flush=True)
