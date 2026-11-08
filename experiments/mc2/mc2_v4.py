"""MC2 experiment v4: model-generic, arms A/B/C (native/fullmesh/hierarchy)."""
import os, json, time

ARM = os.environ.get("MC2_ARM", "A")
WORLD = int(os.environ.get("WORLD_SIZE", "16"))
MODEL = os.environ.get("MODEL_PATH", "/workspace/model")
MAXLEN = int(os.environ.get("MAX_MODEL_LEN", "2048"))
print(json.dumps({"event": "arm_start", "arm": ARM, "model": MODEL, "world": WORLD}), flush=True)

additional_config = {}
if ARM == "B":
    additional_config = {"mc2_comm_alg": "fullmesh"}
elif ARM == "C":
    additional_config = {"mc2_comm_alg": "hierarchy"}

kwargs = dict(
    model=MODEL,
    tensor_parallel_size=WORLD,
    max_model_len=MAXLEN,
    enforce_eager=True,
)
if additional_config:
    kwargs["additional_config"] = additional_config

from vllm import LLM

t0 = time.monotonic_ns()
llm = LLM(**kwargs)
t1 = time.monotonic_ns()
print(json.dumps({"event": "engine_init_ms", "value": (t1-t0)/1e6}), flush=True)

import torch, torch_npu
free, total = torch.npu.mem_get_info(0)
print(json.dumps({"event": "mem_after_init", "consumed_mb": (total-free)//(1024*1024)}), flush=True)

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

# Warmup
t0 = time.monotonic_ns()
llm.generate(prompts[:2], use_tqdm=False)
t1 = time.monotonic_ns()
print(json.dumps({"event": "warmup_ms", "value": (t1-t0)/1e6}), flush=True)

# Measured
for i in range(5):
    t0 = time.monotonic_ns()
    outputs = llm.generate(prompts, use_tqdm=False)
    t1 = time.monotonic_ns()
    dur = (t1-t0)/1e6
    tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(json.dumps({"event": "measured", "run": i, "duration_ms": round(dur,1),
                      "tokens": tokens, "tok_per_s": round(tokens/(dur/1000),1)}), flush=True)

free2, _ = torch.npu.mem_get_info(0)
print(json.dumps({"event": "arm_complete", "arm": ARM,
                  "init_consumed_mb": (total-free)//(1024*1024),
                  "steady_consumed_mb": (total-free2)//(1024*1024)}), flush=True)
