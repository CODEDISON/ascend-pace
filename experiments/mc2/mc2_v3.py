"""MC2 experiment v3 - exact same params as successful TP16 test."""
import os, json, time, sys

ARM = os.environ.get("MC2_ARM", "A")
WORLD = int(os.environ.get("WORLD_SIZE", "16"))
print(json.dumps({"event": "arm_start", "arm": ARM}), flush=True)

# MC2 config
additional_config = {}
if ARM == "B":
    additional_config = {"mc2_comm_alg": "fullmesh"}
elif ARM == "C":
    additional_config = {"mc2_comm_alg": "hierarchy"}

# Build kwargs - only add additional_config if non-empty
kwargs = dict(
    model="/workspace/model",
    tensor_parallel_size=WORLD,
    max_model_len=2048,
    enforce_eager=True,
)
if additional_config:
    kwargs["additional_config"] = additional_config

from vllm import LLM

t0 = time.monotonic_ns()
llm = LLM(**kwargs)
t1 = time.monotonic_ns()
print(json.dumps({"event": "engine_init_ms", "value": (t1-t0)/1e6}), flush=True)

# Memory after init
import torch, torch_npu
free, total = torch.npu.mem_get_info(0)
print(json.dumps({"event": "mem_after_init", "consumed_mb": (total-free)//(1024*1024)}), flush=True)

prompts = ["The capital of France is", "Write Python code", "Explain MoE"] * 4

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
