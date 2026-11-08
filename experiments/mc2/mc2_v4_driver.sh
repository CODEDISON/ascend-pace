#!/bin/bash
# MC2 v4 driver: phase 1 OLMoE arm C (hierarchy), phase 2 Qwen3-30B-A3B arms A/B
set -u
cd /home/workspace/chenyun/hpa_cgo2027_workspace

DEV_ARGS=""
for i in $(seq 0 15); do DEV_ARGS="$DEV_ARGS --device /dev/davinci$i:/dev/davinci$i:rwm"; done
DEV_ARGS="$DEV_ARGS --device /dev/davinci_manager:/dev/davinci_manager:rwm --device /dev/devmm_svm:/dev/devmm_svm:rwm --device /dev/hisi_hdc:/dev/hisi_hdc:rwm"
export DEV_ARGS

IMAGE=hpa/vllm-ascend:v0.20.2rc1-a3-transformers5.9.0
SCRIPT=/home/workspace/chenyun/hpa_cgo2027_workspace/mc2_v4.py
OLMOE=/home/workspace/chenyun/hpa_cgo2027_workspace/models/OLMoE-1B-7B-0125-Instruct
QWEN=/home/workspace/chenyun/models/Qwen3-30B-A3B-hf-ad44e-20261027-v1

run_arm() {
    local OUTDIR=$1 MODEL=$2 ARM=$3 NAME=$4
    mkdir -p "$OUTDIR"
    echo "=== $NAME arm $ARM start $(date +%H:%M:%S) ==="
    docker rm -f mc2v4 2>/dev/null
    docker run -d --name mc2v4 --network host --ipc host $DEV_ARGS \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
        -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
        -v "$MODEL":/workspace/model:ro \
        -v "$SCRIPT":/experiment.py:ro \
        -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
        -e MC2_ARM=$ARM -e WORLD_SIZE=16 \
        "$IMAGE" \
        python3 -u /experiment.py
    local DEADLINE=$((SECONDS + 3600))
    while [ $SECONDS -lt $DEADLINE ]; do
        [ "$(docker inspect -f {{.State.Status}} mc2v4 2>/dev/null)" != "running" ] && break
        sleep 30
    done
    local EXIT=$(docker inspect -f {{.State.ExitCode}} mc2v4 2>/dev/null)
    docker logs mc2v4 > "$OUTDIR/container.log" 2>&1
    docker rm mc2v4 2>/dev/null
    echo "=== $NAME arm $ARM exit=$EXIT $(date +%H:%M:%S) ==="
    if [ "$EXIT" != "0" ]; then
        echo "FAILED: $NAME arm $ARM"
        tail -8 "$OUTDIR/container.log"
    fi
}

# Phase 1: OLMoE hierarchy arm
run_arm results/mc2_v4/olmoe_C "$OLMOE" C "OLMoE"

# Phase 2: Qwen3-30B-A3B native + fullmesh
run_arm results/mc2_v4/qwen3_A "$QWEN" A "Qwen3"
run_arm results/mc2_v4/qwen3_B "$QWEN" B "Qwen3"

echo "MC2_V4 COMPLETE $(date +%H:%M:%S)"
