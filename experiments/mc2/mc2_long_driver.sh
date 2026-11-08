#!/bin/bash
# MC2 long-output driver: sustained decoding comparison (OLMoE TP16)
set -u
cd /home/workspace/chenyun/hpa_cgo2027_workspace

DEV_ARGS=""
for i in $(seq 0 15); do DEV_ARGS="$DEV_ARGS --device /dev/davinci$i:/dev/davinci$i:rwm"; done
DEV_ARGS="$DEV_ARGS --device /dev/davinci_manager:/dev/davinci_manager:rwm --device /dev/devmm_svm:/dev/devmm_svm:rwm --device /dev/hisi_hdc:/dev/hisi_hdc:rwm"
export DEV_ARGS

IMAGE=hpa/vllm-ascend:v0.20.2rc1-a3-transformers5.9.0
SCRIPT=/home/workspace/chenyun/hpa_cgo2027_workspace/mc2_long.py
OLMOE=/home/workspace/chenyun/hpa_cgo2027_workspace/models/OLMoE-1B-7B-0125-Instruct

for ARM in A B; do
    OUTDIR=results/mc2_long/$ARM
    mkdir -p "$OUTDIR"
    echo "=== LONG arm $ARM start $(date +%H:%M:%S) ==="
    docker rm -f mc2lg 2>/dev/null
    docker run -d --name mc2lg --network host --ipc host $DEV_ARGS \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
        -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
        -v "$OLMOE":/workspace/model:ro \
        -v "$SCRIPT":/experiment.py:ro \
        -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
        -e MC2_ARM=$ARM -e WORLD_SIZE=16 \
        "$IMAGE" \
        python3 -u /experiment.py
    DEADLINE=$((SECONDS + 5400))
    while [ $SECONDS -lt $DEADLINE ]; do
        [ "$(docker inspect -f {{.State.Status}} mc2lg 2>/dev/null)" != "running" ] && break
        sleep 30
    done
    EXIT=$(docker inspect -f {{.State.ExitCode}} mc2lg 2>/dev/null)
    docker logs mc2lg > "$OUTDIR/container.log" 2>&1
    docker rm mc2lg 2>/dev/null
    echo "=== LONG arm $ARM exit=$EXIT $(date +%H:%M:%S) ==="
    if [ "$EXIT" != "0" ]; then
        echo "FAILED arm $ARM"
        tail -8 "$OUTDIR/container.log"
        break
    fi
done
echo "LONG COMPLETE $(date +%H:%M:%S)"
