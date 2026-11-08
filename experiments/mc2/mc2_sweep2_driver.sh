#!/bin/bash
cd /home/workspace/chenyun/hpa_cgo2027_workspace
DEV_ARGS=""
for i in $(seq 0 15); do DEV_ARGS="$DEV_ARGS --device /dev/davinci$i:/dev/davinci$i:rwm"; done
DEV_ARGS="$DEV_ARGS --device /dev/davinci_manager:/dev/davinci_manager:rwm --device /dev/devmm_svm:/dev/devmm_svm:rwm --device /dev/hisi_hdc:/dev/hisi_hdc:rwm"
export DEV_ARGS

mkdir -p results/mc2_sweep2/A results/mc2_sweep2/B

for ARM in A B; do
    echo "=== SWEEP2 ARM $ARM $(date +%H:%M:%S) ==="
    docker rm -f mc2sw2 2>/dev/null
    docker run -d --name mc2sw2 --network host --ipc host $DEV_ARGS \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
        -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
        -v /home/workspace/chenyun/hpa_cgo2027_workspace/models/OLMoE-1B-7B-0125-Instruct:/workspace/model:ro \
        -v /home/workspace/chenyun/hpa_cgo2027_workspace/mc2_sweep2.py:/experiment.py:ro \
        -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
        -e MC2_ARM=$ARM \
        hpa/vllm-ascend:v0.20.2rc1-a3-transformers5.9.0 \
        python3 -u /experiment.py
    DEADLINE=$((SECONDS + 3600))
    while [ $SECONDS -lt $DEADLINE ]; do
        [ "$(docker inspect -f {{.State.Status}} mc2sw2 2>/dev/null)" != "running" ] && break
        sleep 30
    done
    EXIT=$(docker inspect -f {{.State.ExitCode}} mc2sw2 2>/dev/null)
    docker logs mc2sw2 > results/mc2_sweep2/$ARM/container.log 2>&1
    docker rm mc2sw2 2>/dev/null
    echo "ARM $ARM exit=$EXIT $(date +%H:%M:%S)"
    [ "$EXIT" != "0" ] && echo FAILED && tail -5 results/mc2_sweep2/$ARM/container.log && break
done
echo "SWEEP2 COMPLETE"
