#!/bin/bash
BASEDIR=$(dirname "$0")

EXECUTION_MODES=( "Performance" "Accuracy" )
MODEL_LIST=( "resnet50" "retinanet" "3d-unet" "bert" )
DEVICE_LIST=( "CPU" )
SCENARIOS=( "SingleStream" "MultiStream" "Offline" "Server" )

for MODEL in "${MODEL_LIST[@]}"
do
    for DEVICE in "${DEVICE_LIST[@]}"
    do
        for EXECUTION_MODE in "${EXECUTION_MODES[@]}"
        do
            for SCENARIO in "${SCENARIOS[@]}"
            do
                ${BASEDIR}/run.sh -e ${EXECUTION_MODE} -m ${MODEL} -d ${DEVICE} -s ${SCENARIO}
            done
        done
    done
done
