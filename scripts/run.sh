#!/bin/bash
BASEDIR=$(dirname "$0")

while getopts m:d:s:e: flag
do
    case "${flag}" in
        m) MODELNAME=${OPTARG};;
        d) DEVICE=${OPTARG};;
        s) SCENARIO=${OPTARG};;
        e) MODE=${OPTARG};;
    esac
done

if [[ -z ${MODELNAME} ]] || [[ -z ${DEVICE} ]] || [[ -z ${SCENARIO} ]] || [[ -z ${MODE} ]]; then
    echo "Usage: run.sh -m <model> -d <device> -s <scenario> -e <mode>"
    exit
fi

echo "Running model ${MODELNAME} on ${DEVICE} device using ${SCENARIO} scenario in ${MODE} mode..."

MODEL_PATH=${MODEL_DIR}/${MODELNAME}/${MODELNAME}.xml

# Provide appropriate mlperf config
MLPERF_CONF=${MLPERF_DIR}/dependencies/mlperf-inference/mlperf.conf

mkdirs() {
   echo "Create directory $1..."
   if [ -d "$1" ]; then
       rm -Rf $1;
   fi
   mkdir -p $1
}

OUT_DIR=${RESULTS_DIR}/${MODELNAME}/${DEVICE}/${MODE}/${SCENARIO}
mkdirs ${OUT_DIR}

ACC_FILENAME=${OUT_DIR}/mlperf_log_accuracy.json

# By default C++ launcher is used
OV_MLPERF_BIN=${BUILD_DIRECTORY}/src/Release/ov_mlperf

case "${MODELNAME}" in
"resnet50")
    REF_DIR=${MLPERF_DIR}/dependencies/mlperf-inference/vision/classification_and_detection
    DATASET=imagenet
    WARMUP_ITERS=500
    TOTAL_SAMPLE_COUNT=50000
    PERF_SAMPLE_COUNT=1024
    export DATA_PATH=${BUILD_DIRECTORY}/datasets/${DATASET}/ILSVRC2012_img_val
    ACC_SCRIPT="${REF_DIR}/tools/accuracy-imagenet.py"
    ACC_ARGS="--imagenet-val-file ${BUILD_DIRECTORY}/datasets/${DATASET}/val_data/val_map.txt --mlperf-accuracy-file ${ACC_FILENAME}"
    ;;
"retinanet")
    REF_DIR=$MLPERF_DIR/dependencies/mlperf-inference/vision/classification_and_detection
    DATASET=openimages
    TOTAL_SAMPLE_COUNT=24781
    PERF_SAMPLE_COUNT=64
    export DATA_PATH=${BUILD_DIRECTORY}/datasets/${DATASET}/openimages_v6/
    ACC_SCRIPT="${REF_DIR}/tools/accuracy-openimages.py"
    ACC_ARGS="--openimages-dir ${DATA_PATH} --output-file ${OUT_DIR}/openimages-results.json --mlperf-accuracy-file ${ACC_FILENAME}"
    ;;
"3d-unet")
    REF_DIR=${MLPERF_DIR}/dependencies/mlperf-inference/vision/medical_imaging/3d-unet-kits19
    export PYTHONPATH=$PYTHONPATH:${REF_DIR}
    OV_MLPERF_BIN="python -u ${BUILD_DIRECTORY}/python/3d-unet-kits19/run.py"
    DATASET=kits
    TOTAL_SAMPLE_COUNT=42
    PERF_SAMPLE_COUNT=${TOTAL_SAMPLE_COUNT}
    PREPROC_DATA_PATH=${BUILD_DIRECTORY}/datasets/${DATASET}/preprocessed_data
    POSTPROC_DATA_PATH=${BUILD_DIRECTORY}/datasets/${DATASET}/postprocessed_data
    mkdirs ${POSTPROC_DATA_PATH}
    export DATA_PATH=${PREPROC_DATA_PATH}/preprocessed_files.pkl
    cp -r ${REF_DIR}/meta ${BUILD_DIRECTORY}/python/3d-unet-kits19
    cd ${REF_DIR}
    ACC_SCRIPT="${REF_DIR}/accuracy_kits.py"
    ACC_ARGS="--preprocessed_data_dir ${PREPROC_DATA_PATH} --postprocessed_data_dir ${POSTPROC_DATA_PATH} --log_file ${ACC_FILENAME}"
    ;;
"bert")
    REF_DIR=${MLPERF_DIR}/dependencies/mlperf-inference/language/bert
    DATASET=squad
    TOTAL_SAMPLE_COUNT=10833
    PERF_SAMPLE_COUNT=${TOTAL_SAMPLE_COUNT}
    export DATA_PATH=${BUILD_DIRECTORY}/datasets/${DATASET}
    ACC_SCRIPT="${REF_DIR}/accuracy-squad.py"
    ACC_ARGS="--vocab_file ${DATA_PATH}/vocab.txt --val_data ${DATA_PATH}/dev-v1.1.json --features_cache_file ${OUT_DIR}/eval_features.pickle --out_file ${OUT_DIR}/bert_predictions.json --log_file ${ACC_FILENAME}"
    ;;

*) echo "Unknown model name ${MODELNAME}. Skip it"
    exit
    ;;
esac

# Set command line arguments
OV_MLPERF_ARGS="--scenario ${SCENARIO} --mode ${MODE} --device ${DEVICE} --batch_size 1"         # General arguments
OV_MLPERF_ARGS="${OV_MLPERF_ARGS} --mlperf_conf ${MLPERF_CONF} --user_conf ${REF_DIR}/user.conf" # MLPerf configs
OV_MLPERF_ARGS="${OV_MLPERF_ARGS} --model_name ${MODELNAME} --model_path ${MODEL_PATH}"          # Model name and path
OV_MLPERF_ARGS="${OV_MLPERF_ARGS} --dataset ${DATASET} --data_path ${DATA_PATH}"                 # Dataset name and path
OV_MLPERF_ARGS="${OV_MLPERF_ARGS} --total_sample_count ${TOTAL_SAMPLE_COUNT} --perf_sample_count ${PERF_SAMPLE_COUNT}"
OV_MLPERF_ARGS="${OV_MLPERF_ARGS} --log_output_dir ${OUT_DIR}"

if [ ${MODE} == "Performance" ]; then
    if [ ! -z ${WARMUP_ITERS} ]; then
        OV_MLPERF_ARGS="${OV_MLPERF_ARGS} --warmup_iters ${WARMUP_ITERS}"
    fi
else
    OV_MLPERF_ARGS="${OV_MLPERF_ARGS} --warmup_iters 0"
fi

# Update OpenVINO parameters if needed
if [ ${SCENARIO} == "SingleStream" ]; then
    OV_MLPERF_ARGS="${OV_MLPERF_ARGS} --nstreams 1"
fi

if [ ${DEVICE} == "GPU" ]; then
    OV_MLPERF_ARGS="${OV_MLPERF_ARGS} --infer_precision f16"
fi

# Running benchmark
OV_MLPERF_FULL_CMD="${OV_MLPERF_BIN} ${OV_MLPERF_ARGS}"
echo "Running benchmark: ${OV_MLPERF_FULL_CMD}"
${OV_MLPERF_FULL_CMD} 2>&1 | tee ${OUT_DIR}/ov_mlperf_log.txt

echo "Results are dumped into the ${OUT_DIR}"

# Check accuracy if it is needed
if [[ ${MODE} == "Accuracy" ]] && [[ ! -z "${ACC_SCRIPT}" ]]; then
    ACC_FULL_CMD="python -u ${ACC_SCRIPT} ${ACC_ARGS}"
    echo "Running accuracy check: ${ACC_FULL_CMD}"
    ${ACC_FULL_CMD} 2>&1 | tee ${OUT_DIR}/ov_mlperf_log_accuracy.txt
fi
