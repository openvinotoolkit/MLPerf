#!/bin/bash
MODEL_LIST="$1"
if [ -z ${MODEL_LIST} ];
then
    MODEL_LIST=("resnet50" "retinanet" "3d-unet" "bert")
fi
echo "Downloading models: ${MODEL_LIST[@]}"

echo "Creating model directory ${MODEL_DIR}..."
mkdir "${MODEL_DIR}"

for MODELNAME in "${MODEL_LIST[@]}"
do
    CURRENT_MODEL_DIR="${MODEL_DIR}/${MODELNAME}"
    mkdir "${CURRENT_MODEL_DIR}"
    cd ${CURRENT_MODEL_DIR}
    echo "Downloading model ${MODELNAME}..."

    case "${MODELNAME}" in
    resnet50)
        FILENAME=resnet50_v1.onnx
        wget https://zenodo.org/record/4735647/files/${FILENAME} -O ${FILENAME}
        CMD_ARGS="-m=${FILENAME} --model_name ${MODELNAME} --input_shape [1,3,224,224] --layout nchw \
                  --mean_values [123.68,116.78,103.94]"
        ;;

    retinanet)
        FILENAME=retinanet.onnx
        wget https://zenodo.org/record/6617879/files/resnext50_32x4d_fpn.onnx -O ${FILENAME}
         CMD_ARGS="-m=${FILENAME} --model_name ${MODELNAME} --input_shape [1,3,800,800] --layout nchw \
                --scale 255"
        ;;

    3d-unet)
        FILENAME=3d-unet.onnx
        wget https://zenodo.org/record/5597155/files/3dunet_kits19_128x128x128_dynbatch.onnx?download=1 -O ${FILENAME}
        CMD_ARGS="-m=${FILENAME} --model_name ${MODELNAME}  --input_shape [1,1,128,128,128] --layout ncdhw"
        ;;

    bert)
        FILENAME=bert.onnx
        wget https://zenodo.org/record/3733910/files/model.onnx?download=1 -O ${FILENAME}
        CMD_ARGS="-m=${CURRENT_MODEL_DIR}/${FILENAME}                                            \
                 --model_name ${MODELNAME} --layout=input_ids(n?),input_mask(n?),segment_ids(n?) \
                 --input_shape=[1,384],[1,384],[1,384] --input=input_ids,input_mask,segment_ids"
        ;;

    *) echo "Unknown model name ${MODELNAME}. Skip it"
        ;;
    esac
    CMD="python ${OPENVINO_DIR}/tools/mo/openvino/tools/mo/mo.py ${CMD_ARGS}"
    echo "Run Model Optimizer ${CMD}"
    ${CMD}
done

cd ${MODEL_DIR}