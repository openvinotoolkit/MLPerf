#!/bin/bash
DATASET_LIST="$1"
if [ -z ${DATASET_LIST} ];
then
    DATASET_LIST=("imagenet" "openimages" "kits" "squad")
fi
echo "Downloading datasets: ${DATASET_LIST[@]}"

echo "Creating dataset directory ${DATASET_DIR}..."
mkdir "${DATASET_DIR}"

for DATASET_NAME in "${DATASET_LIST[@]}"
do
    CURRENT_DATASET_DIR="${DATASET_DIR}/${DATASET_NAME}"
    echo "Downloading dataset ${DATASET_NAME} into ${CURRENT_DATASET_DIR}..."

    case "${DATASET_NAME}" in
    imagenet)
        ${CURRENT_DATASET_DIR}/download_imagenet.sh
        ;;

    openimages)
        ${CURRENT_DATASET_DIR}/download_openimages.sh
        ;;

    kits)
        ${CURRENT_DATASET_DIR}/download_kits.sh
        ;;

    squad)
        ${CURRENT_DATASET_DIR}/download_squad.sh
        ;;

    *) echo "Unknown model name ${DATASET_NAME}. Skip it";;
    esac
done

cd ${DATASET_DIR}