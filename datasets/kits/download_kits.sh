#!/bin/bash

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )
KITS19_DATA_DIR=${SCRIPT_DIR}/kits19

git clone https://github.com/neheller/kits19 ${KITS19_DATA_DIR}
cd ${KITS19_DATA_DIR}
pip3 install -r requirements.txt
python3 -m starter_code.get_imaging
RAW_DATA_DIR=${KITS19_DATA_DIR}/data

if [ -s ${RAW_DATA_DIR}/case_00185/imaging.nii.gz ]; then
    echo "Duplicating KITS19 case_00185 as case_00400..."
    cp -Rf ${RAW_DATA_DIR}/case_00185 ${RAW_DATA_DIR}/case_00400
else
    echo "KITS19 case_00185 not found! please download the dataset first..."
fi

PREPROCESSED_DATA_DIR=${SCRIPT_DIR}/preprocessed_data
mkdir ${PREPROCESSED_DATA_DIR}
cd ${MLPERF_DIR}/dependencies/mlperf-inference/vision/medical_imaging/3d-unet-kits19
python preprocess.py --raw_data_dir ${RAW_DATA_DIR} --results_dir ${PREPROCESSED_DATA_DIR} --mode preprocess