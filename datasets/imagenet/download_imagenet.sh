#!/bin/bash
SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar -O ${SCRIPT_DIR}/ILSVRC2012_img_val.tar

if [ -d "${SCRIPT_DIR}/ILSVRC2012_img_val" ]; then
    rm -r ${SCRIPT_DIR}/ILSVRC2012_img_val
fi

mkdir ${SCRIPT_DIR}/ILSVRC2012_img_val

tar -xvf ${SCRIPT_DIR}/ILSVRC2012_img_val.tar -C ${SCRIPT_DIR}/ILSVRC2012_img_val

cp ${SCRIPT_DIR}/val_data/*.txt ${SCRIPT_DIR}/ILSVRC2012_img_val/
