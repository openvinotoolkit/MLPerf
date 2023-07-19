#!/bin/bash

error() {
    local code="${3:-1}"
    if [[ -n "$2" ]];then
        echo "Error on or near line $1: $2; exiting with status ${code}"
    else
        echo "Error on or near line $1; exiting with status ${code}"
    fi
    exit "${code}"
}
trap 'error ${LINENO}' ERR

sudo apt update
sudo apt-get install libglib2.0-dev libtbb-dev python3-dev python3-pip cmake

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
BUILD_DIRECTORY=${CUR_DIR}
SKIPS=" "
DASHES="================================================"


MLPERF_DIR=${BUILD_DIRECTORY}/MLPerf-Intel-openvino
if [ -e ${MLPERF_DIR} ]; then
	rm -rf ${MLPERF_DIR}
fi

DEPS_DIR=${MLPERF_DIR}/dependencies

#====================================================================
# Build OpenVINO library (If not using publicly available openvino)
#====================================================================

echo " ========== Building OpenVINO libraries ==========="
echo ${SKIPS}

OPENVINO_DIR=${DEPS_DIR}/openvino
git clone -b releases/2023/0 https://github.com/openvinotoolkit/openvino.git ${OPENVINO_DIR}

cd ${OPENVINO_DIR}

git submodule update --init --recursive
sudo -E ./install_build_dependencies.sh
pip install -r tools/mo/requirements.txt
pip install --force-reinstall numpy==1.21.6
mkdir build && cd build

cmake -DTHREADING=TBB                   \
    -DENABLE_OPENCV=ON                  \
    -DENABLE_INTEL_CPU=ON               \
    -DENABLE_INTEL_GPU=ON               \
    -DENABLE_INTEL_GNA=OFF              \
    -DENABLE_TESTS=OFF                  \
    -DENABLE_SYSTEM_OPENCL=OFF          \
    -DENABLE_AUTO=OFF                   \
    -DENABLE_MULTI=OFF                  \
    -DENABLE_HETERO=OFF                 \
    -DENABLE_OV_IR_FRONTEND=ON          \
    -DENABLE_OV_ONNX_FRONTEND=ON        \
    -DENABLE_OV_PADDLE_FRONTEND=OFF     \
    -DENABLE_OV_PYTORCH_FRONTEND=OFF    \
    -DENABLE_OV_TF_FRONTEND=ON          \
    -DENABLE_OV_TF_LITE_FRONTEND=OFF    \
    -DENABLE_PYTHON=ON                  \
    -DPYTHON_EXECUTABLE=`which python3` \
	..

TEMPCV_DIR=${OPENVINO_DIR}/temp/opencv_4*
OPENCV_DIRS=$(ls -d -1 ${TEMPCV_DIR} )
OPENCV_LIBRARIES=${OPENCV_DIRS[0]}/opencv/lib
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${OPENCV_LIBRARIES}

make -j$(nproc)

#=============================================================
#       Build gflags
#=============================================================
echo ${SKIPS}
echo " ============ Building Gflags ==========="
echo ${SKIPS}

GFLAGS_DIR=${DEPS_DIR}/gflags

git clone https://github.com/gflags/gflags.git ${GFLAGS_DIR}
cd ${GFLAGS_DIR}
mkdir gflags-build && cd gflags-build
cmake .. && make


# Build boost
echo ${SKIPS}
echo "========= Building boost =========="
echo ${SKIPS}

BOOST_DIR=${DEPS_DIR}/boost
if [ ! -d ${BOOST_DIR} ]; then
        mkdir ${BOOST_DIR}
fi

cd ${BOOST_DIR}
wget https://sourceforge.net/projects/boost/files/boost/1.72.0/boost_1_72_0.tar.gz
tar -xzf boost_1_72_0.tar.gz
cd boost_1_72_0
./bootstrap.sh --with-libraries=filesystem
./b2 --with-filesystem

#===============================================================
# Build loadgen
#===============================================================
echo ${SKIPS}
echo " =========== Building mlperf loadgenerator =========="
echo ${SKIPS}

MLPERF_INFERENCE_REPO=${DEPS_DIR}/mlperf-inference
if [ -d ${MLPERF_INFERENCE_REPO} ]; then
        rm -r ${MLPERF_INFERENCE_REPO}
fi

python3 -m pip install absl-py numpy pybind11

git clone -b v3.0 --recurse-submodules https://github.com/mlcommons/inference.git ${MLPERF_INFERENCE_REPO}

cd ${MLPERF_INFERENCE_REPO}/loadgen
git submodule update --init --recursive

CFLAGS="-std=c++14 -O3" python3 setup.py install --prefix=`pwd`/install

cd build
cmake -DPYTHON_EXECUTABLE=`which python3` \
	..

make

cp libmlperf_loadgen.a ../

cd ${MLPERF_DIR}

# =============================================================
#        Build ov_mlperf
#==============================================================

echo ${SKIPS}
echo " ========== Building ov_mlperf ==========="
echo ${SKIPS}

SOURCE_DIR=${CUR_DIR}/src
cd ${SOURCE_DIR}

if [ -d build ]; then
	rm -r build
fi

mkdir build && cd build

BOOST_LIBRARIES=${BOOST_DIR}/boost_1_72_0/stage/lib
cmake -DInferenceEngine_DIR=${OPENVINO_DIR}/build/ \
		-DOpenCV_DIR=${OPENCV_DIRS[0]}/opencv/cmake/ \
		-DLOADGEN_DIR=${MLPERF_INFERENCE_REPO}/loadgen \
		-DBOOST_INCLUDE_DIRS=${BOOST_DIR}/boost_1_72_0 \
		-DBOOST_FILESYSTEM_LIB=${BOOST_LIBRARIES}/libboost_filesystem.so \
		-DCMAKE_BUILD_TYPE=Release \
		-Dgflags_DIR=${GFLAGS_DIR}/gflags-build/ \
		..

make

echo ${SKIPS}
echo ${DASHES}
if [ -e ${SOURCE_DIR}/Release/ov_mlperf ]; then
        echo -e "\e[1;32m ov_mlperf built at ${SOURCE_DIR}/Release/ov_mlperf \e[0m"
else
        echo -e "\e[0;31m ov_mlperf not built. Please check logs on screen\e[0m"
fi
echo ${DASHES}
