#!/bin/bash
# Edit path as appropriate
BUILD_DIRECTORY=$(dirname $(dirname $(dirname $(realpath "$BASH_SOURCE"))))/MLPerf

# Other relative paths used during build
MLPERF_DIR=${BUILD_DIRECTORY}/MLPerf-Intel-openvino
DEPS_DIR=${MLPERF_DIR}/dependencies
OPENVINO_DIR=${DEPS_DIR}/openvino

# Libraries
OPENVINO_LIBRARIES=${OPENVINO_DIR}/bin/intel64/Release
OPENCV_LIBRARIES=${OPENCV_DIRS[0]}/opencv/lib
BOOST_LIBRARIES=${DEPS_DIR}/boost/boost_1_72_0/stage/lib
GFLAGS_LIBRARIES=${DEPS_DIR}/gflags

#Back up LD_LIBRARY_PATH
export OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH

PY_V="$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
export PYTHONPATH=${PYTHONPATH}:${OPENVINO_DIR}/bin/intel64/python_api/python${PY_V}
export PYTHONPATH=${PYTHONPATH}:${OPENVINO_DIR}/tools/mo/
export PYTHONPATH=${PYTHONPATH}:${MLPERF_DIR}/dependencies/mlperf-inference/loadgen/install/lib/python${PY_V}/site-packages/mlperf_loadgen-3.0-py${PY_V}-linux-x86_64.egg
export LD_LIBRARY_PATH=${OPENVINO_LIBRARIES}:${OPENCV_LIBRARIES}:${BOOST_LIBRARIES}:${GFLAGS_LIBRARIES}
export BUILD_DIRECTORY=${BUILD_DIRECTORY}
export MLPERF_DIR=${MLPERF_DIR}
export OPENVINO_DIR=${OPENVINO_DIR}
export MODEL_DIR=${BUILD_DIRECTORY}/models
export DATASET_DIR=${BUILD_DIRECTORY}/datasets
export RESULTS_DIR=${BUILD_DIRECTORY}/results
