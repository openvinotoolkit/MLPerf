#! /usr/bin/env python3
# coding=utf-8

import mlperf_loadgen as lg
from pathlib import Path
import subprocess
import argparse
import os
import sys
sys.path.insert(0, os.getcwd())


__doc__ = """
Run 3D UNet performing KiTS19 Kidney Tumore Segmentation task.
Dataset needs to be prepared through preprocessing (python preprocess.py --help).

Run inference in performance mode:
    python3 run.py --backend=$(BACKEND) --scenario=$(SCENARIO) --model=$(MODEL)

Run inference in accuracy mode:
    python3 run.py --backend=$(BACKEND) --scenario=$(SCENARIO) --model=$(MODEL) --accuracy

$(BACKEND): tensorflow, pytorch, or onnxruntime
$(SCENARIO): Offline, SingleStream, MultiStream, or Server (Note: MultiStream may be deprecated)
$(MODEL) should point to correct model for the chosen backend

If run for the accuracy, DICE scores will be summarized and printed at the end of the test, and
inference results will be stored as NIFTI files.

Performance run can be more specific as:
    python3  run.py --backend=$(BACKEND) --scenario=$(SCENARIO) --model=$(MODEL)
                    --preprocessed_data_dir=$(PREPROCESSED_DATA_DIR)
                    --postprocessed_data_dir=$(POSTPROCESSED_DATA_DIR)
                    --mlperf_conf=$(MLPERF_CONF)
                    --user_conf=$(USER_CONF)
                    --perf_sample_count=$(PERF_CNT)

$(MLPERF_CONF) contains various configurations MLPerf-Inference needs and used to configure LoadGen
$(USER_CONF) contains configurations such as target QPS for LoadGen and overrides part of $(MLPERF_CONF)
$(PERF_CNT) sets number of query samples guaranteed to fit in memory

More info for the above LoadGen related configs can be found at:
https://github.com/mlcommons/inference/tree/master/loadgen
"""


def get_args():
    """
    Args used for running 3D UNet KITS19
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--scenario",
                        choices=["SingleStream", "Offline"],
                        default="Offline",
                        help="Scenario")
    parser.add_argument("--mode",
                        choices=["Performance", "Accuracy"],
                        default="Performance",
                        help="execution mode")
    parser.add_argument("--user_conf",
                        default="user.conf",
                        help="user config for user LoadGen settings such as target QPS")
    parser.add_argument("--mlperf_conf",
                        default="build/mlperf.conf",
                        help="mlperf rules config")
    parser.add_argument("--data_path",
                        default="datasets/kits/preprocessed_data/preprocessed_files.pkl",
                        help="Path to preprocessed dataset in pickle format")
    parser.add_argument("--dataset",
                        choices=["kits"], default="kits",
                        help="Dataset name")
    parser.add_argument("--model_name",
                        choices=["3d-unet"],
                        default="3d-unet",
                        help="Model name")
    parser.add_argument("--model_path",
                        default="models/3d-unet/3d-unet.xml",
                        help="Path to OpenVINO IR")
    parser.add_argument("--total_sample_count",
                        type=int, default=42,
                        help="Total Number of samples available for benchmark")
    parser.add_argument("--perf_sample_count",
                        type=int, default=42,
                        help="Number of samples to load for benchmarking")
    parser.add_argument('--nstreams', type=str, required=False, default=None,
                        help='Optional. Number of streams to use for inference on the CPU/GPU '
                        '(for HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> '
                        'or just <nstreams>). '
                        'Default value is determined automatically for a device. Please note that although the automatic selection '
                        'usually provides a reasonable performance, it still may be non - optimal for some cases, especially for very small models. '
                        'Also, using nstreams>1 is inherently throughput-oriented option, while for the best-latency '
                        'estimations the number of streams should be set to 1. '
                        'See samples README for more details.')
    parser.add_argument("--nthreads",
                        type=int, default=None,
                        help="Number of threads")
    parser.add_argument("--batch_size",
                        type=int, default=1,
                        help="Batch size")
    parser.add_argument("--device",
                        choices=["CPU", "GPU"],
                        default="CPU",
                        help="device")
    parser.add_argument('--infer_precision', type=str, default=None, required=False,
                        help='Optional. Specifies the inference precision. Example #1: \'-infer_precision bf16\'. Example #2: \'-infer_precision CPU:bf16,GPU:f32\'')
    parser.add_argument("--warmup_iters",
                        type=str, default=0,
                        help="Number of warmup interations")
    parser.add_argument("--log_output_dir",
                        type=str, default="",
                        help="Optional. Path to the directory for MLPerf output logs.")
    parser.add_argument("--allow_auto_batching",
                        type=bool, default=False,
                        help="Optional. Allow auto batching..")
    args = parser.parse_args()
    return args


def main():
    """
    Runs 3D UNet performing KiTS19 Kidney Tumore Segmentation task as below:

    1. instantiate SUT and QSL for the chosen backend
    2. configure LoadGen for the chosen scenario
    3. configure MLPerf logger
    4. start LoadGen
    5. collect logs and if needed evaluate inference results
    6. clean up
    """
    # scenarios in LoadGen
    scenario_map = {
        "SingleStream": lg.TestScenario.SingleStream,
        "Offline": lg.TestScenario.Offline,
        "Server": lg.TestScenario.Server,
        "MultiStream": lg.TestScenario.MultiStream
    }

    args = get_args()

    # instantiate SUT as per requested backend; QSL is also instantiated
    from openvino_sut import get_sut
    sut = get_sut(args)

    # setup LoadGen
    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]
    settings.FromConfig(args.mlperf_conf, "3d-unet", args.scenario)
    settings.FromConfig(args.user_conf, "3d-unet", args.scenario)
    if args.mode == "Accuracy":
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    # set up mlperf logger
    log_output_settings = lg.LogOutputSettings()
    if args.log_output_dir != "":
        log_path = Path(args.log_output_dir).absolute()
        log_output_settings.outdir = str(log_path)
        print("Output logs directory:", log_output_settings.outdir)
    log_settings = lg.LogSettings()
    log_settings.enable_trace = False
    log_settings.log_output = log_output_settings

    # start running test, from LoadGen
    print("Starting", args.mode, "Benchmark")
    lg.StartTestWithLogSettings(sut.sut, sut.qsl.qsl, settings, log_settings)
    print("Benchmark Completed")

    # cleanup
    lg.DestroySUT(sut.sut)
    lg.DestroyQSL(sut.qsl.qsl)


if __name__ == "__main__":
    main()
