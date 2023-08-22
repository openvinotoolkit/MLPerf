#! /usr/bin/env python3
# coding=utf-8

from pathlib import Path
import os

from base_SUT import BASE_3DUNET_SUT
from utils import parse_devices, parse_value_per_device
from openvino.runtime import Core, properties


class _3DUNET_OpenVINO_SUT(BASE_3DUNET_SUT):
    """
    A class to represent SUT (System Under Test) for MLPerf.
    This inherits BASE_3DUNET_SUT and builds functionality for ONNX Runtime.

    Attributes
    ----------
    sut: object
        SUT in the context of LoadGen
    qsl: object
        QSL in the context of LoadGen
    model_path: str or PosixPath object
        path to the model for ONNX Runtime
    sess: object
        ONNX runtime session instance that does inference

    Methods
    -------
    do_infer(input_tensor):
        Perform inference upon input_tensor with ONNX Runtime
    """

    def __init__(self, args):
        """
        Constructs all the necessary attributes for ONNX Runtime specific 3D UNet SUT

        Parameters
        ----------
            model_path: str or PosixPath object
                path to the model for ONNX Runtime
            preprocessed_data_dir: str or PosixPath
                path to directory containing preprocessed data
            perf_sample_count: int
                number of query samples guaranteed to fit in memory
        """
        preprocessed_data_dir = os.path.dirname(args.data_path)
        super().__init__(preprocessed_data_dir, args.perf_sample_count)
        print("Loading IR...")
        assert Path(args.model_path).is_file(
        ), "Cannot find the model file {:}!".format(args.model_path)

        devices = parse_devices(args.device)
        # device_number_streams = parse_value_per_device(
        #    devices, args.nstreams, "nstreams")
        device_infer_precision = parse_value_per_device(
            devices, args.infer_precision, "infer_precision")

        self.core = Core()
        device_config = {}
        for device in devices:
            # TODO implement async inference
            device_config[properties.hint.performance_mode(
            )] = properties.hint.PerformanceMode.LATENCY
            if device == "CPU":
                if args.nthreads:
                    device_config[properties.inference_num_threads()
                                  ] = args.nthreads
            # Set inference precision of specified
            if device in device_infer_precision.keys():
                device_config[properties.hint.inference_precision(
                )] = device_infer_precision[device]

        print("Input model: ", args.model_path)
        self.compiled_model = self.core.compile_model(
            args.model_path, args.device, device_config)
        self.output_name = next(iter(self.compiled_model.outputs[0].names))
        keys = self.compiled_model.get_property(
            properties.supported_properties())
        print("Model:")
        for k in keys:
            if k != properties.supported_properties():
                value = self.compiled_model.get_property(k)
                print(f'  {k}: {value}')
        self.infer_request = self.compiled_model.create_infer_request()

    def do_infer(self, input_tensor):
        """
        Perform inference upon input_tensor with OpenVINO
        """
        results = self.infer_request.infer(input_tensor)
        return next(iter(results.values()))


def get_sut(args):
    """
    Redirect the call for instantiating SUT to OpenVINO specific SUT
    """
    return _3DUNET_OpenVINO_SUT(args)
