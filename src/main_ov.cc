#include <string>
#include <functional>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <regex>

#include "loadgen.h"
#include "query_sample.h"
#include "query_sample_library.h"
#include "test_settings.h"
#include "system_under_test.h"
#include "bindings/c_api.h"
#include "datasets/imagenet.h"
#include "datasets/squad.h"
#include "datasets/openimages.h"
#include "suts/sut_base.h"
#include "suts/sut_multistream.h"
#include "suts/sut_offline.h"
#include "suts/sut_server.h"

#include "input_flags.h"
#include "workload_helpers.h"
#include "postprocess/post_processors.h"

#define NANO_SEC 1e9
#define MILLI_SEC 1000
#define MILLI_TO_NANO 1000000

int main(int argc, char **argv) {
    std::unique_ptr<QSLBase> ov_qsl;
    std::unique_ptr<SUTBase> ov_sut;
    mlperf::TestSettings settings;
    mlperf::LogSettings log_settings;

    PPFunction post_processor;

    std::unique_ptr<mlperf_ov::WorkloadBase> workload;

    // Parse Command flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    settings.mode = mlperf::TestMode::PerformanceOnly;
    settings.scenario = mlperf::TestScenario::SingleStream;
    size_t image_width = 224, image_height = 224, image_depth = 1, num_channels = 3;
    int max_seq_length = 384, max_query_length = 64, doc_stride = 128;
    std::string image_format = "NCHW", multi_device_streams = "";

    auto load_config = [&](const std::string& config) {
        if (!config.empty()) {
            settings.FromConfig(config, FLAGS_model_name, FLAGS_scenario);
        }
    };
    load_config(FLAGS_mlperf_conf);
    load_config(FLAGS_user_conf);
    load_config(FLAGS_audit_conf);

    const std::map<std::string, mlperf::TestMode> mode_map = {
        { "Accuracy",            mlperf::TestMode::AccuracyOnly},
        { "Performance",         mlperf::TestMode::PerformanceOnly},
        { "Submission",          mlperf::TestMode::SubmissionRun},
        { "FindPeakPerformance", mlperf::TestMode::FindPeakPerformance},
        { "FindPeakPerformance", mlperf::TestMode::FindPeakPerformance},
    };
    if (!FLAGS_mode.empty() && mode_map.count(FLAGS_mode)) {
        settings.mode = mode_map.at(FLAGS_mode);
    } else {
        throw std::runtime_error("Mode " + FLAGS_mode + " is not supported!");
    }

    const std::map<std::string, mlperf::TestScenario> scenario_map = {
        { "SingleStream", mlperf::TestScenario::SingleStream},
        { "MultiStream",  mlperf::TestScenario::MultiStream},
        { "Server",       mlperf::TestScenario::Server},
        { "Offline",      mlperf::TestScenario::Offline},
    };

    if (!FLAGS_scenario.empty() && scenario_map.count(FLAGS_scenario)) {
        settings.scenario = scenario_map.at(FLAGS_scenario);
    } else {
        throw std::runtime_error("Scenario " + FLAGS_scenario + " is not supported!");
    }

    log_settings.enable_trace = false;
    if (!FLAGS_log_output_dir.empty()) {
        log_settings.log_output.outdir = FLAGS_log_output_dir;
        std::cout << "    [INFO] Output logs directory: " << log_settings.log_output.outdir << std::endl;
    }

    if (settings.performance_sample_count_override > 0 ){
	    FLAGS_perf_sample_count = settings.performance_sample_count_override;
    }

    if (FLAGS_model_name.compare("resnet50") == 0) {
	    post_processor = std::bind(&Processors::postprocess_resnet50, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);
        workload = std::unique_ptr<mlperf_ov::ResNet50>(new mlperf_ov::ResNet50());
        image_format = "NCHW";
        image_height = 224;
        image_width = 224;
        num_channels = 3;
    } else if (FLAGS_model_name.compare("bert") == 0) {
        post_processor = std::bind(&Processors::postprocess_bert, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);
	    workload = std::unique_ptr<mlperf_ov::Bert>(new mlperf_ov::Bert());
        max_seq_length = 384;
        max_query_length = 64;
        doc_stride = 128;
    } else if (FLAGS_model_name.compare("retinanet") == 0) {
        post_processor = std::bind(&Processors::postprocess_ssd_retinanet, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);
	    workload = std::unique_ptr<mlperf_ov::RetinaNet>(new mlperf_ov::RetinaNet());
        image_format = "NCHW";
        image_height = 800;
        image_width = 800;
        num_channels = 3;
    } else {
        throw std::runtime_error("Model is not supported: " + FLAGS_model_name);
    }

	// Set Workload Attributes
	std::vector<std::string> in_blobs, out_blobs;
	auto dataset_name = workload->get_dataset_name();
    auto workload_name = workload->get_workload_name();
	in_blobs = workload->get_input_names();
	out_blobs = workload->get_output_names();

    if (dataset_name == mlperf_ov::DatasetName::ImageNet2012) {
        ov_qsl = std::unique_ptr<Imagenet>(new Imagenet(settings, image_width, image_height,
                        num_channels, FLAGS_data_path, image_format,
                        FLAGS_total_sample_count, FLAGS_perf_sample_count, workload_name,
                        dataset_name));
    } else if (dataset_name == mlperf_ov::DatasetName::SQuAD_v1_1) {
        ov_qsl = std::unique_ptr<Squad>(new Squad(settings, max_seq_length, max_query_length,
                        doc_stride, FLAGS_data_path, FLAGS_total_sample_count,
                        FLAGS_perf_sample_count, workload_name, dataset_name));
    } else if (dataset_name == mlperf_ov::DatasetName::OpenImages_v6) {
        ov_qsl = std::unique_ptr<OpenImages>(new OpenImages(settings, image_width, image_height,
                        num_channels, FLAGS_data_path, image_format,
                        FLAGS_total_sample_count, FLAGS_perf_sample_count, workload_name,
                        dataset_name));
    }
    if (ov_qsl->TotalSampleCount() == 0) {
        throw std::runtime_error("There is no samples in data path " + FLAGS_data_path);
    } else {
        std::cout << "    [INFO] Total Sample Count: " << ov_qsl->TotalSampleCount() << std::endl;
    }

    if (settings.mode != mlperf::TestMode::AccuracyOnly) {
        std::cout << "    [INFO] Performance Sample Count: " << ov_qsl->PerformanceSampleCount() << std::endl;
    }

    std::string trail_space(45, ' ');
    OVBackendProperties ov_properties;
    ov_properties.device = FLAGS_device;
    ov_properties.nstreams = FLAGS_nstreams;
    ov_properties.nthreads = FLAGS_nthreads;
    ov_properties.infer_precision = FLAGS_infer_precision;
    ov_properties.allow_auto_batching = FLAGS_allow_auto_batching;
    ov_properties.extensions = FLAGS_extensions;

    // Init SUT
    if (settings.scenario == mlperf::TestScenario::SingleStream) {
        ov_sut = std::unique_ptr<SUTBase>(new SUTBase(settings, ov_qsl.get(), ov_properties, FLAGS_batch_size,
                    FLAGS_dataset, FLAGS_model_name, in_blobs, out_blobs, FLAGS_model_path, post_processor));
    } else if (settings.scenario == mlperf::TestScenario::Offline) {
        ov_sut = std::unique_ptr<SUTOffline>(new SUTOffline(settings, ov_qsl.get(), ov_properties, FLAGS_batch_size,
                    FLAGS_dataset, FLAGS_model_name, in_blobs, out_blobs, FLAGS_model_path, post_processor));
    } else if (settings.scenario == mlperf::TestScenario::MultiStream) {
        ov_sut = std::unique_ptr<SUTMultistream>(new SUTMultistream(settings, ov_qsl.get(), ov_properties, FLAGS_batch_size,
                    FLAGS_dataset, FLAGS_model_name, in_blobs, out_blobs, FLAGS_model_path, post_processor));
     } else if (settings.scenario == mlperf::TestScenario::Server) {
        ov_sut = std::unique_ptr<SUTServer>(new SUTServer(settings, ov_qsl.get(), ov_properties, FLAGS_batch_size,
                    FLAGS_dataset, FLAGS_model_name, in_blobs, out_blobs, FLAGS_model_path, post_processor));
    }
    if (FLAGS_warmup_iters > 0) {
        std::cout << "    [INFO] Warming up \n";
        ov_sut->WarmUp(FLAGS_warmup_iters);
    }

    std::cout << "    [INFO] Starting " << FLAGS_mode << "Benchmark\n";
    mlperf::StartTest(reinterpret_cast<mlperf::SystemUnderTest*>(ov_sut.get()),
                      reinterpret_cast<mlperf::QuerySampleLibrary*>(ov_qsl.get()),
                      settings, log_settings, FLAGS_audit_conf);
    std::cout << "    [INFO] Benchmark Completed" << trail_space << "\n";

    return 0;
}


