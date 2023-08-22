#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <boost/filesystem.hpp>

/** ========================== MLPerf flags ==========================**/

// Scenario flag
static const char scenario_message[] = "MLPerf scenario: one of SingleStream, Offline, Server, MultiStream";

static bool validate_scenario(const char* scenario, const std::string& value){
    std::list<std::string> scenarios = {"SingleStream", "Offline", "Server", "MultiStream"};

    for (std::list<std::string>::iterator it = scenarios.begin(); it != scenarios.end(); ++it){
        if (value.compare(*it) == 0) return true;
    }
    std::cout << value << " is not a valid/supported scenario. " << scenario_message << std::endl;
    return false;
}

DEFINE_string(scenario, "SingleStream", scenario_message);
DEFINE_validator(scenario, &validate_scenario);


// Mode flag
static const char mode_message[] = "MLPerf mode: Performance, Accuracy";

static bool validate_mode(const char* mode, const std::string& value){
    std::list<std::string> modes = {"Performance", "Accuracy"};

    for (std::list<std::string>::iterator it = modes.begin(); it != modes.end(); ++it){
        if (value.compare(*it) == 0) return true;
    }
    std::cout << value << " is not a valid/supported mode. " << mode_message << std::endl;
    return false;
}

DEFINE_string(mode, "Performance", mode_message);
DEFINE_validator(mode, &validate_mode);

// Config file
static const char mlperf_conf_message[] = "MLPerf default config file (mlperf.conf)";

static bool validate_mlperf_config(const char* mlperf_conf, const std::string& mlperf_conf_value){

    if (mlperf_conf_value.length() == 0){
        //std::cout << " --mlperf_conf not provided." << std::endl;
        return false;
    }

    boost::filesystem::path p(mlperf_conf_value);
    if (!boost::filesystem::exists( p ) ){
        std::cout << " [ERROR]: Provided MLPerf config '" << mlperf_conf_value << "' could not be found. Please provide </path/to/mlperf.conf>" << std::endl;
        return false;
    }

    return true;
}

DEFINE_string(mlperf_conf, "", mlperf_conf_message);
DEFINE_validator(mlperf_conf, &validate_mlperf_config);

// User config file
static const char user_conf_message[] = "SUT User config file";

static bool validate_user_config(const char* user_conf, const std::string& user_conf_value){

    if (user_conf_value.length() == 0){
        std::cout << "[WARNING]: --user_conf not provided. Will use use mlperf default config (if --mlperf_conf provided)" << std::endl;
        return true;
    }

    boost::filesystem::path p( user_conf_value );
    if (!boost::filesystem::exists( p ) ){
        std::cout << "[ERROR]: Provided user config '" << user_conf_value << "' could not be found. Please provide valid file or omit (if --mlperf_conf provided)." << std::endl;
        return false;
    }

    return true;
}

DEFINE_string(user_conf, "", user_conf_message);
DEFINE_validator(user_conf, &validate_user_config);

// Audit config file
static const char audit_conf_message[] = "StartTest Audit config file";

static bool validate_audit_config(const char* audit_conf, const std::string& audit_conf_value){

    if (audit_conf_value.length() == 0){
        std::cout << "[WARNING]: --audit_conf not provided." << std::endl;
        return true;
    }

    boost::filesystem::path p( audit_conf_value );
    if (!boost::filesystem::exists( p ) ){
        std::cout << "[ERROR]: Provided audit config '" << audit_conf_value << "' could not be found. Please provide valid file or omit." << std::endl;
        return false;
    }

    return true;
}

DEFINE_string(audit_conf, "", audit_conf_message);
DEFINE_validator(audit_conf, &validate_audit_config);

// Total-Sample-Count
static const char total_sample_count_message[] = "Total Number of samples available for benchmark";
DEFINE_uint32(total_sample_count, 5000, total_sample_count_message);

// Performance-Sample-Count
static const char perf_sample_count_message[] = "Number of samples to load for benchmarking";
DEFINE_uint32(perf_sample_count, 256, perf_sample_count_message);


/**========================== Workload flags ==========================**/

// Dataset type
static const char dataset_message[] = "Name of the dataset (imagenet, openimages, squad); Necessary for right data reading";

static bool validate_dataset(const char* dataset, const std::string& dataset_value){

    if (dataset_value.length()==0) return false;

    std::list<std::string> datasets = {"imagenet", "openimages", "squad"};
    for (std::list<std::string>::iterator it = datasets.begin(); it != datasets.end(); ++it){
	    if (dataset_value.compare(*it)==0) return true;
    }

    std::cout << "Unknown dataset '"<< dataset_value << "'. Allowed values: imagenet, openimages, squad" << std::endl;
    return false;
}

DEFINE_string(dataset, "", dataset_message);
DEFINE_validator(dataset, &validate_dataset);

// Dataset directory
static const char data_path_message[] = "Path to workload dataset";

static bool validate_data_path(const char* data_path, const std::string& data_path_value){

    if (data_path_value.length()==0) return false;

    boost::filesystem::path p( data_path_value );
    if (!(boost::filesystem::exists( p ) ) ) {
        std::cout <<"Dataset path <" << data_path_value << "> not found." << std::endl;
        return false;
    }

    return true;
}

DEFINE_string(data_path, "", data_path_message);
DEFINE_validator(data_path, &validate_data_path);

// Model (workload) name
static const char model_name_message[] = "Supported workloads:\n\tresnet50, \n\tretinanet, \n\tbert";
static bool validate_model_name( const char*, const std::string& model_name_value){

    if (model_name_value.length()==0) return false;

    std::list<std::string> model_names = {"resnet50", "retinanet", "bert"};

    for (std::list< std::string >::iterator it = model_names.begin(); it != model_names.end(); ++it){
	if ( model_name_value.compare(*it) == 0 ) return true;
    }
    std::cout << "Unknown model name '" << model_name_value << "'. " << model_name_message << std::endl;

    return false;
}
DEFINE_string(model_name, "", model_name_message);
DEFINE_validator(model_name, &validate_model_name);


static const char model_path_message[] = "Path to model xml file";
static bool validate_model_path(const char* model_path, const std::string& model_path_value){

    if (model_path_value.length()==0){
        return false;
    }

    boost::filesystem::path p(model_path_value);
    if ( !(boost::filesystem::exists( p )) ){
        std::cout << " Model path <" << model_path_value << "> Not FOUND." << std::endl;
	return false;
    }
    return true;
}

DEFINE_string(model_path, "", model_path_message);
DEFINE_validator(model_path, &validate_model_path);

/**========================== OpenVINO Flags ==========================**/

static const char target_device_message[] = "Optional. Specify a target device to infer on (the list of available devices is shown below). " \
"Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. " \
"Use \"-d MULTI:<comma-separated_devices_list>\" format to specify MULTI plugin. " \
"The application looks for a suitable plugin for the specified device.";

DEFINE_string(device, "CPU", target_device_message);

static const char custom_extensions_library_message[] =
    "Required for custom layers (extensions). Absolute path to a shared library with the kernels "
    "implementations.";
static bool validate_extensions(const char* extensions, const std::string& extensions_value){
    if (extensions_value.length()==0){
        return true;
    }

    boost::filesystem::path p(extensions_value);
    if ( !(boost::filesystem::exists( p )) ){
        return false;
    }

    return true;
}
DEFINE_string(extensions, "", custom_extensions_library_message);
DEFINE_validator(extensions, &validate_extensions);

static const char infer_requests_count_message[] = "Optional. Number of infer requests. Default value is determined automatically for device.";
DEFINE_uint32(nireq, 1, infer_requests_count_message);

static const char infer_num_threads_message[] = "Optional. Number of threads to use for inference on the CPU "
                                                "(including HETERO and MULTI cases).";
DEFINE_uint32(nthreads, 0, infer_num_threads_message);

static const char infer_num_streams_message[] = "Optional. Number of streams to use for inference on the CPU or/and GPU in throughput mode "
                                                "(for HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>)";
DEFINE_string(nstreams, "", infer_num_streams_message);

static const char batch_size_message[] = "Optional. Batch size value. If not specified, the batch size value is determined from Intermediate Representation.";
DEFINE_uint32(batch_size, 1, batch_size_message);

static const char warmup_message[] = "Number of warmup iterations. Defaults to 10.";
DEFINE_uint32(warmup_iters, 10, warmup_message);

static const char inference_precision_message[] =
    "Optional. Specifies the inference precision. Example #1: '-infer_precision bf16'. Example #2: '-infer_precision "
    "CPU:bf16,GPU:f32'";
DEFINE_string(infer_precision, "", inference_precision_message);

static const char log_output_dir_message[] =
    "Optional. Path to the directory for MLPerf output logs.";
DEFINE_string(log_output_dir, "", log_output_dir_message);

static const char allow_auto_batching_message[] =
    "Optional. Allow auto batching.";
DEFINE_bool(allow_auto_batching, false, allow_auto_batching_message);