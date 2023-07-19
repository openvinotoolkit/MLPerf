#ifndef BACKENDOV_H__
#define BACKENDOV_H__

#include <openvino/openvino.hpp>
#include <vector>

#include "infer_request_wrap.h"
#include "item_ov.h"
#include "utils.h"

struct OVBackendProperties {
    std::string device = "CPU";
    std::string nstreams = "";
    std::string infer_precision = "";
    int nthreads = 0;
    uint32_t nireq = 1;
    bool allow_auto_batching = false;
    std::string extensions = "";
};

class OVBackendBase {
 public:
  OVBackendBase(mlperf::TestSettings settings,
                OVBackendProperties ov_properties,
                unsigned batch_size,
                std::string workload,
                std::vector<std::string> input_blob_names,
                std::vector<std::string> output_blob_names,
                std::string input_model,
                PPFunction post_processor)
      : settings_(settings),
        batch_size_(batch_size),
        ov_properties_(ov_properties),
        workload_(workload),
        input_blob_names_(input_blob_names),
        output_blob_names_(output_blob_names),
        input_model_(input_model),
        post_processor_(post_processor),
        inferRequestsQueue_(),
        inferRequestsQueueServer_(),
        object_size_() {}

    ~OVBackendBase() {}

    std::string version() { return ""; }

    std::string name() { return "openvino"; }

    std::string image_format() { return "NCHW"; }

    void set_infer_request() {
        inferRequest_ = compiled_model_.create_infer_request();
    }

    void create_requests() {
        inferRequestsQueue_ = new InferRequestsQueue(
            compiled_model_, ov_properties_.nireq, input_blob_names_, output_blob_names_,
            settings_, workload_, batch_size_, post_processor_);
    }

    void create_server_requests() {
        inferRequestsQueueServer_ = new InferRequestsQueueServer(
            compiled_model_, ov_properties_.nireq, input_blob_names_, output_blob_names_,
            settings_, workload_, batch_size_, post_processor_);
    }

    void warmup(Item input) {
        for (size_t j = 0; j < input_blob_names_.size(); j++) {
            inferRequest_.set_tensor(input_blob_names_[j], input.tensors_[j]);
        }
        inferRequest_.infer();
        std::vector<float> results;
        std::vector<mlperf::ResponseId> response_ids;
        std::vector<unsigned> counts;
        post_processor_(input, inferRequest_, results, response_ids, 1, counts);
    }

    void reset() {
        if (settings_.scenario == mlperf::TestScenario::Offline ||
            settings_.scenario == mlperf::TestScenario::MultiStream) {
            inferRequestsQueue_->reset();
        } else if (settings_.scenario == mlperf::TestScenario::Server) {
            inferRequestsQueueServer_->wait_all();
            inferRequestsQueueServer_->reset();
            inferRequestsQueueServer_->set_warmup(false);
        }
    }

    void set_server_warmup(bool warmup) {
        inferRequestsQueueServer_->set_warmup(warmup);
    }

    std::vector<mlperf::QuerySampleResponse> get_query_sample_responses() {
        return inferRequestsQueue_->get_query_sample_responses();
    }

    // Progress tracker
    void progress(size_t iter, size_t num_batches, size_t progress_bar_length) {
        float p_bar_frac = (float)iter / (float)num_batches;
        int p_length = static_cast<int>(progress_bar_length * p_bar_frac);
        std::string progress_str(p_length, '.');
        std::string remainder_str(progress_bar_length - p_length, ' ');
        progress_str = "    [BENCHMARK PROGRESS] [" + progress_str + remainder_str +
                        "] " + std::to_string((int)(p_bar_frac * 100)) + "%";
        std::cout << progress_str << "\r" << std::flush;
    }

    void predict(Item input, std::vector<float>& results,
                std::vector<mlperf::ResponseId>& response_ids,
                std::vector<unsigned>& counts) {
        for (size_t j = 0; j < input_blob_names_.size(); j++) {
            inferRequest_.set_tensor(input_blob_names_[j], input.tensors_[j]);
        }

        inferRequest_.infer();
        post_processor_(input, inferRequest_, results, response_ids, 1, counts);
    }

    void predict_async(std::vector<Item> input_items) {
        for (size_t j = 0; j < input_items.size(); ++j) {
            auto inferRequest = inferRequestsQueue_->get_idle_request();
            inferRequest->set_inputs(input_items[j]);
            inferRequest->start_async();
        }
        inferRequestsQueue_->wait_all();
    }

    void print_input_outputs_info(const std::shared_ptr<ov::Model>& model) {
        std::cout << "    [INFO] Model inputs:" << std::endl;
        for (auto&& param : model->get_parameters()) {
            std::cout << "         " << param->get_friendly_name() << " : " << param->get_element_type() << " / "
                    << param->get_layout().to_string() << std::endl;
        }
        std::cout << "    [INFO] Model outputs:" << std::endl;
        for (auto&& result : model->get_results()) {
            std::cout << "         " << result->get_friendly_name() << " : " << result->get_element_type() << " / "
                    << result->get_layout().to_string() << std::endl;
        }
    }

    void predict_async_server(Item input_item) {
        auto inferRequest = inferRequestsQueueServer_->get_idle_request();

        inferRequest->set_inputs(input_item);
        inferRequest->start_async();
        return;
    }

    void load() {
        ov::Core core_;

        // Load model to device
        auto devices = parse_devices(ov_properties_.device);

        auto check_device_supports = [&core_](const std::string& device_name,
                                            const std::string& property_name) {
            auto supported_config_keys =
                core_.get_property(device_name, ov::supported_properties);
            if (std::find(supported_config_keys.begin(), supported_config_keys.end(),
                    property_name) == supported_config_keys.end()) {
                throw std::logic_error(
                    "Device " + std::string(device_name) +
                    " doesn't support config key '" + property_name + "'! " +
                    "Please specify it for correct devices in format  "
                    "<dev1>:<value1>,<dev2>:<value2>");
            }
        };

        auto device_nstreams = parse_value_per_device(devices, ov_properties_.nstreams);
        for (auto& pair : device_nstreams) {
            check_device_supports(pair.first, ov::num_streams.name());
        }

        auto device_precisions = parse_value_per_device(devices, ov_properties_.infer_precision);
        for (auto& pair : device_precisions) {
            check_device_supports(pair.first, ov::hint::inference_precision.name());
        }

        if (!ov_properties_.extensions.empty()) {
            core_.add_extension(ov_properties_.extensions);
        }

        ov::AnyMap device_config = {};
        // Enable if needed
        device_config.insert(ov::hint::allow_auto_batching(ov_properties_.allow_auto_batching));
        for (auto& device : devices) {
            // Set number of streams for all scenarios except single stream
            if (settings_.scenario == mlperf::TestScenario::SingleStream) {
                device_config.insert(ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
            } else {
                device_config.insert(ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
                if (device_nstreams.count(device) && std::stoi(device_nstreams.at(device)) > 0) {
                    device_config.insert(ov::num_streams(ov::streams::Num(std::stoi(device_nstreams.at(device)))));
                }
            }
            if (device == "CPU") {
                if (ov_properties_.nthreads > 0) {
                    device_config.insert(ov::inference_num_threads(ov_properties_.nthreads));
                }
            }
            // Set inference precision of specified
            if (device_precisions.count(device)) {
                device_config.insert(ov::hint::inference_precision(device_precisions.at(device)));
            }
        }
        std::cout << "    [INFO] Input model: " << input_model_ << std::endl;
        auto model_ = core_.read_model(input_model_);

        std::cout << "    [INFO] Setting pre-processing..." << std::endl;
        auto preproc = ov::preprocess::PrePostProcessor(model_);
        for (const auto& input : model_->inputs()) {
            auto& in = preproc.input(input.get_any_name());
            if (workload_.compare("resnet50") == 0 ||
                workload_.compare("retinanet") == 0) {
                in.tensor().set_element_type(ov::element::u8);
            } else if (workload_.compare("bert") == 0) {
                in.tensor().set_element_type(ov::element::i32);
            }
        }
        model_ = preproc.build();

        print_input_outputs_info(model_);

        std::cout << "    [INFO] Setting batch size " + std::to_string(batch_size_) + "..." << std::endl;
        ov::set_batch(model_, batch_size_);

        std::cout << "    [INFO] Loading network to device " + ov_properties_.device + "..." << std::endl;
        compiled_model_ = core_.compile_model(model_, ov_properties_.device, device_config);

        std::cout << "    [INFO] Network loaded to device" << std::endl;
        auto supported_properties = compiled_model_.get_property(ov::supported_properties);
        for (const auto& cfg : supported_properties) {
            if (cfg == ov::supported_properties) continue;
            auto prop = compiled_model_.get_property(cfg);
            if (cfg == ov::optimal_number_of_infer_requests) ov_properties_.nireq = prop.as<uint32_t>();
            std::cout << "    [INFO] " << cfg << ": " << prop.as<std::string>() << std::endl;
        }
        std::cout << "    [INFO] Creating " << ov_properties_.nireq << " inference request(s)" << std::endl;
        if (settings_.scenario == mlperf::TestScenario::SingleStream) {
            set_infer_request();
        } else if (settings_.scenario == mlperf::TestScenario::Offline ||
                settings_.scenario == mlperf::TestScenario::MultiStream) {
            create_requests();
        } else if (settings_.scenario == mlperf::TestScenario::Server) {
            create_server_requests();
        }
    }

    uint32_t get_nireq() {
        return ov_properties_.nireq;
    }

public:
    ov::CompiledModel compiled_model_;
    InferRequestsQueue* inferRequestsQueue_;
    InferRequestsQueueServer* inferRequestsQueueServer_;
    ov::InferRequest inferRequest_;
    std::string input_model_;
    std::vector<std::string> output_blob_names_;
    std::vector<std::string> input_blob_names_;
    unsigned batch_size_ = 1;
    OVBackendProperties ov_properties_;

    mlperf::TestSettings settings_;
    std::string workload_;
    int object_size_;

    PPFunction post_processor_;
};

class OVBackendAsync : public OVBackendBase {
public:
    OVBackendAsync(mlperf::TestSettings settings,
                    OVBackendProperties ov_properties,
                    unsigned batch_size,
                    std::string workload,
                    std::vector<std::string> input_blob_names,
                    std::vector<std::string> output_blob_names,
                    std::string input_model, PPFunction post_processor)
        : OVBackendBase(settings, ov_properties, batch_size, workload, input_blob_names,
                        output_blob_names, input_model, post_processor) {}
};

class OVBackendServer : public OVBackendBase {
public:
    OVBackendServer(mlperf::TestSettings settings,
                    OVBackendProperties ov_properties,
                    unsigned batch_size,
                    std::string workload,
                    std::vector<std::string> input_blob_names,
                    std::vector<std::string> output_blob_names,
                    std::string input_model,
                    PPFunction post_processor)
        : OVBackendBase(settings, ov_properties, batch_size, workload, input_blob_names,
                        output_blob_names, input_model, post_processor) {}
};

#endif
