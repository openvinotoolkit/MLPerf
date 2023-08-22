#pragma once

#include "sut_base.h"

class SUTMultistream : public SUTBase {
public:
    SUTMultistream(mlperf::TestSettings settings,
                   QSLBase* ov_qsl,
                   OVBackendProperties ov_properties,
                   int batch_size,
                   std::string dataset,
                   std::string workload,
                   std::vector<std::string> input_blob_names,
                   std::vector<std::string> output_blob_names,
                   std::string input_model,
                   PPFunction post_processor)
        : SUTBase(settings, ov_qsl, ov_properties, batch_size, dataset, workload, input_blob_names,
                output_blob_names, input_model, post_processor, true) {}

    const std::string& Name() override {
        static const std::string name("OpenVINO MultiStream SUT");
        return name;
    }

    void WarmUp(size_t nwarmup_iters) override {
        std::vector<mlperf::QuerySampleIndex> samples;
        std::vector<mlperf::ResponseId> query_ids;

        for (size_t i = 0; i < batch_size_ * backend_ov_async_->get_nireq(); ++i) {
            samples.push_back(0);
            query_ids.push_back(1);
        }

        ov_qsl_->LoadSamplesToRam(samples);
        std::cout << " == Starting Warmup ==\n";
        for (size_t i = 0; i < nwarmup_iters; ++i) {
            std::vector<mlperf::ResponseId> results_ids;
            std::vector<mlperf::QuerySampleIndex> sample_idxs;

            std::vector<Item> items;
            ov_qsl_->GetSamplesBatchedMultistream(samples, query_ids, batch_size_, backend_ov_async_->get_nireq(), items);
            backend_ov_async_->predict_async(items);
        }
        ov_qsl_->UnloadSamplesFromRam(samples);
        backend_ov_async_->reset();
        std::cout << " == Warmup Completed ==\n";
    }

    void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override {
        std::vector<mlperf::QuerySampleResponse> responses;
        std::vector<mlperf::QuerySampleIndex> sample_idxs;
        std::vector<mlperf::ResponseId> response_ids;

        for (size_t i = 0; i < samples.size(); ++i) {
            sample_idxs.push_back(samples[i].index);
            response_ids.push_back(samples[i].id);
        }

        int num_batches = samples.size() / batch_size_;
        ov_qsl_->GetSamplesBatchedMultistream(sample_idxs, response_ids, batch_size_, num_batches, qitems_);
        RunOneItem(response_ids);
    }
private:
    void RunOneItem(std::vector<mlperf::ResponseId> response_id) {
        backend_ov_async_->predict_async(qitems_);

        std::vector<mlperf::QuerySampleResponse> responses =
            backend_ov_async_->get_query_sample_responses();

        mlperf::QuerySamplesComplete(responses.data(), responses.size());
        backend_ov_async_->reset();
        qitems_.clear();
    }
};

