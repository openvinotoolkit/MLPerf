#pragma once

#include "sut_base.h"

class SUTServer : public SUTBase {
public:
    SUTServer(mlperf::TestSettings settings,
            QSLBase* ov_qsl,
            OVBackendProperties ov_properties,
            int batch_size,
            std::string dataset,
            std::string workload,
            std::vector<std::string> input_blob_names,
            std::vector<std::string> output_blob_names, std::string input_model,
            PPFunction post_processor)
        : online_bs_(),
        SUTBase(settings, ov_qsl, ov_properties, batch_size, dataset, workload, input_blob_names,
                output_blob_names, input_model, post_processor, true) {}

    const std::string& Name() override {
        static const std::string name("OpenVINO Server SUT");
        return name;
    }

    void WarmUp(size_t nwarmup_iters) override {
        backend_ov_async_->set_server_warmup(true);

        std::vector<mlperf::QuerySampleIndex> samples;
        std::vector<mlperf::ResponseId> response_ids;

        for (size_t i = 0; i < batch_size_; ++i) {
            samples.push_back(0);
            response_ids.push_back(1);
        }

        ov_qsl_->LoadSamplesToRam(samples);

        std::vector<Blob::Ptr> data;
        Item item;

        ov_qsl_->GetSample(samples, response_ids, 1, &item);

        for (size_t i = 0; i < nwarmup_iters; ++i) {
            backend_ov_async_->predict_async_server(item);
        }

        ov_qsl_->UnloadSamplesFromRam(samples);

        backend_ov_async_->reset();
    }

    void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override {
        std::vector<mlperf::QuerySampleResponse> responses;
        std::vector<mlperf::QuerySampleIndex> sample_idxs;
        std::vector<mlperf::ResponseId> response_ids;

        for (size_t i = 0; i < samples.size(); ++i) {
            sample_idxs.push_back(samples[i].index);
            response_ids.push_back(samples[i].id);
        }

        Item item;
        ov_qsl_->GetSample(sample_idxs, response_ids, batch_size_, &item);
        RunOneItem(response_ids, item);
    }

 private:
    void RunOneItem(std::vector<mlperf::ResponseId> response_id, Item item) {
        backend_ov_async_->predict_async_server(item);
    }
    int qid = 0;
    size_t online_bs_;
};
