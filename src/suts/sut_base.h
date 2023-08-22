#pragma once

#include <condition_variable>
#include <list>
#include <mutex>

// loadgen
#include "backend_ov.h"
#include "bindings/c_api.h"
#include "item_ov.h"
#include "loadgen.h"
#include "query_sample.h"
#include "query_sample_library.h"
#include "system_under_test.h"
#include "test_settings.h"

class SUTBase : public mlperf::SystemUnderTest {
 public:
    SUTBase(mlperf::TestSettings settings,
            QSLBase* ov_qsl,
            OVBackendProperties ov_properties,
            int batch_size,        // Batch size
            std::string dataset,
            std::string workload,
            std::vector<std::string> input_blob_names,
            std::vector<std::string> output_blob_names,
            std::string input_model,
            PPFunction post_processor,
            bool async = false)
        : settings_(settings),
        ov_qsl_(ov_qsl),
        ov_properties_(ov_properties),
        batch_size_(batch_size),
        workload_(workload) {
            if (async) {
                backend_ov_async_ = std::unique_ptr<OVBackendAsync>(new OVBackendAsync(
                    settings, ov_properties, batch_size, workload, input_blob_names, output_blob_names,
                    input_model, post_processor));
                backend_ov_async_->load();
            } else {
                backend_ov_ = std::unique_ptr<OVBackendBase>(new OVBackendBase(
                    settings, ov_properties, batch_size, workload, input_blob_names, output_blob_names,
                    input_model, post_processor));
                backend_ov_->load();
            }
        comm_count_ = 1;
    }
    const std::string& Name() override {
        static const std::string name("OpenVINO SingeStream SUT");
        return name;
    }

    virtual void WarmUp(size_t nwarmup_iters) {
        std::vector<mlperf::QuerySampleIndex> samples;
        std::vector<mlperf::ResponseId> response_ids;

        for (size_t i = 0; i < batch_size_; ++i) {
            samples.push_back(0);
            response_ids.push_back(1);
        }

        ov_qsl_->LoadSamplesToRam(samples);
        std::vector<float> results;
        std::vector<unsigned> counts;

        for (size_t i = 0; i < nwarmup_iters; ++i) {
            ov_qsl_->GetSample(samples, response_ids, 1, &qitem_);
            backend_ov_->warmup(qitem_);
        }
        ov_qsl_->UnloadSamplesFromRam(samples);
    }

    void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override {
        std::vector<mlperf::QuerySampleResponse> responses;
        std::vector<mlperf::QuerySampleIndex> sample_idxs{(samples[0].index)};
        std::vector<mlperf::ResponseId> response_ids{(samples[0].id)};

        int num_batches = samples.size() / batch_size_;
        ov_qsl_->GetSample(sample_idxs, response_ids, 1, &qitem_);

        RunOneItem(response_ids);
    }

    void FlushQueries() override { return; }

private:
    void RunOneItem(std::vector<mlperf::ResponseId> response_id) {
        std::vector<float> result;
        std::vector<unsigned> counts;

        std::vector<mlperf::QuerySampleResponse> responses;
        backend_ov_->predict(qitem_, result, response_id, counts);

        mlperf::QuerySampleResponse response{
            response_id[0], reinterpret_cast<std::uintptr_t>(&result[0]),
            sizeof(float) * counts[0]};
        responses.push_back(response);

        mlperf::QuerySamplesComplete(responses.data(), responses.size());
    }

public:
    mlperf::TestSettings settings_;
    OVBackendProperties ov_properties_;
    int batch_size_ = 1;
    QSLBase* ov_qsl_;
    std::string workload_;
    std::unique_ptr<OVBackendBase> backend_ov_;
    std::unique_ptr<OVBackendAsync> backend_ov_async_;
    Item qitem_;
    std::vector<Item> qitems_;
    int comm_count_;
};