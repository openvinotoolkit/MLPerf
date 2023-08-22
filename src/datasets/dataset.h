#pragma once

#include <map>

// loadgen
#include "loadgen.h"
#include "query_sample.h"
#include "query_sample_library.h"
#include "test_settings.h"
#include "system_under_test.h"
#include "bindings/c_api.h"

#include "item_ov.h"
#include "workload_helpers.h"

class QSLBase : mlperf::QuerySampleLibrary {
public:
    QSLBase(mlperf::TestSettings settings, const std::string& datapath,
            size_t total_count, size_t perf_count,
            const mlperf_ov::WorkloadName& workload_name, const mlperf_ov::DatasetName& dataset_name) :
        datapath_(datapath), total_count_(total_count), perf_count_(perf_count),
        settings_(settings), workload_name_(workload_name), dataset_name_(dataset_name), handle() {}

    ~QSLBase() {}

    const std::string& Name() override {
        static const std::string name("OpenVINO Base QSL");
        return name;
    }

    size_t TotalSampleCount() override {
        throw std::runtime_error("TotalSampleCount is not implemented in base class!");
    }

    size_t PerformanceSampleCount() override {
        return perf_count_;
    }

    void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) override {
        throw std::runtime_error("LoadSamplesToRam is not implemented in base class!");
    }

    void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples) override {
        throw std::runtime_error("UnloadSamplesFromRam is not implemented in base class!");
    }

    virtual void GetSample(const std::vector<mlperf::QuerySampleIndex> samples,
        std::vector<mlperf::ResponseId> query_ids, size_t bs, Item *item) = 0;

    virtual void GetSamplesBatched(const std::vector<mlperf::QuerySampleIndex> samples,
        std::vector<mlperf::ResponseId> query_ids, size_t bs, int num_batches, std::vector<Item> &items) = 0;

    virtual void GetSamplesBatchedServer(const std::vector<mlperf::QuerySampleIndex> samples,
        std::vector<mlperf::ResponseId> query_ids, size_t bs, int num_batches, std::vector<Item> &items) = 0;

    virtual void GetSamplesBatchedMultistream(const std::vector<mlperf::QuerySampleIndex> samples,
        std::vector<mlperf::ResponseId> query_ids, size_t bs, int num_batches, std::vector<Item> &items) = 0;

    std::vector<mlperf::QuerySampleIndex> sample_list_inmemory_;
    size_t total_count_;
    size_t perf_count_;
    mlperf::TestSettings settings_;
    std::string datapath_;
    mlperf_ov::WorkloadName workload_name_;
    mlperf_ov::DatasetName dataset_name_;
    unsigned char * handle;
};