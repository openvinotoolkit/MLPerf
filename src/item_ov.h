#ifndef ITEM_H__
#define ITEM_H__

#include <map>
#include <vector>

// loadgen
#include <openvino/openvino.hpp>

#include "bindings/c_api.h"
#include "loadgen.h"
#include "query_sample.h"
#include "query_sample_library.h"
#include "system_under_test.h"
#include "test_settings.h"

using namespace InferenceEngine;

class Item {
public:
    Item(ov::Tensor tensor, std::vector<mlperf::ResponseId> response_ids,
        std::vector<mlperf::QuerySampleIndex> sample_idxs)
        : tensors_({tensor}),
        response_ids_(response_ids),
        sample_idxs_(sample_idxs),
        label_() {}

    Item(ov::Tensor tensor, std::vector<mlperf::ResponseId> response_ids,
        std::vector<mlperf::QuerySampleIndex> sample_idxs, int label)
        : tensors_({tensor}),
        response_ids_(response_ids),
        sample_idxs_(sample_idxs),
        label_(label) {}

    Item(ov::Tensor tensor, std::vector<mlperf::QuerySampleIndex> sample_idxs)
        : tensors_({tensor}),
        sample_idxs_(sample_idxs) {}

    Item(std::vector<ov::Tensor> tensors,
        std::vector<mlperf::ResponseId> response_ids,
        std::vector<mlperf::QuerySampleIndex> sample_idxs)
        : tensors_(tensors),
        response_ids_(response_ids),
        sample_idxs_(sample_idxs),
        label_() {}

    Item() : label_() {}

public:
    std::vector<mlperf::ResponseId> response_ids_;
    std::vector<mlperf::QuerySampleIndex> sample_idxs_;
    std::vector<ov::Tensor> tensors_;
    int label_;
};

#endif
