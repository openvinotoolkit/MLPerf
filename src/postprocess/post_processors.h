#pragma once

#include "utils.h"

namespace Processors {
void postprocess_ssd_retinanet(Item qitem, ov::InferRequest req,
        std::vector<float> &result, std::vector<mlperf::ResponseId> &response_ids,
        unsigned batch_size, std::vector<unsigned> &counts) {
    cv::Size image_size = {800, 800};
    float score_treshold = 0.05;

    unsigned count = 0;

    auto bbox_tensor = req.get_tensor("boxes");
    auto scores_tensor = req.get_tensor("scores");
    auto labels_tensor = req.get_tensor("labels");

    auto bbox_ptr = bbox_tensor.data<float>();
    auto scores_ptr = scores_tensor.data<float>();
    auto labels_ptr = labels_tensor.data<int64_t>();

    auto prediction_size = scores_tensor.get_shape()[0];

    auto sample_idxs = qitem.sample_idxs_;
    int kept_indexes = 0;
    for (int i = 0; i < prediction_size; i++) {
        auto score = scores_ptr[i];
        if (score >= score_treshold) {
            kept_indexes++;
            auto class_inx = labels_ptr[i];
            // box comes from model as: xmin, ymin, xmax, ymax
            // box comes with dimentions in the range of [0, height]
            // and [0, width] respectively. It is necesary to scale
            // them in the range [0, 1]
            result.push_back(float(sample_idxs[i]));
            result.push_back(bbox_ptr[i*4 + 1]/image_size.height);
            result.push_back(bbox_ptr[i*4 + 0]/image_size.width);
            result.push_back(bbox_ptr[i*4 + 3]/image_size.height);
            result.push_back(bbox_ptr[i*4 + 2]/image_size.width);
            result.push_back(score);
            result.push_back(float(class_inx));
        }
    }
    response_ids.push_back(qitem.response_ids_[0]);
    counts.push_back(7 * kept_indexes);
}

void postprocess_classification(Item qitem, ov::InferRequest req,
                                 std::vector<float>& results,
                                 std::vector<mlperf::ResponseId>& response_ids,
                                 unsigned batch_size,
                                 std::vector<unsigned>& counts,
                                 const std::string& output_name) {
    std::vector<unsigned> res;
    auto out = req.get_tensor(output_name);

    TopResults(1, out, res);

    for (size_t j = 0; j < res.size(); ++j) {
        results.push_back(static_cast<float>(res[j] - 1));
        response_ids.push_back(qitem.response_ids_[j]);
	    counts.push_back(1);
    }
}

void postprocess_resnet50(Item qitem,
                         ov::InferRequest req,
                         std::vector<float> &results,
                         std::vector<mlperf::ResponseId> &response_ids,
                         unsigned batch_size,
                         std::vector<unsigned> &counts) {
    const std::string output_name = "softmax_tensor:0";
    return postprocess_classification(qitem, req, results, response_ids, batch_size, counts, output_name);
}

void postprocess_bert_common(Item qitem,
                             ov::InferRequest req,
                             std::vector<float>& results,
                             std::vector<mlperf::ResponseId>& response_ids,
                             unsigned batch_size,
                             std::vector<unsigned>& counts,
                             const std::string& out_0_name,
                             const std::string& out_1_name) {
	auto out_0 = req.get_tensor(out_0_name);
	auto out_1 = req.get_tensor(out_1_name);

	size_t offset = out_0.get_size() / batch_size;
	const float* out_0_data = out_0.data<const float>();
	const float* out_1_data = out_1.data<const float>();

    size_t n0 = results.size();
    results.resize(n0 + 2 * out_0.get_size());

    for (size_t j = 0; j < batch_size; j++) {
        response_ids.push_back(qitem.response_ids_[j]);
        for (size_t i = 0, k = 0; i < offset; i++, k += 2) {

            results[n0 + j * offset + k] = out_0_data[i];
            results[n0 + j * offset + k + 1] = out_1_data[i];
        }
		counts.push_back(offset * 2);

		// Next sample
		out_0_data += offset;
		out_1_data += offset;
	}
}

void postprocess_bert(Item qitem,
                      ov::InferRequest req,
                      std::vector<float>& results,
                      std::vector<mlperf::ResponseId>& response_ids,
                      unsigned batch_size,
                      std::vector<unsigned>& counts) {
    const std::string out_0_name = "output_start_logits";
    const std::string out_1_name = "output_end_logits";
    return postprocess_bert_common(qitem, req, results, response_ids, batch_size, counts, out_0_name, out_1_name);
}
};