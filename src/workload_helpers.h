#pragma once

#include<string>
#include<vector>
#include<map>

namespace mlperf_ov {
	enum class WorkloadName {
		ResNet50,
        RetinaNet,
		Bert,
	};

    enum class DatasetName {
        ImageNet2012,
        OpenImages_v6,
        SQuAD_v1_1,
    };

	class WorkloadBase {
	public:
		WorkloadBase(const std::vector<std::string>& input_names,
                     const std::vector<std::string>& output_names,
                     const DatasetName& dataset_name,
                     const WorkloadName& workload_name) :
            input_names_(input_names),
            output_names_(output_names),
            dataset_name_(dataset_name),
            workload_name_(workload_name) {}
        WorkloadName get_workload_name() { return workload_name_; };
        DatasetName get_dataset_name() { return dataset_name_; };
        std::vector<std::string> get_input_names() { return input_names_; };
        std::vector<std::string> get_output_names() { return output_names_; };
    protected:
		std::vector<std::string> input_names_, output_names_;
        WorkloadName workload_name_;
		DatasetName dataset_name_;
	};
;
	class ResNet50 : public WorkloadBase {
	public:
		ResNet50() : WorkloadBase({ "input_tensor:0" },
                                  { "softmax_tensor:0" },
                                  DatasetName::ImageNet2012,
                                  WorkloadName::ResNet50) {}
		~ResNet50(){};
		void postprocess(){};
	};
	class RetinaNet : public WorkloadBase {
    public:
        RetinaNet() : WorkloadBase({ "images" },
                                   { "boxes", "scores", "labels" },
                              DatasetName::OpenImages_v6,
                              WorkloadName::RetinaNet) {}
        ~RetinaNet(){};
        void postprocess(){};
    };
	class Bert : public WorkloadBase {
    public:
        Bert() : WorkloadBase({ "input_ids", "input_mask", "segment_ids" },
                              { "output_start_logits", "output_end_logits" },
                              DatasetName::SQuAD_v1_1,
                              WorkloadName::Bert) {}
        ~Bert(){};
        void postprocess(){};
    };
}; // namespace mlperf_ov

std::ostream& operator<<(std::ostream& os, const mlperf_ov::WorkloadName& wn) {
    const std::map<mlperf_ov::WorkloadName, std::string> workload_name_map = {
        { mlperf_ov::WorkloadName::ResNet50, "resnet50"},
        { mlperf_ov::WorkloadName::Bert, "bert"},
        { mlperf_ov::WorkloadName::RetinaNet, "retinanet"},
    };
    os << workload_name_map.at(wn);
    return os;
}

std::ostream& operator<<(std::ostream& os, const mlperf_ov::DatasetName& dn) {
    const std::map<mlperf_ov::DatasetName, std::string> databset_name_map = {
        { mlperf_ov::DatasetName::ImageNet2012, "imagenet"},
        { mlperf_ov::DatasetName::OpenImages_v6, "openimages"},
        { mlperf_ov::DatasetName::SQuAD_v1_1, "squad"},
    };
    os << databset_name_map.at(dn);
    return os;
}