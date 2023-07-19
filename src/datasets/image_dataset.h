#pragma once

#include <boost/property_tree/json_parser.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <map>

// loadgen
#include "loadgen.h"
#include "query_sample.h"
#include "query_sample_library.h"
#include "test_settings.h"
#include "dataset.h"

#include <openvino/openvino.hpp>

using namespace ov;
using namespace std;
using namespace cv;

class ImageDataset : public QSLBase {
public:
    ImageDataset(mlperf::TestSettings settings, size_t image_width, size_t image_height,
            size_t num_channels, string datapath, const std::string& image_format,
            size_t total_count, size_t perf_count, const mlperf_ov::WorkloadName& workload_name, const mlperf_ov::DatasetName& dataset_name) :
            QSLBase(settings, datapath, total_count, perf_count, workload_name, dataset_name),
            image_width_(image_width), image_height_(image_height),
            num_channels_(num_channels),
            image_format_(image_format) {}

    ~ImageDataset() {}

    const std::string& Name() override {
        static const std::string name("OpenVINO Image QSL");
        return name;
    }

    size_t TotalSampleCount() override {
        return image_list_.size();
    }

    void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) override {
        if (image_list_inmemory_.size() == 0) {
            if ((this->settings_.scenario == mlperf::TestScenario::SingleStream)
                    || (this->settings_.scenario == mlperf::TestScenario::Server)) {
                this->image_list_inmemory_.resize(total_count_);
            } else if (this->settings_.scenario == mlperf::TestScenario::Offline) {
                image_list_inmemory_.resize(samples.size());
            } else if (this->settings_.scenario == mlperf::TestScenario::MultiStream) {
                image_list_inmemory_.resize(samples.size());
                sample_list_inmemory_.resize(samples.size());
            }
        }

        mlperf::QuerySampleIndex sample;

        ov::Shape shape;
        handle = (new unsigned char[samples.size() * num_channels_ * image_height_ * image_width_]);

        for (uint i = 0; i < samples.size(); ++i) {
            cv::Mat processed_image;
            sample = samples[i];

            std::string image_path;

            if (sample >= image_list_.size()) {
                throw std::logic_error("Sample is out of image list: " +
                    std::to_string(sample) + " >= " + std::to_string(image_list_.size()));
            }

            if (dataset_name_ == mlperf_ov::DatasetName::ImageNet2012) {
                image_path = this->datapath_ + "/" + image_list_[sample];
            }
            else if (dataset_name_ == mlperf_ov::DatasetName::OpenImages_v6) {
                image_path = this->datapath_ + "/validation/data/" + image_list_[sample];
            }

            auto image = cv::imread(image_path);

            if (image.empty()) {
                throw std::logic_error("Invalid image at path: " + image_path);
            }

            if (this->workload_name_ == mlperf_ov::WorkloadName::ResNet50) {
                shape = ov::Shape{ 1, num_channels_, image_height_, image_width_ };
                preprocess_resnet50(&image, &processed_image);
            } else if (this->workload_name_ == mlperf_ov::WorkloadName::RetinaNet) {
                shape = ov::Shape{ 1, num_channels_, image_height_, image_width_ };
                preprocess_retinanet(&image, &processed_image);
            } else {
                std::stringstream ss;
                ss << "Workload is not supported: " << this->workload_name_;
                throw std::runtime_error(ss.str());
            }

            processed_image.copyTo(image);
            size_t image_size = (image_height_ * image_width_);

            ov::Tensor input_tensor = ov::Tensor(ov::element::u8, shape,
                ((((unsigned char *) handle) + (i * image_size * num_channels_))));
            auto input_data = input_tensor.data<unsigned char>();

            for (size_t image_id = 0; image_id < 1; ++image_id) {
                for (size_t pid = 0; pid < image_size; pid++) {
                    for (size_t ch = 0; ch < num_channels_; ++ch) {
                        input_data[image_id * image_size * num_channels_
                                + ch * image_size + pid] = image.at<cv::Vec3b>(pid)[ch];
                    }
                }
            }

            if (settings_.scenario == mlperf::TestScenario::Offline) {
                image_list_inmemory_[i] = input_tensor;
            } else if ((settings_.scenario == mlperf::TestScenario::SingleStream)
                    || (settings_.scenario == mlperf::TestScenario::Server)) {
                image_list_inmemory_[sample] = input_tensor;
            } else if (this->settings_.scenario == mlperf::TestScenario::MultiStream) {
                image_list_inmemory_[i] = input_tensor;
                sample_list_inmemory_[i] = sample;
            }
        }
    }

    void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples) override {
        image_list_inmemory_.clear();
        delete [] handle;
    }

    void GetSamples(const mlperf::QuerySampleIndex* samples, ov::Tensor* data, int* label) {
        *data = image_list_inmemory_[samples[0]];
        if (dataset_name_ == mlperf_ov::DatasetName::ImageNet2012) {
            *label = label_list_[samples[0]];
        }
    }

    void GetSample(const std::vector<mlperf::QuerySampleIndex> samples,
            std::vector<mlperf::ResponseId> query_ids, size_t bs, Item *item) {
        auto sample_idx = samples[0];
        std::vector<ov::Tensor> input { image_list_inmemory_[sample_idx] };
		*item = Item(input, query_ids, samples);
    }

    void GetSamplesBatched(const std::vector<mlperf::QuerySampleIndex> samples,
            std::vector<mlperf::ResponseId> query_ids, size_t bs, int num_batches, std::vector<Item> &items) {
        size_t image_size = (image_height_ * image_width_);
        ov::Shape shape { (bs), num_channels_, image_height_, image_width_ };

		for (int i = 0; i < num_batches; ++i) {
			auto start = (i * bs) % perf_count_;
            std::vector<ov::Tensor> inputs { ov::Tensor(ov::element::u8, shape, image_list_inmemory_[start].data())};

			std::vector<mlperf::QuerySampleIndex> idxs;
			std::vector<mlperf::ResponseId> ids;
			for (size_t j = (i * bs); j < (unsigned) ((i * bs) + bs); ++j) {
				ids.push_back(query_ids[j]);
				idxs.push_back(samples[j]);
			}
			items.push_back(Item(inputs, ids, idxs));
		}
    }


    void GetSamplesBatchedServer(const std::vector<mlperf::QuerySampleIndex> samples,
            std::vector<mlperf::ResponseId> query_ids, size_t bs, int num_batches, std::vector<Item> &items) {
        size_t image_size = (image_height_ * image_width_);
        ov::Shape shape { (bs), num_channels_, image_height_, image_width_ };

        for (int i = 0; i < num_batches; ++i) {
            ov::Tensor input = ov::Tensor(ov::element::u8, shape);
            for (size_t k = 0; k < bs; ++k) {
                auto start = samples[k];
                std::memcpy((input.data<unsigned char>() + (k * image_size * num_channels_)),
                        image_list_inmemory_[start].data(),
                        (image_size * num_channels_ * sizeof(unsigned char)));
            }
            std::vector<mlperf::QuerySampleIndex> idxs;
            std::vector<mlperf::ResponseId> ids;
            for (size_t j = 0; j < (unsigned) bs; ++j) {
                ids.push_back(query_ids[j]);
                idxs.push_back(samples[j]);
            }

            items.push_back(Item(input, ids, idxs));
        }
    }

    void GetSamplesBatchedMultistream(
            const std::vector<mlperf::QuerySampleIndex> samples,
            std::vector<mlperf::ResponseId> query_ids, size_t bs,
            int num_batches, std::vector<Item> &items) {
        size_t image_size = (image_height_ * image_width_);
        ov::Shape shape { (bs), num_channels_, image_height_, image_width_ };

        // find sample
        std::vector<mlperf::QuerySampleIndex>::iterator it = std::find(
                sample_list_inmemory_.begin(), sample_list_inmemory_.end(),
                samples[0]);
        if (!(it != std::end(sample_list_inmemory_))) {
            std::cout << "    [ERROR] No sample found\n ";
            throw std::logic_error("Aborted");
        }

        int index = std::distance(sample_list_inmemory_.begin(), it);
        auto start = index;
        auto query_start = 0;
        for (int i = 0; i < num_batches; ++i) {
            auto input = ov::Tensor(ov::element::u8, shape, image_list_inmemory_[start].data());

            std::vector<mlperf::QuerySampleIndex> idxs;
            std::vector<mlperf::ResponseId> ids;
            for (size_t j = (query_start); j < (unsigned) (query_start + bs);
                    ++j) {
                ids.push_back(query_ids[j]);
                idxs.push_back(samples[j]);
            }

            items.push_back(Item(input, ids, idxs));

            start = start + bs;
            query_start = query_start + bs;
        }
    }

    // Preprocessing routines
    void center_crop(cv::Mat* image, int out_height, int out_width,
            cv::Mat* cropped_image) {
        int width = (*image).cols;
        int height = (*image).rows;
        int left = int((width - out_width) / 2);
        int top = int((height - out_height) / 2);
        cv::Rect custom_roi(left, top, out_width, out_height);

        (*cropped_image) = (*image)(custom_roi);
    }

    void resize_with_aspect_ratio(cv::Mat* image, cv::Mat* resized_image,
            int out_height, int out_width, int interpol, float scale = 87.5) {
        int width = (*image).cols;
        int height = (*image).rows;
        int new_height = int(100. * out_height / scale);
        int new_width = int(100. * out_width / scale);

        int h, w = 0;
        if (height > width) {
            w = new_width;
            h = int(new_height * height / width);
        } else {
            h = new_height;
            w = int(new_width * width / height);
        }

        cv::resize((*image), (*resized_image), cv::Size(w, h), 0, 0, interpol);
    }

    void preprocess_resnet50(cv::Mat* image, cv::Mat* processed_image, bool bgr=false) {
        cv::Mat img, resized_image, cropped_image, float_image, norm_image,
                sub_image;
        cv::cvtColor(*image, img, cv::COLOR_BGR2RGB);
        resize_with_aspect_ratio(&img, &resized_image, image_height_, image_width_,
                cv::INTER_AREA);

        center_crop(&resized_image, image_height_, image_width_, &cropped_image);

        if(bgr) {
            cv::Mat cropped_image2;
            cv::cvtColor(cropped_image, cropped_image2, cv::COLOR_RGB2BGR);
            cropped_image2.copyTo(*processed_image);
            return;
        }
        cropped_image.copyTo(*processed_image);
    }

    void preprocess_retinanet(cv::Mat* image, cv::Mat* processed_image) {
        cv::Mat img, resized_image;

        cv::cvtColor(*image, img, cv::COLOR_BGR2RGB);

        // Resize after scaling
        cv::resize(img, resized_image, cv::Size(image_height_, image_width_));

        resized_image.copyTo(*processed_image);
    }

public:
    std::vector<string> image_list_;
    std::vector<int> label_list_;
    std::vector<ov::Tensor> image_list_inmemory_;
    std::vector<std::pair<ov::Tensor, ov::Tensor>> data_list_inmemory_;
    size_t image_width_;
    size_t image_height_;
    size_t num_channels_;
    string image_format_;
};
