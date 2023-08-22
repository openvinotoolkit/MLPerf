#pragma once

#include "image_dataset.h"

class Imagenet : public ImageDataset {
public:
    Imagenet(mlperf::TestSettings settings, size_t image_width, size_t image_height,
            size_t num_channels, string datapath, string image_format,
            int total_count, int perf_count, const mlperf_ov::WorkloadName& workload_name, const mlperf_ov::DatasetName& dataset_name) :
        ImageDataset(settings, image_width, image_height, num_channels, datapath, image_format, total_count, perf_count, workload_name, dataset_name) {
        string image_list_file = datapath + "/val_map.txt";

        boost::filesystem::path p(image_list_file);
        if (!(boost::filesystem::exists( p ) ) ) {
            std::cout << " Imagenet validation list file '" + image_list_file + "' not found. "
                        "Please make sure data_path contains 'val_map.txt'\n";
            throw;
        }

        std::ifstream imglistfile;
        imglistfile.open(image_list_file, std::ios::binary);

        std::string line, image_name, label;
        if (imglistfile.is_open()) {
            while (getline(imglistfile, line)) {
                std::regex ws_re("\\s+");
                std::vector < std::string
                        > img_label { std::sregex_token_iterator(line.begin(),
                                line.end(), ws_re, -1), { } };
                label = img_label.back();
                image_name = img_label.front();
                img_label.clear();

                this->image_list_.push_back(image_name);
                this->label_list_.push_back(stoi(label));

                // limit dataset
                if (total_count_
                        && (image_list_.size() >= (uint) total_count_)) {
                    break;
                }
            }
        }

        imglistfile.close();

        if (!image_list_.size()) {
            std::cout << "No images in image list found";
        }
    }

    const std::string& Name() override {
        static const std::string name("OpenVINO ImageNet2012 QSL");
        return name;
    }

    ~Imagenet() {}
};