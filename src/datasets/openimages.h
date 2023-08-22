#pragma once

#ifdef _MSC_VER
#include <boost/config/compiler/visualc.hpp>
#endif
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>

#include "image_dataset.h"

class OpenImages: public ImageDataset {
public:
    OpenImages(mlperf::TestSettings settings, size_t image_width, size_t image_height,
            size_t num_channels, string datapath, string image_format,
            int total_count, int perf_count, const mlperf_ov::WorkloadName& workload_name, const mlperf_ov::DatasetName& dataset_name) :
        ImageDataset(settings, image_width, image_height, num_channels, datapath, image_format, total_count, perf_count, workload_name, dataset_name) {
        std::string annotations_file = datapath + "/annotations/openimages-mlperf.json";

        boost::filesystem::path p(annotations_file);
        if (!(boost::filesystem::exists(p))) {
            std::cout << " OpenImages validation list file '" + annotations_file + "' not found. "
                        "Please make sure data_path contains 'openimages-mlperf.json'\n";
            throw;
        }

        std::ifstream ifs(annotations_file, std::ios::binary);

        boost::property_tree::ptree pt;
        boost::property_tree::read_json(ifs, pt);
        BOOST_FOREACH(boost::property_tree::ptree::value_type &v, pt.get_child("images")){
            BOOST_FOREACH(boost::property_tree::ptree::value_type& item, v.second) {
                size_t id, width = 0, height = 0;
                std::string filename = "";
                if (item.first == "id") {
                    id = item.second.get_value<int>();
                    ids_.push_back(id);
                } else if (item.first == "file_name") {
                    filename = item.second.get_value<std::string>();
                    image_list_.push_back(filename);
                } else if (item.first == "height") {
                    height = item.second.get_value<int>();
                    assert(width == 0);
                    sizes_.push_back(cv::Size(height, width));
                } else if (item.first == "width") {
                    width = item.second.get_value<int>();
                    ids_.push_back(id);
                }
                if (total_count_
                    && (image_list_.size() >= (uint) total_count_)) {
                    break;
                }
            }
        }
        ifs.close();

        if (!image_list_.size()) {
            std::cout << "No images in image list found";
        }
    }

    const std::string& Name() override {
        static const std::string name("OpenVINO OpenImages v6 QSL");
        return name;
    }

    ~OpenImages() {}
private:
    std::vector<int> ids_;
    std::vector<cv::Size> sizes_;
};