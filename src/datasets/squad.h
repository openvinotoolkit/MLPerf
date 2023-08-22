#pragma once

#include "dataset.h"

using namespace ov;
using namespace std;
using namespace cv;

class Squad : public QSLBase {
public:
	Squad(mlperf::TestSettings settings, int max_seq_length, int max_query_length,
          int doc_stride, const std::string& datapath, size_t total_count, size_t perf_count,
          const mlperf_ov::WorkloadName& workload_name, const mlperf_ov::DatasetName& dataset_name) :
        QSLBase(settings, datapath, total_count, perf_count, workload_name, dataset_name) {
        string vocab_file = datapath  + "/vocab.txt";
	    string data_json = datapath + "/dev-v1.1.json";
        string output_dir = datapath + "/samples_cache";

	    boost::filesystem::path p_vocab(vocab_file);
        if (!(boost::filesystem::exists(p_vocab))) {
            throw std::logic_error(" SQUAD vocab file '" + vocab_file + "' not found. Please make sure --data_path contains 'vocab.txt'");
        }

        boost::filesystem::path p_json(data_json);
        if (!(boost::filesystem::exists(p_json))) {
            throw std::logic_error(" SQUAD data file '" + data_json + "' not found. Please makesure --data_path contains 'dev-v1.1.json'");
        }

        string output_json = output_dir + "/squad_examples.json";
        boost::filesystem::path o_json(output_json);

        if (!(boost::filesystem::exists(o_json) )) {
            std::cout << "    [INFO] Preprocessing SQuAD samples\n";
                const std::string cmd = "python3 " + datapath + "/tools/convert.py --vocab_file " + vocab_file + " --output_dir " + output_dir + " --test_file " + data_json;

                int ret_val = system(cmd.c_str());
        }

        if (!(boost::filesystem::exists(o_json))) {
            throw std::logic_error(" SQUAD data preprocessed file '" + output_json + "' not found.");
        } else {
            cout<< "    [INFO] Reading SQuAD data preprocessed file at: "<< output_json <<"\n";
        }
        std::vector<int> sample_input_ids_;
        std::vector<int> sample_input_mask_;
        std::vector<int> sample_segment_ids_;
        std::ifstream squadjsonfile(output_json);
        if (squadjsonfile) {
            std::stringstream buffer;

            buffer << squadjsonfile.rdbuf();
            boost::property_tree::ptree pt;
            boost::property_tree::read_json(buffer, pt);
            for (auto& sample : pt.get_child("samples")) {
                for (auto& prop : sample.second) {
                    if (prop.first == "segment_ids") {
                        boost::property_tree::ptree subtree = (boost::property_tree::ptree) prop.second ;
                        BOOST_FOREACH(boost::property_tree::ptree::value_type &vs,
                            subtree) {
                            sample_segment_ids_.push_back(std::stoi(vs.second.data()));
                        }
                        this->squad_segment_ids_.push_back(
                                sample_segment_ids_);
                        sample_segment_ids_.clear();
                    }
                    if (prop.first == "input_mask") {
                        boost::property_tree::ptree subtree = (boost::property_tree::ptree) prop.second ;
                        BOOST_FOREACH(boost::property_tree::ptree::value_type &vs,
                            subtree) {
                            sample_input_mask_.push_back(std::stoi(vs.second.data()));
                        }
                        this->squad_input_mask_.push_back(
                                sample_input_mask_);
                        sample_input_mask_.clear();
                    }
                    if (total_count_
                        && (squad_input_ids_.size() >= (uint) total_count_)) {
                        break;
                    }
                }
            }
            for (auto& sample : pt.get_child("samples")) {
                for (auto& prop : sample.second) {
                    if (prop.first == "input_ids") {
                        boost::property_tree::ptree subtree = (boost::property_tree::ptree) prop.second ;
                        BOOST_FOREACH(boost::property_tree::ptree::value_type &vs,
                            subtree) {
                            sample_input_ids_.push_back(std::stoi(vs.second.data()));
                        }
                        this->squad_input_ids_.push_back(
                                sample_input_ids_);
                        sample_input_ids_.clear();
                    }
                    if (total_count_
                        && (squad_input_ids_.size() >= (uint) total_count_)) {
                        break;
                    }
                }
            }
        }
    }

    size_t TotalSampleCount() override {
        return squad_input_ids_.size();
    }

    void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) override {
        if (input_ids_inmemory_.size() == 0) {
            if ((this->settings_.scenario == mlperf::TestScenario::SingleStream)
                    || (this->settings_.scenario == mlperf::TestScenario::Server)) {
                this->input_ids_inmemory_.resize(total_count_);
                this->input_mask_inmemory_.resize(total_count_);
                this->segment_ids_inmemory_.resize(total_count_);
            } else if (this->settings_.scenario == mlperf::TestScenario::Offline) {
                input_ids_inmemory_.resize(samples.size());
                input_mask_inmemory_.resize(samples.size());
                segment_ids_inmemory_.resize(samples.size());
            } else if (this->settings_.scenario == mlperf::TestScenario::MultiStream) {
                input_ids_inmemory_.resize(samples.size());
                input_mask_inmemory_.resize(samples.size());
                segment_ids_inmemory_.resize(samples.size());
                sample_list_inmemory_.resize(samples.size());
            }
        }

        mlperf::QuerySampleIndex sample;

        for (uint i = 0; i < samples.size(); ++i) {
            sample = samples[i];

            ov::Shape shape{1, max_seq_length_};
            ov::Tensor m_inp0 = ov::Tensor(ov::element::i32, shape);
            ov::Tensor m_inp1 = ov::Tensor(ov::element::i32, shape);
            ov::Tensor m_inp2 = ov::Tensor(ov::element::i32, shape);

            for (size_t j = 0; j < max_seq_length_; j++){
                m_inp0.data<int32_t>()[j] = static_cast<int32_t>(squad_input_ids_.at(sample).at(j));
                m_inp1.data<int32_t>()[j] = static_cast<int32_t>(squad_input_mask_.at(sample).at(j));
                m_inp2.data<int32_t>()[j] = static_cast<int32_t>(squad_segment_ids_.at(sample).at(j));
            }

            if (settings_.scenario == mlperf::TestScenario::Offline) {
                input_ids_inmemory_[i] = m_inp0;
                input_mask_inmemory_[i] = m_inp1;
		        segment_ids_inmemory_[i] = m_inp2;
            } else if ((settings_.scenario == mlperf::TestScenario::SingleStream)
                    || (settings_.scenario == mlperf::TestScenario::Server)) {
                input_ids_inmemory_[sample] = m_inp0;
                input_mask_inmemory_[sample] = m_inp1;
                segment_ids_inmemory_[sample] = m_inp2;
            } else if (this->settings_.scenario == mlperf::TestScenario::MultiStream) {
                input_ids_inmemory_[i] = m_inp0;
                input_mask_inmemory_[i] = m_inp1;
                segment_ids_inmemory_[i] = m_inp2;
                sample_list_inmemory_[i] = sample;
            }
        }
    }

    void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples) override {
        this->segment_ids_inmemory_.clear();
        this->input_ids_inmemory_.clear();
        this->input_mask_inmemory_.clear();
    }

    void GetSample(const std::vector<mlperf::QuerySampleIndex> samples,
            std::vector<mlperf::ResponseId> query_ids, size_t bs, Item *item) {
         std::vector<ov::Tensor> input;
         input.push_back(input_ids_inmemory_[samples[0]]);
         input.push_back(input_mask_inmemory_[samples[0]]);
         input.push_back(segment_ids_inmemory_[samples[0]]);
             *item = Item(input, query_ids, samples);
     }

    void GetSamplesBatched(const std::vector<mlperf::QuerySampleIndex> samples,
            std::vector<mlperf::ResponseId> query_ids, size_t bs, int num_batches, std::vector<Item> &items) {
        ov::Shape shape{bs, max_seq_length_};

        for (int i = 0; i < num_batches; ++i) {
            auto start = (i * bs) % perf_count_;

            ov::Tensor input_ids = ov::Tensor(ov::element::i32, shape, input_ids_inmemory_[start].data<int32_t>());
            ov::Tensor input_mask = ov::Tensor(ov::element::i32, shape, input_mask_inmemory_[start].data<int32_t>());
            ov::Tensor segment_ids = ov::Tensor(ov::element::i32, shape, segment_ids_inmemory_[start].data<int32_t>());

            std::vector<ov::Tensor> inputs {input_ids, input_mask, segment_ids};

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
            throw std::runtime_error("GetSamplesBatchedServer is not implemented!");
    }
    void GetSamplesBatchedMultistream(const std::vector<mlperf::QuerySampleIndex> samples,
        std::vector<mlperf::ResponseId> query_ids, size_t bs, int num_batches, std::vector<Item> &items) {
        ov::Shape shape{bs, max_seq_length_};

        std::vector<mlperf::QuerySampleIndex>::iterator it = std::find(
                sample_list_inmemory_.begin(), sample_list_inmemory_.end(),
                samples[0]);
        if (!(it != std::end(sample_list_inmemory_))) {
            std::cout << "GetSamplesBatchedMultistream: ERROR: No sample found\n ";
            throw std::logic_error("Aborted");
        }

        int index = std::distance(sample_list_inmemory_.begin(), it);
        auto start = index;
        auto query_start = 0;
        for (int i = 0; i < num_batches; ++i) {
            ov::Tensor input_ids = ov::Tensor(ov::element::i32, shape, input_ids_inmemory_[start].data<int32_t>());
            ov::Tensor input_mask = ov::Tensor(ov::element::i32, shape, input_mask_inmemory_[start].data<int32_t>());
            ov::Tensor segment_ids = ov::Tensor(ov::element::i32, shape, segment_ids_inmemory_[start].data<int32_t>());

            std::vector<ov::Tensor> inputs {input_ids, input_mask, segment_ids};

            std::vector<mlperf::QuerySampleIndex> idxs;
            std::vector<mlperf::ResponseId> ids;
            for (size_t j = (query_start); j < (unsigned) (query_start + bs);
                    ++j) {
                ids.push_back(query_ids[j]);
                idxs.push_back(samples[j]);
            }

            items.push_back(Item(inputs, ids, idxs));

            start = start + bs;
            query_start = query_start + bs;
        }
    }

    const std::string& Name() override {
        static const std::string name("OpenVINO Squad v1.1 QSL");
        return name;
    }

	~Squad(){};
private:
    size_t max_seq_length_ = 384;
    int max_query_length_ = 64;
    int doc_stride_ = 128;
    std::vector<std::vector<int>> squad_input_ids_;
    std::vector<std::vector<int>> squad_input_mask_;
    std::vector<std::vector<int>> squad_segment_ids_;
    std::vector<std::vector<string>> squad_tokens_;
    std::vector<ov::Tensor> input_ids_inmemory_;
    std::vector<ov::Tensor> input_mask_inmemory_;
    std::vector<ov::Tensor> segment_ids_inmemory_;
};