#pragma once

#include <queue>
#include <condition_variable>
#include <mutex>

#include <openvino/openvino.hpp>
#include <ie_blob.h>

// loadgen
#include "loadgen.h"
#include "query_sample.h"
#include "query_sample_library.h"
#include "test_settings.h"
#include "system_under_test.h"
#include "bindings/c_api.h"

#include "item_ov.h"

extern std::unique_ptr<QSLBase> ds;

/// Post processor function type
typedef std::function<
        void(Item qitem,
            ov::InferRequest req,
            std::vector<float> &results,
            std::vector<mlperf::ResponseId>&response_ids,
            unsigned batch_size,
            std::vector<unsigned> &counts)
    > PPFunction;

typedef std::function<
        void(size_t id,
            ov::InferRequest req,
            Item input,
            std::vector<float> &results,
            std::vector<mlperf::ResponseId> &response_ids,
            std::vector<unsigned> &counts,
            const std::exception_ptr& ptr)
	> QueueCallbackFunction;

typedef std::function<
        void(size_t id,
            ov::InferRequest req,
            Item input,
            std::vector<float> &results,
            std::vector<mlperf::ResponseId> &response_ids,
            std::vector<unsigned> &counts,
            std::vector<mlperf::QuerySampleResponse> &respns,
            const std::exception_ptr& ptr)
        > ServerQueueCallbackFunction;

/// @brief Wrapper class for ov::InferRequest. Handles asynchronous callbacks .
class InferReqWrap final {
public:
    using Ptr = std::shared_ptr<InferReqWrap>;

    ~InferReqWrap() = default;

    InferReqWrap(ov::CompiledModel& model,
                 size_t id,
                 std::vector<std::string> input_blob_names,
                 std::vector<std::string> output_blob_names,
                 mlperf::TestSettings settings,
                 std::string workload,
                 QueueCallbackFunction callback_queue) :
            request_(model.create_infer_request()),
            id_(id),
            input_blob_names_(input_blob_names),
            output_blob_names_(output_blob_names),
            settings_(settings),
            workload_(workload),
            callback_queue_(callback_queue) {
            request_.set_callback([&](const std::exception_ptr& ptr) {
                callback_queue_(id_, request_, input_, results_, response_ids_, counts_, ptr);
        });
    }

    InferReqWrap(ov::CompiledModel& model,
                 size_t id,
                 std::vector<std::string> input_blob_names,
                 std::vector<std::string> output_blob_names,
                 mlperf::TestSettings settings,
                 std::string workload,
                 ServerQueueCallbackFunction callback_queue,
                 bool server) :
            request_(model.create_infer_request()),
            id_(id),
            input_blob_names_(input_blob_names),
            output_blob_names_(output_blob_names),
            settings_(settings),
            workload_(workload),
            callback_queue_server_(callback_queue) {
        request_.set_callback([&](const std::exception_ptr& ptr) {
            callback_queue_server_(id_, request_, input_, results_, response_ids_, counts_, respns_, ptr);
        });
    }

    void start_async() {
        request_.start_async();
    }

    void infer() {
        request_.infer();
    }

    ov::Tensor get_tensor(const std::string &name) {
        return request_.get_tensor(name);
    }

    Item get_output_tensors() {
        Item outputs;
	    for (size_t i = 0; i < output_blob_names_.size(); ++i) {
            outputs.tensors_.push_back( request_.get_tensor(output_blob_names_[i]));
	    }
	    return outputs;
    }

    void set_inputs(Item input, std::string name) {
        input_ = input;
        request_.set_tensor(name, input_.tensors_[0]);
    }

    void set_inputs(Item input) {
        input_ = input;
        for (size_t i= 0; i < input_blob_names_.size(); ++i){
            request_.set_tensor(input_blob_names_[i], input_.tensors_[i]);
        }
    }

    unsigned get_batch_size(){
        return input_.sample_idxs_.size();
    }

    void set_is_warmup(bool warmup) {
        is_warmup = warmup;
    }

    size_t get_request_id(){
	    return id_;
    }

    void reset() {
        response_ids_.clear();
        counts_.clear();
        results_.clear();
        respns_.clear();
    }

public:
    std::vector<mlperf::ResponseId> response_ids_;
    std::vector<unsigned> counts_;
    std::vector<float> results_;
    std::vector<mlperf::QuerySampleResponse> respns_;

private:
    ov::InferRequest request_;
    size_t id_;
    std::vector<std::string> input_blob_names_, output_blob_names_;
    mlperf::TestSettings settings_;
    std::string workload_;
    QueueCallbackFunction callback_queue_;
    ServerQueueCallbackFunction callback_queue_server_;

    Item input_;
    Item outputs_;
    bool is_warmup = false;
};

class InferRequestsQueue final {
public:
    InferRequestsQueue(ov::CompiledModel& model,
                       size_t nireq,
                       std::vector<std::string> input_blob_names,
                       std::vector<std::string> output_blob_names,
                       mlperf::TestSettings settings,
                       std::string workload,
                       unsigned batch_size,
                       PPFunction post_processor) :
            num_batches_(),
            settings_(settings),
            workload_(workload),
            batch_size_(batch_size),
            post_processor_(post_processor) {
        for (size_t id = 0; id < nireq; id++) {
            requests.push_back(
                    std::make_shared<InferReqWrap>(model, id, input_blob_names, output_blob_names, settings, workload,
                            std::bind(
                                    &InferRequestsQueue::put_idle_request, this,
                                    std::placeholders::_1,
                                    std::placeholders::_2,
				                    std::placeholders::_3,
				                    std::placeholders::_4,
				                    std::placeholders::_5,
				                    std::placeholders::_6,
                                    std::placeholders::_7)));
            idle_ids_.push(id);
        }
    }

    ~InferRequestsQueue() = default;

    void put_idle_request(size_t id,
                          ov::InferRequest req,
                           Item qitem,
                          std::vector<float> &results,
                          std::vector<mlperf::ResponseId> &response_ids,
                          std::vector<unsigned> &counts,
                          const std::exception_ptr& ptr) {
        post_processor_(qitem, req, results, response_ids, batch_size_, counts);
        std::unique_lock<std::mutex> lock(mutex_);
        if (ptr) {
            inference_exception_ = ptr;
        } else {
            idle_ids_.push(id);
        }
        cv_.notify_one();
    }

    InferReqWrap::Ptr get_idle_request() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] {
            if (inference_exception_) {
                std::rethrow_exception(inference_exception_);
            }
            return idle_ids_.size() > 0;
        });
        auto request = requests.at(idle_ids_.front());
        idle_ids_.pop();
        return request;
    }

    void wait_all() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] {
            if (inference_exception_) {
                std::rethrow_exception(inference_exception_);
            }
            return idle_ids_.size() == requests.size();
        });

	    size_t j = 0;
	    for (auto &req : requests ){
            size_t idx = 0;
            for (size_t i = 0; i < req->counts_.size(); ++i) {
                mlperf::QuerySampleResponse response { req->response_ids_[i],
                    reinterpret_cast<std::uintptr_t>(&(req->results_[idx])),
                    (sizeof(float) * req->counts_[i]) };
                responses_.push_back(response);
                idx = idx + req->counts_[i];
            }
	    }
    }

    std::vector<Item> get_outputs() {
        return outputs_;
    }

    std::vector<mlperf::QuerySampleResponse> get_query_sample_responses(){
        return responses_;
    }

    void reset() {
        outputs_.clear();
        responses_.clear();
        for (auto &req : requests){
            req->reset();
        }
    }

    std::vector<InferReqWrap::Ptr> requests;

private:
    std::queue<size_t> idle_ids_;
    std::mutex mutex_;
    std::condition_variable cv_;

    mlperf::TestSettings settings_;
    std::string out_name_;
    std::string workload_;
    std::vector<Item> outputs_;
    std::vector<mlperf::QuerySampleResponse> responses_;
    unsigned batch_size_, num_batches_;
    PPFunction post_processor_;
    std::exception_ptr inference_exception_ = nullptr;
};



/**================================== Server Queue Runner ================================**/

class InferRequestsQueueServer {
public:
    InferRequestsQueueServer(ov::CompiledModel& model,
                             size_t nireq,
                             std::vector<std::string> input_blob_names,
                             std::vector<std::string> output_blob_names,
                             mlperf::TestSettings settings,
                             std::string workload,
                             unsigned batch_size,
                             PPFunction post_processor) :
            num_batches_(),
            settings_(settings),
            workload_(workload),
            batch_size_(batch_size),
            post_processor_(post_processor),
            is_warmup_(false) {
	    for (size_t id = 0; id < nireq; id++) {
            requests.push_back(
                    std::make_shared<InferReqWrap>(model, id, input_blob_names, output_blob_names, settings, workload,
                                std::bind(
                                    &InferRequestsQueueServer::put_idle_request, this,
                                    std::placeholders::_1,
                                    std::placeholders::_2,
                                    std::placeholders::_3,
                                    std::placeholders::_4,
                                    std::placeholders::_5,
                                    std::placeholders::_6,
				                    std::placeholders::_7,
                                    std::placeholders::_8),
                                    true));
            idle_ids_.push(id);
        }
    }

    ~InferRequestsQueueServer() = default;

    void put_idle_request(size_t id,
                          ov::InferRequest req,
                          Item qitem,
                          std::vector<float> &results,
                          std::vector<mlperf::ResponseId> &response_ids,
                          std::vector<unsigned> &counts,
                          std::vector<mlperf::QuerySampleResponse> &respns,
                          const std::exception_ptr& ptr = nullptr){
        post_processor_(qitem, req, results, response_ids, batch_size_, counts);

        size_t idx = 0;

        for (size_t i = 0; i < counts.size(); ++i) {
            mlperf::QuerySampleResponse response { response_ids[i],
                reinterpret_cast<std::uintptr_t>(&(results[idx])),
                (sizeof(float) * counts[i]) };
            respns.push_back(response);
            idx = idx + counts[i];
        }

        if (!(is_warmup_)){
            mlperf::QuerySamplesComplete( respns.data(), respns.size() );
        }

        results.clear();
        counts.clear();
        response_ids.clear();
        respns.clear();

        std::unique_lock < std::mutex > lock(mutex_);
        if (ptr) {
            inference_exception_ = ptr;
        } else {
            idle_ids_.push(id);
        }
        cv_.notify_one();
    }


    void set_warmup(bool warmup){
	    is_warmup_ = warmup;
    }

    // Maybe different post-completion tasks for warmup
    void warmup_put_idle_req(size_t id){
        std::unique_lock<std::mutex> lock(mutex_);
        idle_ids_.push(id);
        cv_.notify_one();
    }

    InferReqWrap::Ptr get_idle_request() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] {
            if (inference_exception_) {
                std::rethrow_exception(inference_exception_);
            }
            return idle_ids_.size() > 0;
        });
        auto request = requests.at(idle_ids_.front());
        idle_ids_.pop();
        return request;
    }

    void wait_all() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] {
            if (inference_exception_) {
                std::rethrow_exception(inference_exception_);
            }
            return idle_ids_.size() == requests.size();
        });
    }

    std::vector<Item> getOutputs() {
        return outputs_;
    }

    std::vector<mlperf::QuerySampleResponse> get_query_sample_responses(){
        return responses_;
    }

    void reset() {
        outputs_.clear();
        responses_.clear();
        for (auto &req : requests){
            req->reset();
        }
    }

    std::vector<InferReqWrap::Ptr> requests;

private:
    std::queue<size_t> idle_ids_;
    std::mutex mutex_;
    std::condition_variable cv_;

    mlperf::TestSettings settings_;
    std::string out_name_;
    std::string workload_;
    std::vector<Item> outputs_;
    std::vector<mlperf::QuerySampleResponse> responses_;
    unsigned batch_size_, num_batches_;
    PPFunction post_processor_;
    bool is_warmup_;
    std::exception_ptr inference_exception_ = nullptr;
};

