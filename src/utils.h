// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/openvino.hpp>

#include <string>
#include <algorithm>
#include <thread>
#include <utility>
#include <vector>
#include <map>

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;

    while (getline(ss, item, delim)) {
        result.push_back(item);
    }
    return result;
}

std::vector<std::string> parse_devices(const std::string& device_string) {
    std::string comma_separated_devices = device_string;
    if (comma_separated_devices.find(":") != std::string::npos) {
        comma_separated_devices = comma_separated_devices.substr(comma_separated_devices.find(":") + 1);
    }
    auto devices = split(comma_separated_devices, ',');
    for (auto& device : devices)
        device = device.substr(0, device.find_first_of(".("));
    return devices;
}

std::map<std::string, std::string> parse_value_per_device(const std::vector<std::string>& devices,
                                                          const std::string& values_string) {
    //  Format: <device1>:<value1>,<device2>:<value2> or just <value>
    std::map<std::string, std::string> result;
    auto device_value_strings = split(values_string, ',');
    for (auto& device_value_string : device_value_strings) {
        auto device_value_vec = split(device_value_string, ':');
        if (device_value_vec.size() == 2) {
            auto device_name = device_value_vec.at(0);
            auto value = device_value_vec.at(1);
            auto it = std::find(devices.begin(), devices.end(), device_name);
            if (it != devices.end()) {
                result[device_name] = value;
            } else {
                std::string devices_list = "";
                for (auto& device : devices)
                    devices_list += device + " ";
                devices_list.pop_back();
                throw std::logic_error("Failed to set property to '" + device_name +
                                       "' which is not found whthin the target devices list '" + devices_list + "'!");
            }
        } else if (device_value_vec.size() == 1) {
            auto value = device_value_vec.at(0);
            for (auto& device : devices) {
                result[device] = value;
            }
        } else if (device_value_vec.size() != 0) {
            throw std::runtime_error("Unknown string format: " + values_string);
        }
    }
    return result;
}

template <class T>
void TopResults(unsigned int n, const ov::Tensor& input, std::vector<unsigned>& output) {
        ov::Shape shape = input.get_shape();
        size_t input_rank = shape.size();
        OPENVINO_ASSERT(input_rank != 0 && shape[0] != 0, "Input tensor has incorrect dimensions!");
        size_t batchSize = shape[0];
        std::vector<unsigned> indexes(input.get_size() / batchSize);

        n = static_cast<unsigned>(std::min<size_t>((size_t)n, input.get_size()));

        output.resize(n * batchSize);

        for (size_t i = 0; i < batchSize; i++) {
            size_t offset = i * (input.get_size() / batchSize);
            const T* batchData = input.data<const T>();
            batchData += offset;

            std::iota(std::begin(indexes), std::end(indexes), 0);
            std::partial_sort(std::begin(indexes), std::begin(indexes) + n, std::end(indexes),
                              [&batchData](unsigned l, unsigned r) {
                                  return batchData[l] > batchData[r];
                              });
            for (unsigned j = 0; j < n; j++) {
                output.at(i * n + j) = indexes.at(j);
            }
        }
    }


    /**
     * @brief Gets the top n results from a blob
     *
     * @param n Top n count
     * @param input 1D blob that contains probabilities
     * @param output Vector of indexes for the top n places
     */

    void TopResults(unsigned int n, const ov::Tensor& input, std::vector<unsigned>& output) {
    #define TENSOR_TOP_RESULT(elem_type)                                                                           \
        case ov::element::Type_t::elem_type: {                                                                     \
            using tensor_type = ov::fundamental_type_for<ov::element::Type_t::elem_type>;                          \
            TopResults<tensor_type>(n, input, output);                                                             \
            break;                                                                                                 \
        }

        switch (input.get_element_type()) {
            TENSOR_TOP_RESULT(f32);
            TENSOR_TOP_RESULT(f64);
            TENSOR_TOP_RESULT(f16);
            TENSOR_TOP_RESULT(i16);
            TENSOR_TOP_RESULT(u8);
            TENSOR_TOP_RESULT(i8);
            TENSOR_TOP_RESULT(u16);
            TENSOR_TOP_RESULT(i32);
            TENSOR_TOP_RESULT(u32);
            TENSOR_TOP_RESULT(i64);
        default:
            OPENVINO_ASSERT(false, "cannot locate tensor with element type: ", input.get_element_type());
        }
    #undef TENSOR_TOP_RESULT
}


