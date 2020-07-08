/*******************************************************************************
* Distributed under the MIT license. https://opensource.org/licenses/MIT
* @ File    infer_wrapper.cpp
* @ Author  chankent (chensjustc@gmail.com)
* @ Create  30/06/2020
* @ LastEditor    chankent (chensjustc@gmail.com)
* @ LastEditTime  
* @ Brief
*******************************************************************************/

#include "infer_wrapper.h"

#include <string>
#include <iostream>
#include <numeric>
#include <chrono>
using namespace std::chrono;

namespace ortdemo {
  
OrtInferWrapper::OrtInferWrapper() {
  Init();
}

OrtInferWrapper::~OrtInferWrapper() {

}

bool OrtInferWrapper::Init() {
#ifdef DYNAMIC
  std::string model_path("../models/dynamic_semantic_map_evaluator.onnx");
#else
  std::string model_path("../models/image_fixed.onnx");
#endif
  size_t gpu_id = 0;
  size_t thread_num = 16;

  // init session
  env_.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test"));
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(thread_num);
  if (gpu_id >= 0) {
    auto status =
        OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, gpu_id);
  }
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  session_.reset(new Ort::Session(*env_, model_path.c_str(), session_options));

  // init model info
  int num_inputs = static_cast<int>(session_->GetInputCount());
  // std::vector<char*> input_names;
  std::vector<std::vector<int64_t>> input_dims;
  std::vector<int64_t> input_sizes; 
  input_tensor_names_.reserve(num_inputs);
  input_dims.reserve(num_inputs);
  input_sizes.reserve(num_inputs);

  Ort::AllocatorWithDefaultOptions allocator;
  std::cout << "input number=" << num_inputs << std::endl;
  for (int i = 0; i < num_inputs; ++i) {
    char* input_name = session_->GetInputName(i, allocator);
    std::cout << "id=" << i << " name=" << input_name;
    input_tensor_names_.emplace_back(strdup(input_name));
    allocator.Free(input_name);

    Ort::TypeInfo type_info = session_->GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    auto input_shape = tensor_info.GetShape();
    input_dims.emplace_back(input_shape);
    input_sizes.emplace_back(
        std::accumulate(input_shape.begin(), input_shape.end(), 1,
                        std::multiplies<int64_t>()));

    std::cout << " type=" << type << " dims=" << input_shape.size() << std::endl;
    for (int j = 0; j < static_cast<int>(input_shape.size()); ++j) {
      std::cout << " dim=" << j << " size=" << input_shape[j] << std::endl;
    }
  }

  int num_outputs = static_cast<int>(session_->GetOutputCount());
  // std::vector<char*> input_names;
  std::vector<std::vector<int64_t>> output_dims;
  std::vector<int64_t> output_sizes; 
  output_tensor_names_.reserve(num_outputs);
  output_dims.reserve(num_outputs);
  output_sizes.reserve(num_outputs);

  std::cout << "output number=" << num_outputs << std::endl;
  for (int i = 0; i < num_outputs; ++i) {
    char* output_name = session_->GetOutputName(i, allocator);
    std::cout << "id=" << i << " name=" << output_name;
    output_tensor_names_.emplace_back(strdup(output_name));
    allocator.Free(output_name);

    Ort::TypeInfo type_info = session_->GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    auto output_shape = tensor_info.GetShape();
    output_dims.emplace_back(output_shape);
    output_sizes.emplace_back(
        std::accumulate(output_shape.begin(), output_shape.end(), 1,
                        std::multiplies<int64_t>()));

    std::cout << " type=" << type << " dims=" << output_shape.size() << std::endl;
    for (int j = 0; j < static_cast<int>(output_shape.size()); ++j) {
      std::cout << " dim=" << j << " size=" << output_shape[j] << std::endl;
    }
  }

  return true;
}

static size_t g_loop_count = 0;
static double g_total_time = 0.0;

bool OrtInferWrapper::Infer(const std::vector<float*> inputs, 
                            const std::vector<std::vector<int64_t>> inputs_dims,
                            std::vector<float*>* outputs,
                            std::vector<std::vector<int64_t>>* outputs_dims) {
  // TODO, 是否可以分配一个max的空间，将数据进行拷贝给member就行？
  // many checks
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<Ort::Value> ort_inputs;

  // compute size
  std::vector<size_t> inputs_sizes;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto dims = inputs_dims[i];
    auto size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());
    ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(memory_info,
        inputs[i], static_cast<size_t>(size), inputs_dims[i].data(), inputs_dims[i].size()));
  }

  auto start_t = system_clock::now();
  std::vector<Ort::Value> ort_outputs =
      session_->Run(Ort::RunOptions{nullptr}, input_tensor_names_.data(),
                    ort_inputs.data(), ort_inputs.size(), output_tensor_names_.data(),
                    output_tensor_names_.size());
  auto end_t = system_clock::now();
  auto duration_t = duration_cast<microseconds>(end_t - start_t);
  g_loop_count++;
  // skip the first iter, which is always slow to warmup the device
  if (g_loop_count > 1) {
    g_total_time += double(duration_t.count()) * microseconds::period::num / microseconds::period::den;
    std::cout << "cur loop=" << g_loop_count << " avg_time="
              << g_total_time / g_loop_count << std::endl;
  }

  int num_outputs = static_cast<int>(session_->GetOutputCount());
  outputs->resize(num_outputs);
  outputs_dims->resize(num_outputs);
  Ort::AllocatorWithDefaultOptions allocator;
  for (int i = 0; i < num_outputs; ++i) {
    char* output_name = session_->GetOutputName(i, allocator);
    allocator.Free(output_name);

    Ort::TypeInfo type_info = session_->GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    auto output_shape = tensor_info.GetShape();
    auto size = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int64_t>());

    // std::cout << " type=" << type << " dims=" << output_shape.size() << std::endl;
    // for (int j = 0; j < static_cast<int>(output_shape.size()); ++j) {
    //   std::cout << " dim=" << j << " size=" << output_shape[j] << std::endl;
    // }

    // copy info
    outputs->at(i) = new float[size];
    memcpy(outputs->at(i), ort_outputs[i].GetTensorMutableData<float>(),
           size * sizeof(float));
    outputs_dims->at(i) = output_shape;
  }

  return true;
}

}  // namespace ortdemo
