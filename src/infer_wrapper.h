/*******************************************************************************
* Distributed under the MIT license. https://opensource.org/licenses/MIT
* @ File    infer_wrapper.h
* @ Author  chankent (chensjustc@gmail.com)
* @ Create  30/06/2020
* @ LastEditor    chankent (chensjustc@gmail.com)
* @ LastEditTime  
* @ Brief
*******************************************************************************/

#pragma once

#include <vector>

#include "onnxruntime_cxx_api.h"
#include "cuda_provider_factory.h"

// #define DYNAMIC 1

namespace ortdemo {

class OrtInferWrapper {
 public:
  OrtInferWrapper();
  ~OrtInferWrapper();

  bool Infer(const std::vector<float*> inputs, 
             const std::vector<std::vector<int64_t>> inputs_dims,
             std::vector<float*>* outputs,
             std::vector<std::vector<int64_t>>* outputs_dims);

 private:
  bool Init();

 private:
  std::unique_ptr<Ort::Session> session_{nullptr};
  std::unique_ptr<Ort::Env> env_{nullptr};
  // Ort::AllocatorWithDefaultOptions ort_allocator_;

  size_t gpu_id_;

  std::vector<float*> input_data_;
  std::vector<float*> output_data_;

  std::vector<char*> input_tensor_names_;
  std::vector<char*> output_tensor_names_;
  std::vector<std::vector<int64_t>> input_tensor_dims_;
  std::vector<std::vector<int64_t>> output_tensor_dims_;
  std::vector<int64_t> input_tensor_sizes_;
  std::vector<int64_t> output_tensor_sizes_;
}; // OrtInferWrapper

}  // namespace ortdemo
