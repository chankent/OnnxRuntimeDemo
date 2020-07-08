// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//

#include <assert.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <onnxruntime_cxx_api.h>
#include <cuda_provider_factory.h>

//using namespace std;


int main(int argc, char* argv[]) {
  //*************************************************************************
  // initialize  enviroment...one enviroment per process
  // enviroment maintains thread pools and other state information
  //
  if (argc > 2) {
    printf("Usage: ./main (gpu_id)");
    return -1;
  }

  int gpu_id = -1;
  if (argc == 2) {
    gpu_id = std::atoi(argv[1]);
  }
  clock_t start, end;
  srand(time(0));

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

  // initialize session options if needed
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(16);

  // If onnxruntime.dll is built with CUDA enabled, we can uncomment out this line to use CUDA for this
  // session (we also need to include cuda_provider_factory.h above which defines it)
  // #include "cuda_provider_factory.h"
  if (gpu_id >= 0) {
    OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, gpu_id);
  }

  // Sets graph optimization level
  // Available levels are
  // ORT_DISABLE_ALL -> To disable all optimizations
  // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
  // ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
  // ORT_ENABLE_ALL -> To Enable All possible opitmizations
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  //*************************************************************************
  // create session and load model into memory
  // using squeezenet version 1.3
  // URL = https://github.com/onnx/models/tree/master/squeezenet
#ifdef _WIN32
  const wchar_t* model_path = L"squeezenet.onnx";
#else
  const char* model_path = "../models/ego_cuda.onnx";
#endif

  printf("Using Onnxruntime C++ API\n");
  Ort::Session session(env, model_path, session_options);

  //*************************************************************************
  // print model input layer (node names, types, shape etc.)
  Ort::AllocatorWithDefaultOptions allocator;

  // print number of model input nodes
  size_t num_input_nodes = session.GetInputCount();
  std::vector<const char*> input_node_names(num_input_nodes);
  std::vector<std::vector<int64_t>> input_node_dims(num_input_nodes);  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                         // Otherwise need vector<vector<>>

  printf("Number of inputs = %zu\n", num_input_nodes);

  // iterate over all input nodes
  for (int i = 0; i < num_input_nodes; i++) {
    // print input node names
    char* input_name = session.GetInputName(i, allocator);
    printf("Input %d : name=%s\n", i, input_name);
    input_node_names[i] = input_name;

    // print input node types
    Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf("Input %d : type=%d\n", i, type);

    // print input shapes/dims
    input_node_dims[i] = tensor_info.GetShape();
    printf("Input %d : num_dims=%zu\n", i, input_node_dims[i].size());
    for (int j = 0; j < input_node_dims[i].size(); j++)
      printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[i][j]);
  }

  // Results should be...
  // Number of inputs = 1
  // Input 0 : name = data_0
  // Input 0 : type = 1
  // Input 0 : num_dims = 4
  // Input 0 : dim 0 = 1
  // Input 0 : dim 1 = 3
  // Input 0 : dim 2 = 224
  // Input 0 : dim 3 = 224

  //*************************************************************************
  // Similar operations to get output node information.
  // Use OrtSessionGetOutputCount(), OrtSessionGetOutputName()
  // OrtSessionGetOutputTypeInfo() as shown above.

  //*************************************************************************
  // Score the model using sample data, and inspect values

  size_t input_semantic_map_size = 4 * 1 * 4 * 224 * 224;  // simplify ... using known dim values to calculate size
                                             // use OrtGetTensorShapeElementCount() to get official size!
  size_t input_state_size = 4 * 1 * 80;

  std::vector<float> input_semantic_map_values(input_semantic_map_size);
  std::vector<const char*> output_node_names = {"pred_trajs"};

  std::vector<float> input_state_values(input_semantic_map_size);

  // initialize input data with values in [0.0, 1.0]
  for (unsigned int i = 0; i < input_semantic_map_size; i++)
    input_semantic_map_values[i] = ((float)rand()) / RAND_MAX;

  for (unsigned int i = 0; i < input_state_size; ++i) {
    input_state_values[i] =  ((float)rand()) / RAND_MAX;
  }

  // Make input
  std::vector<Ort::Value> ort_inputs;

  // create input tensor object from data values
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(memory_info, 
              input_semantic_map_values.data(), input_semantic_map_size, 
              input_node_dims[0].data(), input_node_dims[0].size()));
  //assert(input_semantic_map_tensor.IsTensor());
  ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(memory_info, 
              input_state_values.data(), input_state_size, 
              input_node_dims[1].data(), input_node_dims[1].size()));
  //assert(input_state_tensor.IsTensor());

  // score model & input tensor, get back output tensor 
  start = clock();
  std::vector<Ort::Value> output_tensors;
  for (int i = 0; i < 200; ++i) {
    output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), 
          ort_inputs.data(), ort_inputs.size(), output_node_names.data(), 1);
  }
  end = clock();
  if (gpu_id >= 0) {
    printf("GPU time cost: %0.5f\n", (double)(end-start)/CLOCKS_PER_SEC/200);
  } else {
    printf("CPU time cost: %0.5f\n", (double)(end-start)/CLOCKS_PER_SEC/200);
  }
  assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

  // Get pointer to output tensor float values
  float* floatarr = output_tensors.front().GetTensorMutableData<float>();

  // score the model, and print scores for first 5 classes
  for (int i = 90; i < 120; i++)
    printf("x-y: %0.3f %0.3f\n", floatarr[2*i], floatarr[2*i+1]);

  // Results should be as below...
  // Score for class[0] = 0.000045
  // Score for class[1] = 0.003846
  // Score for class[2] = 0.000125
  // Score for class[3] = 0.001180
  // Score for class[4] = 0.001317
  printf("Done!\n");
  return 0;
}
