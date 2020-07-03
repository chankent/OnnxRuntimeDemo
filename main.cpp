#include "infer_wrapper.h"

#include <chrono>

#include "opencv2/opencv.hpp"

using namespace std::chrono;

void CVMatHWC2CHW(const cv::Mat& cv_img, std::vector<float>* data) {
  int height = cv_img.rows;
  int width = cv_img.cols;
  int channels = cv_img.channels();
  int total = height * width * channels;
  data->resize(total);
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < channels; ++c) {
        int data_index = (c * height + h) * width + w;
        data->at(data_index) = static_cast<float>(ptr[img_index++]);
      }
    }
  }
}

int main(int argc, char** argv) {
  ortdemo::OrtInferWrapper infer_wrapper;

  std::vector<float*> inputs;
  std::vector<std::vector<int64_t>> inputs_dims;
  std::vector<float*> outputs;
  std::vector<std::vector<int64_t>> outputs_dims;

#ifdef PRED
  // size_t size1 = 4 * 1 * 4 * 224 * 224;
  size_t size1 = 224 * 224 * 6;
  inputs.emplace_back(new float[size1]);
  // inputs_dims.emplace_back(std::vector<int64_t>{8, 1, 4, 224, 224});
  inputs_dims.emplace_back(std::vector<int64_t>{224, 224, 12});
  // initialize input data with values in [0.0, 1.0]
  for (size_t i = 0; i < size1; ++i) {
    inputs[0][i] = (float)rand() / RAND_MAX;
  }

  size_t size2 = 4 * 1 * 80;
  inputs.emplace_back(new float[size2]);
  inputs_dims.emplace_back(std::vector<int64_t>{4, 1, 80});
  for (size_t i = 0; i < size2; ++i) {
    inputs[1][i] = (float)rand() / RAND_MAX;
  }
#else
  cv::Mat src_image = cv::imread("../img100.jpg");
  int resize_height = 512;
  int resize_width = 1024;
  cv::Mat image;
  cv::resize(src_image(cv::Rect(0, 120, 1920, 960)), image, cv::Size(resize_width, resize_height));
  std::vector<float> data;
  CVMatHWC2CHW(image, &data);
  inputs.emplace_back(data.data());
  inputs_dims.emplace_back(std::vector<int64_t>{1, image.channels(), image.rows, image.cols});
#endif

  auto start_t = system_clock::now();
  int loops = 200;
  for (int i = 0; i < loops; ++i) {
    infer_wrapper.Infer(inputs, inputs_dims, &outputs, &outputs_dims);
  }
  auto end_t = system_clock::now();
  auto duration_t = duration_cast<microseconds>(end_t - start_t);
  std::cout << "loops=" << loops << " avg_time="
            << double(duration_t.count()) * microseconds::period::num / microseconds::period::den / loops << std::endl;

#ifdef PRED
  for (size_t i = 0; i < inputs.size(); ++i) {
    delete [] inputs[i];
  }
#else
  int out_h = outputs_dims[0][2];
  int out_w = outputs_dims[0][3];
  cv::Mat gray_mat(cv::Size(out_w, out_h), CV_32FC1);
  gray_mat.setTo(cv::Scalar(0));
  memcpy(gray_mat.data, outputs[0] + out_h * out_w, gray_mat.total() * sizeof(float));
  cv::Mat temp;
  gray_mat.convertTo(temp, CV_8UC1, 255.0f);
  cv::imshow("temp gray", temp);
  // cv::Mat show_mat;
  cv::waitKey();
#endif

  for (size_t i = 0; i < outputs.size(); ++i) {
    delete [] outputs[i];
  }

  return 0;
}