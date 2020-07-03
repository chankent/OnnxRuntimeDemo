cmake_minimum_required(VERSION 3.5)

project(OnnxRuntimeDemo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

# use cuda
add_definitions(-DUSE_CUDA)

find_package(OpenCV)
message(STATUS "OpenCV_INCLUDE_DIRS:" ${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV_LIBRARIES:" ${OpenCV_LIBRARIES})
include_directories(${OpenCV_INCLUDE_DIRS})

set(ORT_ROOT_DIR $ENV{ORT_ROOT_DIR})
message(STATUS "ORT_ROOT_DIR: " ${ORT_ROOT_DIR})
include_directories(${ORT_ROOT_DIR}/include)
link_directories(${ORT_ROOT_DIR}/lib)

include_directories(${${PROJECT_NAME}_SRC_DIR})

file(GLOB LIB_SRCS infer_wrapper.*)
add_library(infer_wrapper STATIC ${LIB_SRCS})

add_executable(main main.cpp)
target_link_libraries(main infer_wrapper onnxruntime ${OpenCV_LIBRARIES})

add_executable(test1 test.cpp)
target_link_libraries(test1 onnxruntime)