cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

# use cuda
add_definitions(-DUSE_CUDA)

set(ORT_ROOT_DIR $ENV{ORT_ROOT_DIR})
message($ENV{ORT_ROOT_DIR})
message(STATUS "ORT_ROOT_DIR: " ${ORT_ROOT_DIR})
include_directories(${ORT_ROOT_DIR}/include)
link_directories(${ORT_ROOT_DIR}/lib)

add_executable(main main.cpp)
target_link_libraries(main onnxruntime)