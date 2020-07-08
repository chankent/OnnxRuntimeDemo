project(demolib)

#file(GLOB LIB_SRCS infer_wrapper.*)
set(LIB_SRCS infer_wrapper.cpp)

add_library(infer_wrapper STATIC ${LIB_SRCS})

target_link_libraries(infer_wrapper onnxruntime)
