#project(tools)

add_executable(main main.cpp)
target_link_libraries(main infer_wrapper ${OpenCV_LIBRARIES})

add_executable(test1 test.cpp)
target_link_libraries(test1 infer_wrapper ${OpenCV_LIBRARIES})
