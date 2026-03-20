cmake_minimum_required(VERSION 3.14)
project(speecht5_vc)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
find_package(TTNN REQUIRED)

add_executable(speecht5_vc_main speecht5_vc.py)
target_link_libraries(speecht5_vc_main Python3::Interpreter ${TORCH_LIBRARIES} ${TTNN_LIBRARIES})
target_include_directories(speecht5_vc_main PRIVATE ${TORCH_INCLUDE_DIRS} ${TTNN_INCLUDE_DIRS})