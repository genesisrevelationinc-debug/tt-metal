cmake_minimum_required(VERSION 3.14)
project(speecht5_vc)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)

add_executable(speecht5_vc_main speecht5_vc.py)
target_link_libraries(speecht5_vc_main ${TORCH_LIBRARIES})

install(TARGETS speecht5_vc_main DESTINATION bin)