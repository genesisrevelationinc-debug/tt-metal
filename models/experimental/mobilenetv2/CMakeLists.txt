cmake_minimum_required(VERSION 3.10)
project(mobilenetv2)

find_package(ttnncpp REQUIRED)

add_executable(mobilenetv2 main.cpp)

target_link_libraries(mobilenetv2 ttnncpp)

include_directories(${CMAKE_SOURCE_DIR}/tt_metal)
include_directories(${CMAKE_SOURCE_DIR}/tt_stl)
include_directories(${CMAKE_SOURCE_DIR}/ttnn)

enable_testing()
add_test(NAME mobilenetv2_test COMMAND mobilenetv2)