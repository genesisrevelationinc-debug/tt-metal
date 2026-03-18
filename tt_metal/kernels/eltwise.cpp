#include "tt_metal/kernels/eltwise.hpp"
#include "tt_metal/detail/kernels.hpp"
#include "tt_metal/detail/buffer.hpp"
#include "tt_metal/detail/tensor.hpp"
#include "tt_metal/detail/program.hpp"
#include "tt_metal/detail/device.hpp"
#include "tt_metal/detail/launch.hpp"
    const Buffer& output,
    const std::vector<uint32_t>& shape,
    const std::string& op) {
    // Directly handle row major tensors without conversion
    auto device = input.device();
    auto program = create_program(device, op);
    auto kernel = create_kernel(program, op);
    set_arg(kernel, 2, output.buffer());
    set_arg(kernel, 3, shape.size());

    // Assuming row major layout, calculate strides
    std::vector<uint32_t> strides(shape.size());
    strides.back() = 1;
    for (int i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    set_arg(kernel, 4, strides.data());

    launch(kernel, device, shape);
}

Program create_program(const Device& device, const std::string& op) {
    std::string kernel_code = R"(
__kernel void eltwise(__global const float* input1, __global const float* input2, __global float* output, __global const uint32_t* shape, __global const uint32_t* strides, uint32_t dim) {
    uint32_t idx = get_global_id(0);
    uint32_t coords[DIM];
    uint32_t index = idx;
        coords[i] = index / strides[i];
        index %= strides[i];
    }
    uint32_t flat_index = 0;
    for (uint32_t i = 0; i < dim; ++i) {
        flat_index = flat_index * shape[i] + coords[i];
    }