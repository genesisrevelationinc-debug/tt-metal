#include "tt_metal/kernels/eltwise.hpp"
#include "tt_metal/detail/kernels.hpp"
#include "tt_metal/detail/buffer.hpp"
#include "tt_metal/detail/tensor.hpp"

namespace tt {

    // Convert inputs to tiled format
    Tensor tiled_input1 = convert_to_tiled(input1);
    Tensor tiled_input2 = convert_to_tiled(input2);

    // Perform eltwise operation on tiled inputs
    Tensor tiled_output = perform_eltwise(tiled_input1, tiled_input2, op);

    convert_to_row_major(tiled_output, output);
}

void eltwise_row_major(const Tensor& input1, const Tensor& input2, Tensor& output, EltwiseOp op) {
    // Ensure inputs and output are in row major format
    if (input1.layout() != Layout::ROW_MAJOR || input2.layout() != Layout::ROW_MAJOR || output.layout() != Layout::ROW_MAJOR) {
        throw std::invalid_argument("Inputs and output must be in row major format for native row major eltwise operation.");
    }

    // Get the dimensions of the tensors
    auto [rows, cols] = input1.shape();

    // Launch the kernel for row major eltwise operation
    launch_row_major_eltwise_kernel(input1, input2, output, rows, cols, op);
}

void launch_row_major_eltwise_kernel(const Tensor& input1, const Tensor& input2, Tensor& output, size_t rows, size_t cols, EltwiseOp op) {
    // Define the kernel launch parameters
    KernelLaunchParams params;
    params.grid_dim = {static_cast<uint32_t>(cols), static_cast<uint32_t>(rows)};
    params.block_dim = {1, 1};

    // Define the kernel arguments
    KernelArgs args = {
        {"input1", input1.buffer()},
        {"input2", input2.buffer()},
        {"output", output.buffer()},
        {"rows", static_cast<uint32_t>(rows)},
        {"cols", static_cast<uint32_t>(cols)},
        {"op", static_cast<uint32_t>(op)}
    };

    // Launch the kernel
    launch_kernel("row_major_eltwise_kernel", params, args);
}

} // namespace tt