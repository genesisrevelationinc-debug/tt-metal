#include "tt_metal/kernels/kernel_launch.hpp"
#include "tt_metal/detail/buffer.hpp"
#include "tt_metal/detail/device.hpp"
#include "tt_metal/detail/tensor.hpp"

namespace tt {

    // Launch the kernel with the provided parameters and arguments
    device.launch_kernel(kernel_name, params, args);
}

void launch_row_major_eltwise_kernel(const Tensor& input1, const Tensor& input2, Tensor& output, size_t rows, size_t cols, EltwiseOp op) {
    // Define the kernel launch parameters
    KernelLaunchParams params = {static_cast<uint32_t>(cols), static_cast<uint32_t>(rows), 1, 1};
    launch_kernel("row_major_eltwise_kernel", params, {{"input1", input1.buffer()}, {"input2", input2.buffer()}, {"output", output.buffer()}, {"rows", static_cast<uint32_t>(rows)}, {"cols", static_cast<uint32_t>(cols)}, {"op", static_cast<uint32_t>(op)}});
}

} // namespace tt