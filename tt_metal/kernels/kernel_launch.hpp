#pragma once

#include "tt_metal/detail/buffer.hpp"
#include "tt_metal/detail/tensor.hpp"

namespace tt {

    uint32_t block_dim_y;
};

void launch_kernel(const std::string& kernel_name, const KernelLaunchParams& params, const KernelArgs& args);
void launch_row_major_eltwise_kernel(const Tensor& input1, const Tensor& input2, Tensor& output, size_t rows, size_t cols, EltwiseOp op);

} // namespace tt