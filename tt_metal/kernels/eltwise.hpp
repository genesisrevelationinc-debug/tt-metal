#pragma once

#include "tt_metal/detail/buffer.hpp"
#include "tt_metal/detail/tensor.hpp"

namespace tt {

    DIVIDE,
};

void eltwise(const Tensor& input1, const Tensor& input2, Tensor& output, EltwiseOp op);
void eltwise_row_major(const Tensor& input1, const Tensor& input2, Tensor& output, EltwiseOp op);

} // namespace tt