#ifndef TT_METAL_KERNELS_ELTWISE_HPP
#define TT_METAL_KERNELS_ELTWISE_HPP

#include <vector>
#include "tt_metal/detail/buffer.hpp"

namespace tt {
    void eltwise(
        const Buffer& input1,
        const Buffer& input2,
        const Buffer& output,
        const std::vector<uint32_t>& shape,
        const std::string& op);
