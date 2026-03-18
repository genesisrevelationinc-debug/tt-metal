#ifndef TT_METAL_DETAIL_TENSOR_HPP
#define TT_METAL_DETAIL_TENSOR_HPP

#include <vector>
#include "tt_metal/detail/buffer.hpp"

namespace tt {
    class Tensor {
    public:
        Tensor(const Device& device, const std::vector<uint32_t>& shape);
        std::vector<uint32_t> strides() const;

        Buffer buffer() const;
        std::vector<uint32_t> shape() const;
        Buffer buffer_;
        std::vector<uint32_t> shape_;

        std::vector<uint32_t> strides_;
    public:
        std::vector<uint32_t> strides() const {
            return strides_;
        }

    };

}