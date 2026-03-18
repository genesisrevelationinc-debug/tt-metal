#include "tt_metal/detail/tensor.hpp"
#include "tt_metal/detail/device.hpp"
#include "tt_metal/detail/buffer.hpp"
#include <numeric>

namespace tt {

        buffer_(create_buffer(device, std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint32_t>()))),
        shape_(shape) {
        strides_.resize(shape.size());
        if (!shape.empty()) {
            strides_.back() = 1;
            for (int i = shape.size() - 2; i >= 0; --i) {
                strides_[i] = strides_[i + 1] * shape_[i + 1];
            }
        }
    }

    Buffer Tensor::buffer() const {
        return buffer_;