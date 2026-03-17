#include <iostream>
#include <vector>
#include <ttnn/ttnn.hpp>
#include <tt_metal/tt_metal.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::ttnn;

int main() {
    // Initialize the device
    Device device = CreateDevice();

    // Define the model architecture
    Model model;

    // Input tensor
    Tensor input = model.AddInput({1, 3, 224, 224}, DataType::FLOAT32, Layout::NHWC);

    // First convolutional layer
    Tensor conv1 = model.AddConv2D(input, 32, {3, 3}, {2, 2}, {1, 1}, Activation::RELU);

    // Bottleneck blocks
    std::vector<std::tuple<int, int, int, int>> bottlenecks = {
        {1, 16, 1, 1},
        {6, 24, 2, 2},
        {6, 32, 2, 2},
        {6, 64, 2, 2},
        {6, 96, 1, 1},
        {6, 160, 2, 2},
        {6, 320, 1, 1}
    };

    Tensor prev_tensor = conv1;
    for (const auto& [expansion, out_channels, stride, repeat] : bottlenecks) {
        for (int i = 0; i < repeat; ++i) {
            Tensor expanded = model.AddConv2D(prev_tensor, expansion * prev_tensor.shape()[1], {1, 1}, {1, 1}, {0, 0}, Activation::RELU);
            Tensor depthwise = model.AddDepthwiseConv2D(expanded, {3, 3}, {stride, stride}, {1, 1}, Activation::RELU);
            Tensor pointwise = model.AddConv2D(depthwise, out_channels, {1, 1}, {1, 1}, {0, 0}, Activation::NONE);
            if (i == 0 && stride == 1 && prev_tensor.shape()[1] == out_channels) {
                prev_tensor = model.AddAdd(prev_tensor, pointwise);
            } else {
                prev_tensor = pointwise;
            }
        }
    }

    // Final convolutional layer
    Tensor conv_final = model.AddConv2D(prev_tensor, 1280, {1, 1}, {1, 1}, {0, 0}, Activation::RELU);

    // Global average pooling
    Tensor global_avg_pool = model.AddGlobalAveragePool(conv_final);

    // Output layer
    Tensor output = model.AddConv2D(global_avg_pool, 1000, {1, 1}, {1, 1}, {0, 0}, Activation::NONE);
    model.AddOutput(output);

    // Compile the model
    model.Compile(device);

    // Create input data
    std::vector<float> input_data(input.shape().NumElements(), 0.5f);

    // Run the model
    std::vector<float> output_data(output.shape().NumElements());
    model.Run(input_data, output_data);

    // Print the output
    for (size_t i = 0; i < output_data.size(); ++i) {
        std::cout << "Output[" << i << "] = " << output_data[i] << std::endl;
    }

    return 0;
}