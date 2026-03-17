# MobileNet-v2 via TTNN C++ APIs

This directory contains the implementation of the MobileNet-v2 model using the TTNN C++ APIs.

## Description
MobileNet-v2 is a convolutional neural network architecture that is optimized for mobile and edge devices. This implementation targets the Wormhole (N150 or N300) hardware.

## Model Input Target
- **Input Shape/Resolution**: (224, 224, 3)

## Hardware Target
- **Hardware**: Wormhole (N150 or N300)

## Target FPS
- **Baseline**: ~222 FPS
- **Optimized**: ~333 FPS

## Building and Running
1. Ensure you have the TT-Metal environment set up.
2. Navigate to the `models/experimental/mobilenetv2` directory.
3. Build the model:
   