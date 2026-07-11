# LLVC (Low-Latency Low-Resource Voice Conversion) on Tenstorrent

This directory contains the TTNN implementation of LLVC, a real-time voice conversion model from Koe AI optimized for low latency and CPU efficiency.

## Model Overview

LLVC is a voice conversion model designed for:
- **Ultra-low latency**: Real-time voice conversion with minimal delay
- **CPU-optimized**: Efficient enough to run on CPU in real-time
- **Streaming support**: True streaming inference with chunked processing
- **High quality**: Natural voice conversion while maintaining low latency
- **Small model size**: Low resource requirements for edge deployment
- **No F0 dependency**: Optional pitch-independent conversion mode

## Architecture

The model consists of:
- **Lightweight encoder**: Optimized convolutional layers with reduced complexity
- **Streaming-capable decoder**: Causal convolutions with state caching
- **Optional pitch extraction**: F0-based and F0-free modes
- **Vocoder integration**: HiFi-GAN or compatible vocoder

## Setup

### Prerequisites

1. Install tt-metal following the [installation guide](../../../INSTALLING.md)
2. Install additional dependencies:
