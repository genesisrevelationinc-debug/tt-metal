# LLVC (Low-Latency Low-Resource Voice Conversion) on Tenstorrent

This directory contains the TTNN implementation of LLVC, a real-time voice conversion model optimized for low latency and CPU efficiency.

## Overview

LLVC is a voice conversion model from Koe AI that achieves real-time factor (RTF) < 0.1 on CPU. This implementation brings LLVC to Tenstorrent hardware for ultra-high-throughput, ultra-low-latency voice conversion.

### Key Features

- **Ultra-low latency**: Designed for real-time voice conversion with minimal delay
- **Streaming support**: True streaming inference with chunked processing
- **High quality**: Natural voice conversion while maintaining low latency
- **Small model size**: Low resource requirements for edge deployment
- **No F0 dependency**: Optional pitch-independent conversion mode
- **MIT License**: Free for commercial use

## Architecture

The model consists of:
- **Lightweight encoder**: Optimized convolutional layers with reduced complexity
- **Streaming-capable decoder**: Causal convolutions with state caching
- **Optional pitch extraction**: F0-based and F0-free modes
- **Vocoder integration**: HiFi-GAN or compatible vocoder

## Requirements

- Tenstorrent hardware (N150 or N300)
- tt-metal installed and configured
- Python 3.8+
- PyTorch (for reference comparison)
- librosa, soundfile (for audio processing)

## Setup

1. Install tt-metal following the [installation guide](../../../INSTALLING.md)
2. Install additional dependencies:
   