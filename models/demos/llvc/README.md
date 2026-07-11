# LLVC (Low-Latency Low-Resource Voice Conversion) on Tenstorrent

## Overview

LLVC is a real-time voice conversion model from Koe AI, optimized for low latency and CPU efficiency. This implementation brings LLVC to Tenstorrent hardware (Wormhole/Blackhole) using TTNN APIs, enabling ultra-high-throughput, ultra-low-latency voice conversion for real-time applications.

Key features:
- Ultra-low latency streaming voice conversion
- CPU-optimized architecture adapted for TT hardware
- Supports both streaming and non-streaming modes
- Optional F0-based and F0-free pitch conversion modes
- MIT License (free for commercial use)

## Architecture

The model consists of:
- **Lightweight Encoder**: Optimized convolutional layers with reduced complexity
- **Streaming-capable Decoder**: Causal convolutions with state caching
- **Optional Pitch Extractor**: F0-based and F0-free modes
- **Vocoder Integration**: HiFi-GAN or compatible vocoder

## Requirements

- Tenstorrent hardware (N150/N300 Wormhole or Blackhole)
- tt-metal installed and configured
- Python 3.8+
- PyTorch (for reference comparison)
- Audio processing libraries (librosa, soundfile)

## Setup

1. Install tt-metal following the [installation guide](../../../INSTALLING.md)
2. Install additional dependencies:
