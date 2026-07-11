# LLVC (Low-Latency Low-Resource Voice Conversion) on Tenstorrent

## Overview

This demo implements the LLVC model from [Koe AI](https://github.com/KoeAI/LLVC) using TTNN APIs on Tenstorrent hardware (Wormhole/Blackhole).

LLVC is a real-time voice conversion model optimized for low latency and CPU efficiency. Key features:
- Ultra-low latency streaming voice conversion
- CPU-optimized architecture (RTF < 0.1 on CPU)
- True streaming inference with chunked processing
- Optional F0-independent conversion mode
- MIT License

## Architecture

The model consists of:
- **Lightweight Encoder**: Optimized convolutional layers with reduced complexity
- **Streaming Decoder**: Causal convolutions with state caching for chunked processing
- **Optional Pitch Extractor**: F0-based and F0-free modes
- **Vocoder Integration**: HiFi-GAN or compatible vocoder

## Requirements

- Tenstorrent hardware (N150/N300 Wormhole or Blackhole)
- tt-metal installed and configured
- Python 3.8+
- PyTorch (for reference comparison)
- librosa, soundfile (for audio processing)

## Setup

