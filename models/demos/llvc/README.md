# LLVC (Low-Latency Low-Resource Voice Conversion) on Tenstorrent

## Overview

LLVC is a real-time voice conversion model from Koe AI, optimized for low latency and CPU efficiency. This implementation brings LLVC to Tenstorrent hardware (Wormhole/Blackhole) using TTNN APIs, enabling ultra-high-throughput, ultra-low-latency voice conversion.

Key features:
- Ultra-low latency streaming voice conversion
- CPU-optimized architecture adapted for TT hardware
- Supports both streaming and non-streaming modes
- Optional F0-based and F0-free pitch extraction modes
- MIT License (free for commercial use)

## Architecture

The model consists of:
- **Lightweight Encoder**: Optimized convolutional layers with reduced complexity
- **Decoder**: Streaming-capable architecture with causal convolutions
- **Pitch Extractor** (optional): F0-based or F0-free modes
- **Vocoder**: HiFi-GAN based vocoder for waveform generation

## Requirements

- Tenstorrent Wormhole (N150/N300) or Blackhole hardware
- tt-metal installed and configured
- Python 3.8+
- PyTorch (for reference comparison)
- librosa, soundfile (for audio processing)

## Setup

1. Install tt-metal following the [installation guide](../../../INSTALLING.md)
2. Install additional dependencies:
