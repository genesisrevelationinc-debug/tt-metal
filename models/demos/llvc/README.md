# LLVC (Low-Latency Low-Resource Voice Conversion) on Tenstorrent

This demo implements the LLVC model from [Koe AI](https://github.com/KoeAI/LLVC) using TTNN APIs for ultra-low-latency voice conversion on Tenstorrent hardware.

## Overview

LLVC is a real-time voice conversion model optimized for low latency and CPU efficiency. This implementation brings LLVC to Tenstorrent hardware (Wormhole/Blackhole) for:
- Ultra-high-throughput voice conversion
- Ultra-low-latency streaming
- Batch processing for multiple concurrent streams

## Architecture

The model consists of:
- **Lightweight Encoder**: Optimized convolutional layers with reduced complexity
- **Decoder**: Streaming-capable architecture with causal convolutions
- **Optional Pitch Extractor**: F0-based and F0-free modes
- **Vocoder**: HiFi-GAN based vocoder for waveform generation

## Features

- Streaming mode: Real-time conversion with chunked processing
- Non-streaming mode: Full-context conversion
- F0-based and F0-free voice conversion modes
- Batch processing support
- Optimized memory layout for Tenstorrent hardware

## Requirements

- Tenstorrent N150 or N300 hardware
- tt-metal installed
- Python 3.8+
- PyTorch (for reference comparison)
- librosa, soundfile (for audio processing)

## Quick Start

