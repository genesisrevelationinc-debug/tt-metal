# LLVC (Low-Latency Low-Resource Voice Conversion) on Tenstorrent

This directory contains the TTNN implementation of the [LLVC (Low-Latency Low-Resource Voice Conversion)](https://github.com/KoeAI/LLVC) model for Tenstorrent hardware.

## Overview

LLVC is a real-time voice conversion model optimized for low latency and CPU efficiency. This implementation brings LLVC to Tenstorrent hardware (Wormhole N150/N300 and Blackhole) using TTNN APIs for ultra-high-throughput, ultra-low-latency voice conversion.

### Key Features

- **Ultra-low latency**: Real-time voice conversion with minimal delay
- **Streaming support**: True streaming inference with chunked processing
- **Dual mode**: Supports both streaming and non-streaming modes
- **High quality**: Natural voice conversion with speaker similarity > 70%
- **Efficient**: Optimized for Tenstorrent hardware with sharded memory configs

## Architecture

The LLVC model consists of:

1. **Lightweight Encoder**: Optimized convolutional layers with reduced complexity
2. **Content Encoder**: Extracts content features from source audio
3. **Speaker Encoder**: Extracts speaker embedding from target speaker
4. **Decoder**: Generates converted audio features
5. **Vocoder**: Converts features to waveform (HiFi-GAN based)

### Streaming Architecture

