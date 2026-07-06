# LLVC (Low-Latency Low-Resource Voice Conversion) on Tenstorrent

This directory contains the TTNN implementation of the LLVC (Low-Latency Low-Resource Voice Conversion) model for Tenstorrent hardware.

## Overview

LLVC is a real-time voice conversion model from Koe AI, optimized for low latency and CPU efficiency. This implementation brings LLVC to Tenstorrent hardware using TTNN APIs for ultra-high-throughput, ultra-low-latency voice conversion.

## Features

- **Streaming and non-streaming modes**: Supports both real-time chunked processing and full-context conversion
- **F0-based and F0-free modes**: Optional pitch-dependent or pitch-independent conversion
- **Optimized for Tenstorrent hardware**: Leverages TTNN fused ops and sharded memory configurations
- **Efficient state management**: Causal convolution state caching for streaming

## Architecture

The LLVC model consists of:

1. **Lightweight Encoder**: Optimized convolutional layers with reduced complexity
2. **Content Encoder**: Extracts content features from source audio
3. **Speaker Encoder**: Extracts speaker embedding from target speaker
4. **Decoder**: Reconstructs audio with target speaker characteristics
5. **Vocoder**: Converts mel-spectrogram to waveform (HiFi-GAN based)

## Setup

### Prerequisites

