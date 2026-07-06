# LLVC (Low-Latency Low-Resource Voice Conversion) on Tenstorrent

This directory contains the TTNN implementation of the LLVC (Low-Latency Low-Resource Voice Conversion) model for Tenstorrent hardware.

## Overview

LLVC is a real-time voice conversion model optimized for low latency and CPU efficiency. This implementation brings LLVC to Tenstorrent hardware using TTNN APIs for ultra-high-throughput, ultra-low-latency voice conversion.

## Features

- **Streaming and non-streaming modes**: Supports both real-time chunked processing and full-context conversion
- **F0-based and F0-free modes**: Optional pitch-dependent or pitch-independent conversion
- **Optimized for Tenstorrent hardware**: Leverages TTNN fused ops, sharded memory configs, and efficient tensor manipulations
- **State caching**: Efficient causal convolution state management for streaming

## Architecture

The LLVC model consists of:

1. **Lightweight Encoder**: Optimized convolutional layers with reduced complexity
2. **Content Encoder**: Extracts content features from source audio
3. **Speaker Encoder**: Extracts speaker embedding from target speaker reference
4. **Decoder**: Generates converted audio features
5. **Vocoder**: Converts features to final audio waveform

## Requirements

- Tenstorrent hardware (N150 or N300)
- tt-metal built with TTNN support
- Python 3.8+
- See `requirements.txt` for Python dependencies

## Installation

