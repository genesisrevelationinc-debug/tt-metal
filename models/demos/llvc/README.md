# LLVC (Low-Latency Low-Resource Voice Conversion) on Tenstorrent

This directory contains the TTNN implementation of the LLVC (Low-Latency Low-Resource Voice Conversion) model for Tenstorrent hardware.

## Overview

LLVC is a real-time voice conversion model optimized for low latency and CPU efficiency. This implementation brings LLVC to Tenstorrent hardware using TTNN APIs for ultra-high-throughput, ultra-low-latency voice conversion.

## Features

- **Streaming and non-streaming modes**: Supports both real-time chunked processing and full-context conversion
- **F0-based and F0-free modes**: Optional pitch-dependent or pitch-independent conversion
- **Optimized for Tenstorrent hardware**: Leverages TTNN fused ops, sharded memory layouts, and efficient tensor manipulation

## Architecture

The LLVC model consists of:

1. **Lightweight Encoder**: Optimized convolutional layers with reduced complexity
2. **Content Encoder**: Extracts content features from source audio
3. **Speaker Encoder**: Encodes target speaker characteristics
4. **Decoder**: Reconstructs converted audio with target speaker voice
5. **Vocoder**: Converts mel spectrograms to waveform (HiFi-GAN based)

## Setup

### Prerequisites

