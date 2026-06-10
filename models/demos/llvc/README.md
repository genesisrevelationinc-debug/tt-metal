# LLVC (Low-Latency Low-Resource Voice Conversion) on Tenstorrent

This directory contains the TTNN implementation of the LLVC (Low-Latency Low-Resource Voice Conversion) model for Tenstorrent hardware.

## Overview

LLVC is a real-time voice conversion model from Koe AI, optimized for low latency and CPU efficiency. This implementation brings LLVC to Tenstorrent hardware (Wormhole/Blackhole) using TTNN APIs for ultra-high-throughput, ultra-low-latency voice conversion.

## Features

- **Streaming and non-streaming modes**: Real-time chunked processing or full-context conversion
- **F0-based and F0-free modes**: Optional pitch-dependent or pitch-independent conversion
- **Optimized for Tenstorrent hardware**: Leverages TTNN fused ops, sharded memory, and efficient tensor manipulation
- **High performance**: Targets >50 tokens/sec, RTF < 0.3, latency < 100ms for streaming

## Architecture

The model consists of:

1. **Lightweight Encoder**: Optimized convolutional layers with reduced complexity
2. **Content Encoder**: Extracts content features from source audio
3. **Speaker Encoder**: Extracts speaker embedding from target speaker
4. **Decoder**: Reconstructs audio with target speaker characteristics
5. **Vocoder**: Converts mel-spectrogram to waveform (HiFi-GAN based)

## Directory Structure

