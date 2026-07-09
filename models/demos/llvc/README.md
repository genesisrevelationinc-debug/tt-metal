# LLVC (Low-Latency Low-Resource Voice Conversion) on Tenstorrent Hardware

## Overview

This directory contains a TTNN-based implementation of the LLVC model from Koe AI, optimized for ultra-low-latency voice conversion on Tenstorrent hardware (Wormhole/Blackhole).

LLVC is a real-time voice conversion model designed for:
- Ultra-low latency (< 100ms streaming chunks)
- CPU-efficient operation (RTF < 0.1 on CPU)
- Streaming and non-streaming modes
- Optional F0-independent conversion
- High-quality natural voice output

## Architecture

The model consists of:
- **Lightweight Encoder**: Optimized convolutional layers with reduced complexity
- **Decoder**: Streaming-capable with cached convolution states
- **Optional Pitch Extractor**: F0-based and F0-free modes
- **Vocoder**: Integration for waveform synthesis

## Requirements

- Tenstorrent hardware (N150/N300 Wormhole or Blackhole)
- tt-metal installed and configured
- Python 3.8+
- PyTorch (for reference comparison)
- librosa, soundfile (for audio processing)

## Setup

