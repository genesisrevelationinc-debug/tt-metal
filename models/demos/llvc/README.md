# LLVC (Low-Latency Low-Resource Voice Conversion) on Tenstorrent

This directory contains the TTNN implementation of the LLVC model for ultra-low-latency voice conversion on Tenstorrent hardware.

## Overview

LLVC is a real-time voice conversion model optimized for low latency and CPU efficiency. This implementation brings LLVC to Tenstorrent hardware using TTNN APIs, achieving:

- **Streaming mode**: Real-time conversion with chunked processing
- **Non-streaming mode**: Full-context conversion
- Target performance: 50+ tokens/second, RTF < 0.3, latency < 100ms

## Architecture

The model consists of:
- **Lightweight Encoder**: Optimized convolutional layers with reduced complexity
- **Content Encoder**: Extracts content features from source audio
- **Speaker Encoder**: Extracts speaker identity embeddings
- **Decoder**: Reconstructs audio with target speaker characteristics
- **Vocoder**: Converts mel-spectrograms to waveform (optional, can use external)

## Directory Structure

