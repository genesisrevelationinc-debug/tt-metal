# LLVC (Low-Latency Low-Resource Voice Conversion) on Tenstorrent

This demo brings up the [LLVC](https://github.com/KoeAI/LLVC) real-time voice conversion model using TTNN APIs on Tenstorrent hardware.

## Overview

LLVC is a real-time voice conversion model optimized for low latency and CPU efficiency. This implementation enables it to run on Tenstorrent Wormhole/Blackhole hardware for ultra-high-throughput, ultra-low-latency voice conversion.

## Features

- **Streaming mode**: Real-time conversion with chunked processing (< 100ms latency)
- **Non-streaming mode**: Full-context conversion
- **F0-based and F0-free modes**: Optional pitch-dependent or pitch-independent conversion
- **Optimized for Tenstorrent hardware**: Uses sharded memory layouts, fused ops, and efficient tensor manipulation

## Architecture

The model consists of:
- **Lightweight Encoder**: Optimized convolutional layers with reduced complexity
- **Content Encoder**: Extracts content features from source audio
- **Decoder**: Generates converted audio features
- **Vocoder**: Converts features to waveform (HiFi-GAN based)

## Setup

