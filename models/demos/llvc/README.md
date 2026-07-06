# LLVC (Low-Latency Low-Resource Voice Conversion) on Tenstorrent

This directory contains the TTNN implementation of the LLVC model for real-time voice conversion on Tenstorrent hardware.

## Overview

LLVC is a real-time voice conversion model optimized for low latency and CPU efficiency. This implementation brings LLVC to Tenstorrent hardware using TTNN APIs for ultra-high-throughput, ultra-low-latency voice conversion.

## Features

- **Streaming mode**: Real-time conversion with chunked processing
- **Non-streaming mode**: Full-context conversion
- **F0-based and F0-free modes**: Optional pitch-dependent or pitch-independent conversion
- **Optimized for Tenstorrent hardware**: Uses sharded memory configs, fused ops, and efficient tensor manipulation

## Architecture

The model consists of:
- **Lightweight Encoder**: Optimized convolutional layers with reduced complexity
- **Content Encoder**: Extracts content features from source audio
- **Speaker Encoder**: Extracts speaker embedding from target speaker
- **Decoder**: Generates converted audio features
- **Vocoder**: Converts features to waveform (HiFi-GAN based)

## Setup

