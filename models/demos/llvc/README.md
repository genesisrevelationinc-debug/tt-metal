# LLVC (Low-Latency Low-Resource Voice Conversion) on TT-Metal

This directory contains the TT-Metal/TTNN implementation of the LLVC model for ultra-low-latency real-time voice conversion on Tenstorrent hardware.

## Overview

LLVC is a real-time voice conversion model optimized for low latency and CPU efficiency. This implementation brings LLVC to Tenstorrent hardware (Wormhole N150/N300, Blackhole) using TTNN APIs, achieving significant speedups over CPU baseline while maintaining audio quality.

## Key Features

- **Ultra-low latency**: < 50ms per chunk in streaming mode
- **Real-time factor (RTF)**: < 0.1 for streaming, < 0.3 for non-streaming
- **Streaming support**: True chunked processing with causal convolutions
- **High quality**: > 70% speaker similarity, WER < 3.0
- **Concurrent streams**: Support for 10+ simultaneous streams
- **Two modes**: F0-based and F0-free voice conversion

## Architecture

