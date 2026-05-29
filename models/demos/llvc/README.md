# LLVC (Low-Latency Low-Resource Voice Conversion) on Tenstorrent

This directory contains the TTNN implementation of the LLVC (Low-Latency Low-Resource Voice Conversion) model for Tenstorrent hardware.

## Overview

LLVC is a real-time voice conversion model optimized for low latency and CPU efficiency. This implementation brings LLVC to Tenstorrent hardware (Wormhole N150/N300) using TTNN APIs for ultra-high-throughput, ultra-low-latency voice conversion.

## Features

- **Full LLVC pipeline**: Encoder, decoder, optional pitch extraction, and vocoder integration
- **Streaming mode**: Real-time chunked processing with causal convolutions
- **Non-streaming mode**: Full-context conversion
- **Dual mode support**: F0-based and F0-free voice conversion
- **Optimized for TT hardware**: Sharded memory layouts, fused ops, efficient state management

## Architecture

