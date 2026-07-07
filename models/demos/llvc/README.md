# LLVC (Low-Latency Low-Resource Voice Conversion) on Tenstorrent

This directory contains the LLVC model implementation using TTNN APIs for Tenstorrent hardware (Wormhole/Blackhole).

## Overview

LLVC is a real-time voice conversion model optimized for low latency and CPU efficiency. This implementation brings LLVC to Tenstorrent hardware for ultra-high-throughput, ultra-low-latency voice conversion.

## Features

- **Streaming mode**: Real-time conversion with chunked processing
- **Non-streaming mode**: Full-context conversion
- **F0-based and F0-free modes**: Optional pitch extraction
- **Optimized for TT hardware**: Sharded memory, fused ops, efficient state management

## Architecture

