# Microsoft Phi-1 on Tenstorrent Wormhole

This directory contains the implementation of [microsoft/phi-1](https://huggingface.co/microsoft/phi-1) for Tenstorrent's Wormhole hardware (N150/N300).

## Model Overview

Phi-1 is a 1.3B parameter transformer-based language model optimized for code and text generation. It uses a compact architecture with:

- 24 layers
- 32 attention heads
- 2048 hidden dimension
- 2048 context length
- Rotary Position Embeddings (RoPE)
- SwiGLU activation
- LayerNorm (not RMSNorm)

## Files

- `model.py` - Core model implementation using tt-transformers base modules
- `demo.py` - Inference demo script
- `test_phi1.py` - Unit tests for model components

## Usage

### Running the Demo

