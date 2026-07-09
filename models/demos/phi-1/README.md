# Phi-1 Model Demo on Wormhole

This directory contains the implementation and demo for running Microsoft's Phi-1 model on Tenstorrent Wormhole hardware (N150/N300).

## Model Overview

Phi-1 is a 1.3B parameter transformer-based language model designed by Microsoft for efficient code and text generation tasks. It features:
- 24 transformer layers
- Hidden size of 2048
- 32 attention heads
- Rotary position embeddings (RoPE)
- GELU activation
- Maximum sequence length of 2048

## Quick Start

### Prerequisites
- Tenstorrent Wormhole N150 or N300 card
- tt-metal installed and configured
- PyTorch with transformers library

### Running the Demo

