# Phi-1 Model Demo on Wormhole

This directory contains the implementation and demo for running `microsoft/phi-1` on Tenstorrent Wormhole hardware (N150/N300).

## Model Overview

Phi-1 is a 1.3B parameter transformer-based language model developed by Microsoft, designed for code and text generation tasks with efficiency and compactness.

### Architecture

- **Parameters**: ~1.3B
- **Layers**: 24 transformer layers
- **Hidden Size**: 2048
- **Intermediate Size**: 8192 (MLP expansion)
- **Attention Heads**: 32 (with rotary position embeddings)
- **Vocabulary Size**: 51200
- **Max Sequence Length**: 2048
- **Activation**: GELU (approximated)
- **Normalization**: LayerNorm

## Quick Start

### Prerequisites

- Tenstorrent Wormhole N150/N300 hardware
- tt-metal installed and configured
- Model weights downloaded from HuggingFace

### Download Model Weights

