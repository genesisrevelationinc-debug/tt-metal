# AudioX Model Bring-Up on TT-Metal

This directory contains the TTNN implementation of **AudioX**, a unified framework for anything-to-audio generation.

## Overview

AudioX supports multimodal control signals for audio generation:
- **Text-to-audio**: Generate sound effects from text descriptions
- **Text-to-music**: Generate music from text prompts
- **Video-to-audio**: Generate audio synchronized with video
- **Video-to-music**: Generate background music for video
- **Audio inpainting**: Fill in missing audio segments
- **Music completion**: Continue musical pieces

## Architecture

The model consists of the following TTNN modules:

1. **Multimodal Encoders** (`ttnn_audiox_encoders.py`)
   - Text encoder (CLAP-based)
   - Video encoder
   - Image encoder
   - Audio encoder

2. **Multimodal Adaptive Fusion** (`ttnn_audiox_fusion.py`)
   - Adaptive fusion of multimodal inputs
   - Cross-modal attention mechanisms

3. **Diffusion Transformer** (`ttnn_audiox_transformer.py`)
   - Diffusion-based generation with transformer architecture
   - Self-attention and cross-attention layers
   - DDPM/DDIM sampling

4. **Vocoder** (`ttnn_audiox_vocoder.py`)
   - Waveform generation from mel-spectrograms

## Requirements

- Tenstorrent hardware: N150 or N300 (Wormhole or Blackhole)
- tt-metal built with TTNN support

## Setup

