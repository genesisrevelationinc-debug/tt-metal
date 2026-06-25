# AudioX Model Bring-Up on TTNN

This directory contains the TTNN implementation of **AudioX**, a unified framework for anything-to-audio generation, brought up on Tenstorrent hardware.

## Overview

AudioX is a diffusion transformer-based model for multimodal audio generation. This implementation uses TTNN APIs to run the model on Tenstorrent Wormhole (N150/N300) and Blackhole hardware.

## Supported Generation Modes

- **Text-to-Audio**: Generate sound effects from text descriptions
- **Text-to-Music**: Generate music from text prompts
- **Video-to-Audio**: Generate audio synchronized with video
- **Video-to-Music**: Generate background music for video
- **Audio Inpainting**: Fill in missing portions of audio
- **Music Completion**: Continue musical pieces

## Architecture

The model consists of the following components:

1. **Multimodal Encoders**: Process text, video, image, and audio inputs
2. **Multimodal Adaptive Fusion**: Fuses diverse multimodal signals
3. **Diffusion Transformer Decoder**: Generates latent audio representations
4. **Vocoder**: Converts latents into 16 kHz waveform output

## Setup

### Prerequisites

- tt-metal built and installed (see [INSTALLING.md](../../../INSTALLING.md))
- Python 3.8+
- 32 GB+ RAM recommended

### Installation

