# AudioX Model Bring-Up on TTNN

This directory contains the TTNN implementation of **AudioX**, a unified framework for anything-to-audio generation from HKUST.

## Overview

AudioX supports multimodal control signals for audio generation:
- **Text-to-audio**: Generate sound effects from text descriptions
- **Text-to-music**: Generate music from text prompts
- **Video-to-audio**: Generate audio synchronized with video
- **Video-to-music**: Generate background music for video
- **Audio inpainting**: Fill in missing audio segments
- **Music completion**: Continue musical pieces

## Architecture

The model consists of:
1. **Multimodal Encoders**: Text (CLAP), video, image, and audio encoders
2. **Multimodal Adaptive Fusion**: Novel fusion module for diverse multimodal inputs
3. **Diffusion Transformer**: Diffusion-based generation with transformer architecture
4. **Vocoder**: Waveform generation for 16 kHz audio output

## Setup

### Prerequisites

