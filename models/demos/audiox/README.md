# AudioX Model Bring-Up using TTNN APIs

This directory contains the implementation of the AudioX model using TTNN APIs on Tenstorrent hardware (Wormhole or Blackhole).

## Key Features
- **Anything-to-audio**: Supports text, video, image, and audio inputs for audio generation
- **Multiple tasks**: Text-to-audio, text-to-music, video-to-audio, video-to-music, audio inpainting, music completion
- **Multimodal Adaptive Fusion**: Novel module for effective fusion of diverse multimodal inputs
- **Large-scale training**: IF-caps dataset with 7 million high-quality samples
- **State-of-the-art performance**: Superior results on AudioCaps, MusicCaps, T2A-bench, and AudioTime benchmarks
- **Diffusion Transformer**: Uses diffusion-based generation with transformer architecture
- **Instruction following**: Handles complex temporal instructions (timestamps, ordering, counting)
- **High quality**: 16 kHz audio output with natural sound quality

## Setup
1. Install TT-Metal and TT-NN following the [INSTALLING.md](../../INSTALLING.md) guide.
2. Clone the AudioX repository and place it in the `models/demos/audiox` directory.
3. Run the setup script:
   