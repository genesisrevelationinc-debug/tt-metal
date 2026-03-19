# SpeechT5 Voice Conversion Model Bring-Up using TTNN APIs

This demo brings up the Microsoft SpeechT5 Voice Conversion model using TTNN APIs on Tenstorrent hardware (Wormhole or Blackhole).

## Overview

SpeechT5-VC is a unified-modal encoder-decoder model for voice conversion (speech-to-speech) introduced by Microsoft Research. This demo implements the full generation pipeline:

- Speech pre-net (input processing)
- Shared encoder (speech feature extraction)
- Shared decoder (with speaker conditioning)
- Speech post-net (mel-spectrogram generation)
- HiFi-GAN vocoder (waveform generation)

## Setup

1. Clone the repository:
   