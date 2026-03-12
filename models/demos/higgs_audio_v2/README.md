# Higgs Audio v2 Model Bring-Up using TTNN APIs

This directory contains the implementation of the Higgs Audio v2 model using TTNN APIs on Tenstorrent hardware.

## Overview

Higgs Audio v2 is a cutting-edge text-audio foundation model from Boson AI that redefines expressiveness in audio generation. This implementation focuses on bringing up the model on Tenstorrent hardware for high-throughput, low-latency audio generation.

## Features

- **LLM Backbone with DualFFN Architecture**: Implements the LLM backbone with a DualFFN architecture for enhanced acoustic modeling.
- **Audio Tokenizer**: Integrates a custom audio tokenizer for unified semantic and acoustic tokenization.
- **Audio Decoder**: Converts tokens to waveform for audio generation.
- **Multiple Generation Modes**: Supports text-to-speech, voice cloning, and multi-speaker dialog.

## Setup

1. **Install Dependencies**: Follow the instructions in the [INSTALLING.md](../../INSTALLING.md) file.
2. **Load Model**: Use the `load_model` function to initialize the model.
3. **Generate Audio**: Use the `generate_audio` function to generate audio from text.

## Usage

