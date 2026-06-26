# LiquidAI LFM2.5-VL-1.6B on TT-Metal

This directory contains the implementation of [LiquidAI LFM2.5-VL-1.6B](https://www.liquid.ai/liquid-foundation-models), a general-purpose vision-language model designed for OCR and document comprehension with variable resolution support.

## Model Overview

LFM2.5-VL-1.6B is a vision-language model that can process both text and images at variable resolutions. It is suitable for:
- Optical Character Recognition (OCR)
- Document comprehension and understanding
- General vision-language tasks

## Architecture

The model consists of:
- **Vision Encoder**: Processes input images and extracts visual features
- **Vision-Language Connector**: Projects vision features to the language model space
- **Language Model**: A 1.6B parameter transformer for text generation

## Setup

Ensure you have the required dependencies installed:

