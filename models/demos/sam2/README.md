# SAM2 (Segment Anything Model 2) - TTNN Implementation

This directory contains the TTNN implementation of [facebook/sam2-hiera-tiny](https://huggingface.co/facebook/sam2-hiera-tiny) for Tenstorrent hardware (Wormhole/Blackhole).

## Overview

SAM 2 is Meta's foundation model for promptable visual segmentation. This implementation covers the image-mode pipeline:
- Hiera image encoder
- Prompt encoder (point, box, mask prompts)
- Two-way transformer mask decoder

Video tracking (memory encoder/attention/bank) is out of scope.

## Model Architecture

