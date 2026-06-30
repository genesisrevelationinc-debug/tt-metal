# SAM2 (Segment Anything Model 2) on TTNN

This directory contains the TTNN implementation of Meta's SAM2 (Segment Anything Model 2) for image segmentation on Tenstorrent hardware.

## Model

- **Model**: facebook/sam2-hiera-tiny
- **Input**: 1024×1024 images
- **Modes**: Image mode (encoder + prompt encoder + mask decoder)
- **Hardware**: N150, N300, Wormhole, Blackhole

## Architecture

The implementation consists of three main components:

1. **Image Encoder (Hiera)**: Processes input images through patch embedding and transformer blocks
2. **Prompt Encoder**: Encodes point, box, and mask prompts
3. **Mask Decoder**: Two-way transformer decoder that produces segmentation masks

## Files

- `tt/sam2_model.py` - Main SAM2 model implementation
- `tt/sam2_hiera_encoder.py` - Hiera image encoder
- `tt/sam2_prompt_encoder.py` - Prompt encoder for points, boxes, masks
- `tt/sam2_mask_decoder.py` - Mask decoder with two-way transformer
- `tt/sam2_utils.py` - Utility functions and helpers
- `reference/sam2_reference.py` - Reference implementation wrapper
- `tests/test_sam2_encoder.py` - Encoder unit tests
- `tests/test_sam2_full_model.py` - Full model integration tests
- `demo/demo.py` - Interactive demo script

## Usage

### Running Tests

