# Phi-1 Demo on Wormhole

This directory contains the bring-up and demo for Microsoft's Phi-1 model on Tenstorrent Wormhole hardware (N150/N300).

## Overview

Phi-1 is a 1.3B parameter transformer-based language model optimized for code and text generation. This implementation leverages the `tt-transformers` base modules to run end-to-end inference on Wormhole.

## Quick Start

1. Ensure you have a working tt-metal installation and Wormhole hardware.
2. Download the Phi-1 weights from HuggingFace:
   