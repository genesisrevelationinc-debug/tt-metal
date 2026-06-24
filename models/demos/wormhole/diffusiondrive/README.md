# DiffusionDrive TTNN Implementation

This directory contains the TTNN implementation of [DiffusionDrive](https://github.com/hustvl/DiffusionDrive), a truncated diffusion model for real-time end-to-end autonomous driving.

## Overview

DiffusionDrive replaces the slow multi-step denoising process of conventional diffusion policies with a lightweight anchored and truncated diffusion process, achieving over 10× faster inference while maintaining multi-modal driving behavior.

## Architecture

The model consists of:
- **Backbone**: ResNet-34 for feature extraction from camera images
- **Neck**: Feature pyramid network for multi-scale features
- **Heads**: Trajectory prediction heads using truncated diffusion

## Setup

### Prerequisites

