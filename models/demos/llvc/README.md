# LLVC (Low-Latency Low-Resource Voice Conversion) Model Demo

This directory contains the implementation of the LLVC model using TTNN APIs for Tenstorrent hardware.

## Overview

LLVC is a real-time voice conversion model optimized for low latency and CPU efficiency. This implementation enables the model to run on Tenstorrent hardware (Wormhole or Blackhole) with ultra-low latency and high throughput.

## Features

* Ultra-low latency real-time voice conversion
* Streaming and non-streaming inference modes
* Optimized for N150/N300 Tenstorrent hardware
* Pitch extraction (F0-based and F0-free modes)
* Vocoder integration
* Efficient memory management with TTNN APIs

## Directory Structure

