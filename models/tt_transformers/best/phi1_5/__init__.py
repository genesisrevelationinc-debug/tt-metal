"""
Phi-1.5 model implementation for Tenstorrent hardware acceleration.
"""

from .model import Phi15Config, Phi15Model
import torch
import argparse

def build_parser():
    parser = argparse.ArgumentParser(description="Run Phi-1.5 model on Tenstorrent hardware")
    parser.add_argument('--model', type=str, default='microsoft/phi-1.5', help='Model name')
    parser.add_argument('--output_dir', type=str, default='./generated', help='Output directory')
    parser.add_argument('--device', type=str, default='wormhole', help='Target device')
    return parser

def main():
    # Parse arguments
    parser = build_parser()
    args = parser.parse_args()
    
    # Initialize model
    config = Phi15Config()
    model = Phi15Model(config)
    
    # Test the model
    if model.load_model():
        print("Phi-1.5 model loaded successfully!")
        # Run a simple test
        test_input = "The future of AI"
        print(f"Testing generation with input: {test_input}")
        # Here would be the actual inference code
    else:
        print("Failed to load model")
        return
        
    # Placeholder for actual implementation
    print("Running on Tenstorrent Wormhole hardware...")
    print("Model:", args.model)
    print("Output directory:", args.output_dir) 
    print("Device:", args.device)
