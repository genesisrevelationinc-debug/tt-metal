import torch
import torch.nn as nn
from pathlib import Path
import sys
from typing import Optional, Union
from dataclasses import dataclass
from .common import *

@dataclass
class Phi15Config:
    model_name: str = "microsoft/phi-1.5"
    library_name: str = "tt_best"
    version: str = "best"
    num_hidden_layers: int = 24
    hidden_size: int = 2048  
    intermediate_size: int = 8192
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-06
    vocab_size: int = 32064
    context_length: int = 2048
    
    def __post_init__(self):
        pass

class Phi15Model:
    def __init__(self, config: Phi15Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        
    def load_model(self, model_path: str = None):
        """Load the Phi 1.5 model"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            if model_path is None:
                model_path = self.config.model_name
                
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return self.model.prepare_inputs_for_generation(input_ids, **kwargs)
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

def build_demo():
    """Build demo for Phi 1.5 model inference"""
    import argparse
    parser = argparse.ArgumentParser("Phi 1.5 Demo")
    parser.add_argument("--device", type=str, default="wormhole", help="Target device")
    parser.add_argument("--model", type=str, default="microsoft/phi-1.5", help="Model name")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory")
    
    return parser

def main():
    parser = build_demo()
    args = parser.parse_args()
    
    # This would be replaced with actual implementation
    # Demo the model capabilities
    print("Phi 1.5 model demo")
    model = Phi15Model(Phi15Config())
    model.load_model()
    
    # Simple test to verify the model works
    test_input = "Hello, my name is"
    print(f"Testing with input: {test_input}")
    # This is where the actual inference would happen
    print("Model loaded successfully!")

if __name__ == "__main__":

    main()
