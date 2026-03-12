# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from typing import Optional, Dict, Any, Tuple
import numpy as np

from .dualffn import DualFFNTransformer
from .tokenizer import AudioTokenizer
from .decoder import AudioDecoder


class HiggsAudioV2:
    """
    Higgs Audio v2 model implementation using TTNN APIs.
    
    This model provides text-to-audio generation with support for:
    - Text-to-speech
    - Voice cloning
    - Multi-speaker dialog
    """
    
    def __init__(
        self,
        device: ttnn.Device,
        model_path: str,
        max_seq_len: int = 2048,
        dtype: ttnn.DataType = ttnn.bfloat16,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ):
        """
        Initialize Higgs Audio v2 model.
        
        Args:
            device: TTNN device
            model_path: Path to model weights
            max_seq_len: Maximum sequence length
            dtype: Data type for tensors
            memory_config: Memory configuration for tensors
        """
        self.device = device
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.memory_config = memory_config
        
        # Initialize components
        self.tokenizer = AudioTokenizer(device, model_path, dtype=dtype)
        self.llm = DualFFNTransformer(
            device=device,
            model_path=model_path,
            max_seq_len=max_seq_len,
            dtype=dtype,
            memory_config=memory_config,
        )
        self.decoder = AudioDecoder(device, model_path, dtype=dtype)
        
        # KV cache for autoregressive generation
        self.kv_cache = {}
        self.cache_seqlens = {}
        
    def reset_kv_cache(self):
        """Reset KV cache for new generation."""
        self.kv_cache = {}
        self.cache_seqlens = {}
        
    def prepare_text_tokens(self, text: str) -> ttnn.Tensor:
        """
        Prepare text tokens for generation.
        
        Args:
            text: Input text
            
        Returns:
            Text tokens tensor
        """
        # Tokenize text using the model's text tokenizer
        tokens = self.tokenizer.encode_text(text)
        
        # Convert to TTNN tensor
        tokens_torch = torch.tensor(tokens, dtype=torch.int32).unsqueeze(0)
        tokens_ttnn = ttnn.from_torch(
            tokens_torch,
            device=self.device,
            dtype=ttnn.int32,
            memory_config=self.memory_config,
        )
        
        return tokens_ttnn
        
    def prepare_audio_tokens(self, audio_path: Optional[str] = None) -> Optional[ttnn.Tensor]:
        """
        Prepare audio tokens for voice cloning.
        
        Args:
            audio_path: Path to reference audio file
            
        Returns:
            Audio tokens tensor or None if no audio provided
        """
        if audio_path is None:
            return None
            
        # Extract audio features and tokenize
        audio_tokens = self.tokenizer.encode_audio(audio_path)
        
        # Convert to TTNN tensor
        audio_torch = torch.tensor(audio_tokens, dtype=torch.int32).unsqueeze(0)
        audio_ttnn = ttnn.from_torch(
            audio_torch,
            device=self.device,
            dtype=ttnn.int32,
            memory_config=self.memory_config,
        )
        
        return audio_ttnn
        
    def generate(
        self,
        text: str,
        reference_audio: Optional[str] = None,
        speaker_id: Optional[int] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> np.ndarray:
        """
        Generate audio from text.
        
        Args:
            text: Input text
            reference_audio: Path to reference audio for voice cloning
            speaker_id: Speaker ID for multi-speaker generation
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            top_k: Top-k sampling threshold
            
        Returns:
            Generated audio as numpy array
        """
        self.reset_kv_cache()
        
        # Prepare input tokens
        text_tokens = self.prepare_text_tokens(text)
        audio_tokens = self.prepare_audio_tokens(reference_audio)
        
        # Combine tokens based on generation mode
        if audio_tokens is not None:
            # Voice cloning mode - prepend audio tokens
            input_tokens = ttnn.concat([audio_tokens, text_tokens], dim=-1)
        else:
            # Text-to-speech mode
            input_tokens = text_tokens
            
        # Add speaker embedding if specified
        if speaker_id is not None:
            speaker_emb = self.get_speaker_embedding(speaker_id)
        else:
            speaker_emb = None
            
        # Generate audio tokens autoregressively
        generated_tokens = self.generate_tokens(
            input_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            speaker_emb=speaker_emb,
        )
        
        # Decode tokens to audio
        audio = self.decoder.decode(generated_tokens)
        
        return audio
        
    def generate_tokens(
        self,
        input_tokens: ttnn.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        speaker_emb: Optional[ttnn.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate audio tokens using the LLM.
        
        Args:
            input_tokens: Input token sequence
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            top_k: Top-k sampling threshold
            speaker_emb: Speaker embedding for multi-speaker
            
        Returns:
            Generated tokens as PyTorch tensor
        """
        # Convert to PyTorch for processing
        input_torch = ttnn.to_torch(input_tokens).squeeze(0)
        
        generated = input_torch.tolist()
        
        for _ in range(max_new_tokens):
            # Prepare input for current step
            current_input = torch.tensor(generated[-self.max_seq_len:]).unsqueeze(0)
            current_ttnn = ttnn.from_torch(
                current_input,
                device=self.device,
                dtype=ttnn.int32,
                memory_config=self.memory_config,
            )
            
            # Forward pass through LLM
            logits = self.llm.forward(
                current_ttnn,
                kv_cache=self.kv_cache,
                cache_seqlens=self.cache_seqlens,
                speaker_emb=speaker_emb,
            )
            
            # Convert logits to PyTorch for sampling
            logits_torch = ttnn.to_torch(logits).squeeze(0)[-1, :]
            
            # Apply temperature
            logits_torch = logits_torch / temperature
            
            # Apply top-p and top-k filtering
            filtered_logits = self.apply_sampling(logits_torch, top_p, top_k)
            
            # Sample next token
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated.append(next_token)
            
            # Check for end token
            if next_token == self.tokenizer.end_token:
                break
                
        return torch.tensor(generated)
        
    def apply_sampling(
        self,
        logits: torch.Tensor,
        top_p: float,
        top_k: int,
    ) -> torch.Tensor:
        """
        Apply top-p and top-k sampling to logits.
        
        Args:
            logits: Input logits
            top_p: Top-p threshold
            top_k: Top-k threshold
            
        Returns:
            Filtered logits
        """
        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
            
        # Top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, source=sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
            
        return logits
        
    def get_speaker_embedding(self, speaker_id: int) -> ttnn.Tensor:
        """
        Get speaker embedding for multi-speaker generation.
        
        Args:
            speaker_id: Speaker ID
            
        Returns:
            Speaker embedding tensor
        """
        # Load speaker embedding from model weights
        speaker_emb = self.llm.get_speaker_embedding(speaker_id)
        return speaker_emb
        
    def benchmark(
        self,
        test_texts: list,
        num_runs: int = 10,
    ) -> Dict[str, float]:
        """
        Benchmark model performance.
        
        Args:
            test_texts: List of test texts
            num_runs: Number of benchmark runs
            
        Returns:
            Performance metrics
        """
        import time
        
        total_tokens = 0
        total_time = 0
        
        for text in test_texts[:num_runs]:
            start_time = time.time()
            
            # Generate audio
            audio = self.generate(text, max_new_tokens=512)
            
            end_time = time.time()
            
            # Count generated tokens (approximate)
            tokens = len(audio) // 256  # Rough approximation
            total_tokens += tokens
            total_time += (end_time - start_time)
            
        tokens_per_second = total_tokens / total_time
        rtf = total_time / (total_tokens * 0.02)  # Assuming 20ms per token
        
        return {
            "tokens_per_second": tokens_per_second,
            "real_time_factor": rtf,
            "total_tokens": total_tokens,
            "total_time": total_time,
        }
