Based on the issue description, I need to create a Phi-1 model implementation for the Tenstorrent tt-metal repository. Let me examine the existing structure and create the necessary files.

Looking at the repository structure, I need to:
1. Create a demo directory for phi-1 under `models/demos/`
2. Implement the model using tt-transformers base modules
3. Create a demo script for inference

Let me create the implementation files:

```diff
--- /dev/null
+++ b/models/demos/phi1/tt/tt_model.py
@@ -0,0 +1,312 @@
+# SPDX-License-Identifier: Apache-2.0
+#
+# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
+
+import torch
+import ttnn
+from models.tt_transformers.tt.model_config import ModelConfig
+from models.tt_transformers.tt.common import (
+    get_rotary_matrix,
+    precompute_freqs,
+    gather_rotary_emb,
+    apply_rotary_emb,
+    HostEmbedding,
+    HostLMHead,
+)
+from models.tt_transformers.modules import (
+    TransformerBlock,
+    RMSNorm,
+)
+from typing import Optional, Tuple
+
+
+class Phi1ModelConfig(ModelConfig):
+    """Configuration for Phi-1 model."""
+    
+    def __init__(
+        self,
+        num_layers: int = 24,
+        num_heads: int = 32,
+        num_kv_heads: int = 32,
+        hidden_size: int = 2048,
+        vocab_size: int = 51200,
+        max_seq_len: int = 2048,
+        max_batch_size: int = 1,
+        rope_theta: float = 10000.0,
+        partial_rotary_factor: float = 0.5,
+        norm_eps: float = 1e-5,
+        use_qkv_fusion: bool = True,
+        use_attention_fusion: bool = True,
+        use_sdpa: bool = True,
+        use_l1_cache: bool = True,
+        use_sharding: bool = True,
+        dtype: ttnn.DataType = ttnn.bfloat16,
+        cache_dtype: ttnn.DataType = ttnn.bfloat16,
+    ):
+        super().__init__()
+        self.num_layers = num_layers
+        self.num_heads = num_heads
+        self.num_kv_heads = num_kv_heads
+        self.hidden_size = hidden_size
+        self.vocab_size = vocab_size
+        self.max_seq_len = max_seq_len
+        self.max_batch_size = max_batch_size
+        self.rope_theta = rope_theta
+        self.partial_rotary_factor = partial_rotary_factor
+        self.norm_eps = norm_eps
+        self.use_qkv_fusion = use_qkv_fusion
+        self.use_attention_fusion = use_attention_fusion
+        self.use_sdpa = use_sdpa
+        self.use_l1_cache = use_l1_cache
+        self.use_sharding = use_sharding
+        self.dtype = dtype
+        self.cache_dtype = cache_dtype
+        
+        # Derived dimensions
+        self.head_dim = hidden_size // num_heads
+        self.num_kv_groups = num_heads // num_kv_heads
+        self.rotary_dim = int(self.head_dim * partial_rotary_factor)
+        
+        # QKV dimensions
+        self.q_size = num_heads * self.head_dim
+        self.k_size = num_kv_heads * self.head_dim
+        self.v_size = num_kv_heads * self.head_dim
+        self.qkv_size = self.q_size + self.k_size + self.v_size
+
+
+class Phi1Model:
+    """Phi-1 model implementation for Tenstorrent hardware."""
+    
+    def __init__(
+        self,
+        config: Phi1ModelConfig,
+        state_dict: dict,
+        device: ttnn.Device,
+        cache_path: Optional[str] = None,
+    ):
+        self.config = config
+        self.device = device
+        self.state_dict = state_dict
+        self.cache_path = cache_path
+        
+        # Initialize embeddings
+        self.embedding = HostEmbedding(
+            config.vocab_size,
+            config.hidden_size,
+            state_dict["transformer.embd.wte.weight"],
+            device,
+            dtype=config.dtype,
+        )
+        
+        # Initialize transformer layers
+        self.layers = []
+        for i in range(config.num_layers):
+            layer = TransformerBlock(
+                config=config,
+                state_dict=state_dict,
+                layer_idx=i,
+                device=device,
+                cache_path=cache_path,
+                layer_prefix=f"transformer.h.{i}.",
+            )
+            self.layers.append(layer)
+        
+        # Final layer norm
+        self.final_norm = RMSNorm(
+            config.hidden_size,
+            config.norm_eps,
+            state_dict["lm_head.ln.weight"],
+            device,
+            dtype=config.dtype,
+        )
+        
+        # LM head
+        self.lm_head = HostLMHead(
+            config.hidden_size,
+            config.vocab_size,
+            state_dict["lm_head.linear.weight"],
+            state_dict.get("lm_head.linear.bias"),
+            device,
+            dtype=config.dtype,
+        )
+        
+        # Precompute rotary embeddings
+        self.freqs_cis = precompute_freqs(
+            config.head_dim,
+            config.max_seq_len * 2,
+            config.rope_theta,
+        )
+        
+        # Initialize KV cache
+        self.kv_cache = self._init_kv_cache()
+    
+    def _init_kv_cache(self):
+        """Initialize KV cache for efficient inference."""
+        cache = []
+        for _ in range(self.config.num_layers):
+            k_cache = torch.zeros(
+                self.config.max_batch_size,
+                self.config.num_kv_heads,
+                self.config.max_seq_len,
+                self.config.head_dim,
+                dtype=torch.bfloat16,
+            )
+            v_cache = torch.zeros(
+                self.config.max_batch_size,
+                self.config.num_kv_heads,
+                self.config.max_seq_len,
+                self.config.head_dim,
+                dtype=torch.bfloat16,
