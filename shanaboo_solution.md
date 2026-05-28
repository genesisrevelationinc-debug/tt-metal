```diff
--- a/models/tt_transformers/tt_transformers/llama.py
+++ b/models/tt_transformers/tt_transformers/llama.py
@@ -1,6 +1,7 @@
 # SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 
 # SPDX-License-Identifier: Apache-2.0
+
 import math
 import torch
 import ttnn
@@ -10,6 +11,7 @@
 from models.utility_functions import nearest_32
 from models.tt_transformers.tt.common import (
     precompute_freqs,
+    precompute_freqs_for_phi,
     sample,
     precompute_rotary_embeddings,
     encode_prompt_llama_instruct,
@@ -18,6 +20,7 @@
 from models.tt_transformers.tt.llama_decoder import TtTransformerBlock
 from models.tt_transformers.tt.llama_ccl import send_to_all_gather, send_to_mcast_parallel
 from models.tt_transformers.tt.model_config import ModelArgs, TransformerConfig
+from models.tt_transformers.tt.model_config import PhiConfig
 from models.tt_transformers.tt.distributed_processing import (
     DistributedNorm,
     DistributedFFN,
@@ -30,6 +33,7 @@
     TtLlamaEmbedding,
     TtLlamaRotaryEmbedding,
     TtLlamaRotaryEmbeddingQwen,
+    TtLlamaRotaryEmbeddingPhi,
 )
 from models.tt_transformers.tt.norms import TtRMSNorm, TtFalconMambaRMSNorm
 from models.tt_transformers.tt.llama_mlp import TtLlamaMLP
@@ -40,6 +44,7 @@
     TtLlamaAttentionQwen,
     TtLlamaAttentionFalconMamba,
     TtLlamaAttentionMamba,
+    TtLlamaAttentionPhi,
 )
 from models.tt_transformers.tt.llama_mlp import TtLlamaMLP, TtLlamaMLPQwen
 from models.tt_transformers.tt.llama_decoder import TtTransformerBlock
@@ -56,6 +61,7 @@
     "llama": TtLlamaAttention,
     "qwen": TtLlamaAttentionQwen,
     "falcon_mamba": TtLlamaAttentionFalconMamba,
+    "phi": TtLlamaAttentionPhi,
 }
 
 MLP_CLASS = {
@@ -68,6 +74,7 @@
     "llama": TtLlamaRotaryEmbedding,
     "qwen": TtLlamaRotaryEmbeddingQwen,
     "falcon_mamba": TtLlamaRotaryEmbedding,
+    "phi": TtLlamaRotaryEmbeddingPhi,
 }
 
 NORM_CLASS = {
@@ -75,6 +82,7 @@
     "qwen": TtRMSNorm,
     "falcon_mamba": TtFalconMambaRMSNorm,
     "mamba": TtRMSNorm,
+    "phi": TtRMSNorm,
 }
 
 
@@ -85,6 +93,7 @@
     "qwen": TtLlamaMLPQwen,
     "falcon_mamba": TtLlamaMLP,
     "mamba": TtLlamaMLP,
+    "phi": TtLlamaMLP,
 }
 
 
@@ -93,6 +102,7 @@
     "qwen": TtLlamaMLPQwen,
     "falcon_mamba": TtLlamaMLP,
     "mamba": TtLlamaMLP,
+    "phi": TtLlamaMLP,
 }
 
 
@@ -101,6 +111,7 @@
     "qwen": TtLlamaMLPQwen,
     "falcon_mamba": TtLlamaMLP,
     "mamba": TtLlamaMLP,
+    "phi": TtLlamaMLP,
 }
 
 
@@ -109,6 +120,7 @@
     "qwen": TtLlamaMLPQwen,
     "falcon_mamba": TtLlamaMLP,
     "mamba": TtLlamaMLP,
+    "phi": TtLlamaMLP,
 }
 
 
@@ -117,6 +129,7 @@
     "qwen": TtLlamaMLPQwen,
     "falcon_mamba": TtLlamaMLP,
     "mamba": TtLlamaMLP,
+    "phi": TtLlamaMLP,
 }
 
 
@@ -125,6 +138,7 @@
     "qwen": TtLlamaMLPQwen,
     "falcon_mamba": TtLlamaMLP,
     "mamba": TtLlamaMLP,
+    "phi": TtLlamaMLP,
 }
 
 
@@ -133,6 +147,7 @@
     "qwen": TtLlamaMLPQwen,
     "falcon_mamba": TtLlamaMLP,
     "mamba": TtLlamaMLP,
+    "phi": TtLlamaMLP,
 }
 
 
@@ -141,6 +156,7 @@
     "qwen": TtLlamaMLPQwen,
     "falcon_mamba": TtLlamaMLP,
     "mamba": TtLlamaMLP,
+    "phi": TtLlamaMLP,
 }
 
 
@@ -149,6 +165,7 @@
     "qwen": TtLlamaMLPQwen,
     "falcon_mamba": TtLlamaMLP,
     "mamba": TtLlamaMLP,
+    "phi": TtLlamaMLP,
 }
 
 
@@ -157,6 +174,7 @@
     "qwen": TtLlamaMLPQwen,
     "falcon_mamba": TtLlamaMLP,
     "mamba": TtLlamaMLP,
+    "phi": TtLlamaMLP,
 }
 
 
@@ -165,6 +183,7 @@
     "qwen": TtLlamaMLPQwen,
     "falcon_mamba": TtLlamaMLP,
     "mamba": TtLlamaMLP,
+    "phi": TtLlamaMLP,
 }
 
 
@@ -173,6 +192,7 @@
     "qwen": TtLlamaMLPQwen,
     "falcon_mamba": TtLlamaMLP,
     "mamba": TtLlamaMLP,
+    "phi": TtLlamaMLP,
 }
 
 
@@ -181,6 +201,7 @@
     "qwen": TtLlamaMLPQwen,
     "falcon_mamba": TtLlamaMLP,
     "mamba": TtLlamaMLP,
+    "phi": TtLlamaMLP,
 }
 
 
@@ -189,6 +210,7 @@
    