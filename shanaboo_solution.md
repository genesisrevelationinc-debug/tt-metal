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
 from typing import List, Optional, Tuple
 from models.tt_transformers.tt.common import (
     precompute_freqs,
+    precompute_freqs_for_finetune,
     sample,
     HostEmbedding,
     PagedAttentionConfig,
@@ -24,6 +26,7 @@
     TtLlamaAttention,
     TtLlamaDecoderLayer,
 )
+from models.tt_transformers.tt.phi_attention import TtPhiAttention, TtPhiDecoderLayer
 from models.tt_transformers.tt.llama_mlp import TtLlamaMLP
 from models.tt_transformers.tt.norm import TtRMSNorm, TtLayerNorm
 from models.tt_transformers.tt.embeddings import TtLlamaEmbedding
@@ -32,6 +35,7 @@
     ModelArgs,
     TransformerBlock,
     LlamaLikeConfig,
+    PhiConfig,
 )
 from models.tt_transformers.tt.distributed import TtDistributedLlama
 from models.tt_transformers.tt.rope import TtLlamaRotarySetup
@@ -43,6 +47,7 @@
     "LlamaForCausalLM",
     "LlamaForConditionalGeneration",
     "Qwen2ForCausalLM",
+    "PhiForCausalLM",
 ]
 
 
@@ -52,6 +57,7 @@
     "LlamaForCausalLM": "llama",
     "LlamaForConditionalGeneration": "llama",
     "Qwen2ForCausalLM": "qwen2",
+    "PhiForCausalLM": "phi",
 }
 
 
@@ -61,6 +67,7 @@
     "LlamaForCausalLM": "llama",
     "LlamaForConditionalGeneration": "llama",
     "Qwen2ForCausalLM": "qwen2",
+    "PhiForCausalLM": "phi",
 }
 
 
@@ -68,6 +75,7 @@
     "LlamaForCausalLM": TtLlamaAttention,
     "LlamaForConditionalGeneration": TtLlamaAttention,
     "Qwen2ForCausalLM": TtLlamaAttention,
+    "PhiForCausalLM": TtPhiAttention,
 }
 
 
@@ -75,6 +83,7 @@
     "LlamaForCausalLM": TtLlamaDecoderLayer,
     "LlamaForConditionalGeneration": TtLlamaDecoderLayer,
     "Qwen2ForCausalLM": TtLlamaDecoderLayer,
+    "PhiForCausalLM": TtPhiDecoderLayer,
 }
 
 
@@ -82,6 +91,7 @@
     "LlamaForCausalLM": TtRMSNorm,
     "LlamaForConditionalGeneration": TtRMSNorm,
     "Qwen2ForCausalLM": TtRMSNorm,
+    "PhiForCausalLM": TtLayerNorm,
 }
 
 
@@ -89,6 +99,7 @@
     "LlamaForCausalLM": TtLlamaEmbedding,
     "LlamaForConditionalGeneration": TtLlamaEmbedding,
     "Qwen2ForCausalLM": TtLlamaEmbedding,
+    "PhiForCausalLM": TtLlamaEmbedding,
 }
 
 
@@ -96,6 +107,7 @@
     "LlamaForCausalLM": LlamaLikeConfig,
     "LlamaForConditionalGeneration": LlamaLikeConfig,
     "Qwen2ForCausalLM": LlamaLikeConfig,
+    "PhiForCausalLM": PhiConfig,
 }
 
 
@@ -103,6 +115,7 @@
     "LlamaForCausalLM": "rope",
     "LlamaForConditionalGeneration": "rope",
     "Qwen2ForCausalLM": "rope",
+    "PhiForCausalLM": "partial_rotary",
 }
 
 
@@ -110,6 +123,7 @@
     "LlamaForCausalLM": False,
     "LlamaForConditionalGeneration": False,
     "Qwen2ForCausalLM": False,
+    "PhiForCausalLM": True,
 }
 
 
@@ -117,6 +131,7 @@
     "LlamaForCausalLM": False,
     "LlamaForConditionalGeneration": False,
     "Qwen2ForCausalLM": False,
+    "PhiForCausalLM": True,
 }
 
 
@@ -124,6 +139,7 @@
     "LlamaForCausalLM": False,
     "LlamaForConditionalGeneration": False,
     "Qwen2ForCausalLM": False,
+    "PhiForCausalLM": True,
 }
 
 
@@ -131,6 +147,7 @@
     "LlamaForCausalLM": False,
     "LlamaForConditionalGeneration": False,
     "Qwen2ForCausalLM": False,
+    "PhiForCausalLM": True,
 }
 
 
@@ -138,6 +155,7 @@
     "LlamaForCausalLM": False,
     "LlamaForConditionalGeneration": False,
     "Qwen2ForCausalLM": False,
+    "PhiForCausalLM": True,
 }
 
 
@@ -145,6 +163,7 @@
     "LlamaForCausalLM": False,
     "LlamaForConditionalGeneration": False,
     "Qwen2ForCausalLM": False,
+    "PhiForCausalLM": True,
 }
 
 
@@ -152,6 +171,7 @@
     "LlamaForCausalLM": False,
     "LlamaForConditionalGeneration": False,
     "Qwen2ForCausalLM": False,
+    "PhiForCausalLM": True,
 }
 
 
@@ -159,6 +179,7 @@
     "LlamaForCausalLM": False,
     "LlamaForConditionalGeneration": False,
     "Qwen2ForCausalLM": False,
+    "PhiForCausalLM": True,
 }
 
 
@@ -166,6 +187,7 @@
     "LlamaForCausalLM": False,
     "LlamaForConditionalGeneration": False,
     "Qwen2ForCausalLM": False,
+    "PhiForCausalLM": True,
 }
 
 
@@ -173,6 +195,7 @@
     "LlamaForCausalLM": False,
     "LlamaForConditionalGeneration": False,
     "Qwen2ForCausalLM": False,
+    "PhiForCausalLM": True,
 }
 
 
@@ -180,6 +203