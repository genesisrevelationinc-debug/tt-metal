 ```diff
--- a/models/README.md
+++ b/models/README.md
@@ -1,5 +1,7 @@
 # Model Matrix
 
+## Featured Models
+
 | Model | Variant | Hardware | Batch | TTFT (MS) | T/S/U | Target<br>T/S/U | T/S | TT-Metalium Release | vLLM Tenstorrent Repo Release |
 |-------|---------|----------|-------|-----------|-------|-----------------|-----|---------------------|-------------------------------|
 | Llama 3.3 70B | [70B](./demos/llama3_70b_galaxy) | Galaxy (Wormhole) | 32 | 53 | 72.5 | 80 | 2268.8 | [v0.65.0-rc7](https://github.com/tenstorrent/tt-metal/tree/v0.65.0-rc7) | [59be953](https://github.com/tenstorrent/vllm/tree/59be953f2bbd21e227f9ef4b779f545f9c3bf599/tt_metal) |
@@ -7,6 +9,7 @@
 | Stable Diffusion 1.4 | [1.4](./demos/stable_diffusion) | N150 | 1 | - | - | - | - | - | -