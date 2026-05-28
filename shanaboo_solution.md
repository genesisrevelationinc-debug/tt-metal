```diff
--- a/README.md
+++ b/README.md
@@ -1,3 +1,39 @@
+<details>
+<summary>Table of Contents</summary>
+
+<!-- Please update this table of contents with new tt-metal PRs -->
+- [Installation](#installation)
+- [Model List](#model-list)
+- [Performance Metrics](#performance-metrics)
+- [Models](#models)
+  - [Llama 3.3 70B (TP=32)](#llama-33-70b-tp32)
+  - [Qwen 2.5 7B (TP=2)](#qwen-25-7b-tp2)
+- [Hardware](#hardware)
+- [Install](#install)
+- [Discord](#discord)
+- [Join Us](#join-us)
+- [Bounty $](#bounty-)
+</details>
+
+<h1>
+<br>
+
+# <a name="model-list"></a>Model List
+
+The following models are currently supported and optimized for performance, accuracy, and compatibility:
+
+- [Llama 3.3 70B (TP=32)](./models/demos/llama3_70b_galaxy)
+- [Qwen 2.5 7B (TP=2)](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)
+
+## Performance Metrics
+
+The following performance metrics are based on an input sequence length of 128 tokens for all models.
+
+## Hardware
+
+For hardware specifications, visit [Tenstorrent Hardware](https://tenstorrent.com/hardware).
+
+## Install
+
+To install, follow the instructions in [INSTALLING.md](./INSTALLING.md).
+
+## Discord
+
+Join our community on [Discord](https://discord.gg/tvhGzHQwaj).
+
+## Join Us
+
+We're hiring! Check out our [careers page](https://boards.greenhouse.io/tenstorrent/jobs/4155609007).
+
+## Bounty $
+
+See [Bounty $](https://github.com/tenstorrent/tt-metal/issues?q=is%3Aissue%20state%3Aopen%20label%3Abounty).
+
+## Model Demos
+
+Check out our [model demos](./models/demos/).
+
+## LLM Performance
+
+The following performance metrics are based on an input sequence length of 128 tokens for all models.
+
+## TT-Metalium Release
+
+For the latest release, see [TT-Metalium Releases](https.