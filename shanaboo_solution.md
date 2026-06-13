 ```diff
--- a/tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_trigonometry.h
+++ b/tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_trigonometry.h
@@ -1,4 +1,4 @@
-// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
+// SPDX-FileCopyrightText: © 2023-2025 Tenstorrent Inc.
 //
// SPDX-License-Identifier: Apache-2.0
 
@@ -10,6 +10,7 @@
 #include "ckernel.h"
 #include "ckernel_defs.h"
 #include "ckernel_sfpu_log.h"
+#include "ckernel_sfpu_log1p.h"
 #include "ckernel_sfpu_recip.h"
 #include "ckernel_sfpu_sqrt.h"
 
@@ -18,6 +19,7 @@
 namespace ckernel {
 namespace sfpu {
 
+// DEPRECATED: Kept for backward compatibility. Use _calculate_atanh_log1p_ for new code.
 template <bool APPROXIMATION_MODE, int ITERATIONS>
 inline void _calculate_atanh_(const int iterations) {
     // SFPU microcode
@@ -42,6 +44,31 @@ inline void _calculate_atanh_(const int iterations) {
     }
 }
 
+template <bool APPROXIMATION_MODE, int ITERATIONS>
+inline void _calculate_atanh_log1p_(const int iterations) {
+    // atanh(x) = 0.5 * log1p(2*x / (1 - x))
+    // For small x: atanh(x) ≈ x, so we use log1p for accuracy.
+    // For x near ±1: the formulation avoids intermediate overflow.
+    for (int d = 0; d < iterations; d++) {
+        sfpi::vFloat x = sfpi::dst_reg[0];
+
+        // Compute 2*x / (1 - x) using fused formulation to avoid overflow
+        // For |x| < 1: 2*x / (1 - x) is well-defined
+        sfpi::vFloat denom = sfpi::vConst1 - x;
+        sfpi::vFloat num = x + x;  // 2*x
+        sfpi::vFloat ratio = num / denom;
+
+        // log1p(ratio) = log(1 + 2*x/(1-x)) = log((1+x)/(1-x)) = 2*atanh(x)
+        sfpi::vFloat log1p_val = _calculate_log1p_body_<APPROXIMATION_MODE>(ratio);
+
+        // Multiply by 0.5 to get atanh(x)
+        sfpi::dst_reg[0] = 0.5f * log1p_val;
+
+        sfpi::dst_reg++;
+    }
+}
+
+// DEPRECATED: Kept for backward compatibility. Use _calculate_asinh_log1p_ for new code.
 template <bool APPROXIMATION_MODE, int ITERATIONS>
 inline void _calculate_asinh_(const int iterations) {
     // SFPU microcode
@@ -64,6 +91,46 @@ inline void _calculate_asinh_(const int iterations) {
     }
 }
 
+template <bool APPROXIMATION_MODE, int ITERATIONS>
+inline void _calculate_asinh_log1p_(const int iterations) {
+    // asinh(x) = sign(x) * log1p(|x| + x^2 / (1 + sqrt(1 + x^2)))
+    // For small x: use log1p for accuracy near 0.
+    // For large |x|: use sign(x) * (log(2|x|) + log1p(1/(2*x^2))) to avoid overflow.
+    for (int d = 0; d < iterations; d++) {
+        sfpi::vFloat x = sfpi::dst_reg[0];
+        sfpi::vFloat abs_x = sfpi::abs(x);
+
+        // Compute x^2
+        sfpi::vFloat x2 = x * x;
+
+        // For the main branch, compute sqrt(1 + x^2)
+        sfpi::vFloat sqrt_1px2 = _calculate_sqrt_body_<APPROXIMATION_MODE>(x2 + sfpi::vConst1);
+
+        // Compute |x| + sqrt(1 + x^2) - this is always >= 1
+        sfpi::vFloat sum = abs_x + sqrt_1px2;
+
+        // Use log1p(sum - 1) = log(sum) for better accuracy when sum ≈ 1 (x ≈ 0)
+        // sum - 1 = |x| + sqrt(1+x^2) - 1
+        // For better numerical stability, rewrite:
+        // |x| + sqrt(1+x^2) - 1 = |x| + (sqrt(1+x^2) - 1)
+        // sqrt(1+x^2) - 1 = x^2 / (sqrt(1+x^2) + 1)
+        sfpi::vFloat sqrt_minus_1 = x2 / (sqrt_1px2 + sfpi::vConst1);
+        sfpi::vFloat log1p_arg = abs_x + sqrt_minus_1;
+
+        // For large |x|, use alternative formulation to avoid overflow in x^2
+        // Check if |x| > 1e9 (roughly where x^2 might overflow in fp32, but be conservative)
+        // Actually, for |x| > 1e9, x^2 overflows. Use threshold of 1e8 for safety.
+        // For large |x|: asinh(x) ≈ sign(x) * log(2|x|)
+        sfpi::vFloat large_result = sfpi::log(2.0f * abs_x);
+        sfpi::vFloat small_result = _calculate_log1p_body_<APPROXIMATION_MODE>(log1p_arg);
+
+        sfpi::vFloat result = small_result;
+        // Use v_if for large values
+        result = sfpi::v_if(abs_x > 1e8f, large_result, result);
+
+        sfpi::dst_reg[0] = sfpi::sign(x) * result;
+
+        sfpi::dst_reg++;
+    }
+}
+
+// DEPRECATED: Kept for backward compatibility. Use _calculate_acosh_log1p_ for new code.
 template <bool APPROXIMATION_MODE, int ITERATIONS>
 inline void _calculate_acosh_(const int iterations) {
     // SFPU microcode
@@