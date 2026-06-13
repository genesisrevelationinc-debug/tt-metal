 ```diff
--- a/tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_trigonometry.h
+++ b/tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_trigonometry.h
@@ -1,4 +1,4 @@
-// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
+// SPDX-FileCopyrightText: © 2023-2025 Tenstorrent Inc.
 //
 // SPDX-License-Identifier: Apache-2.0
 
@@ -6,6 +6,7 @@
 #define _CKERNEL_SFPU_TRIGONOMETRY_H_
 
 #include "ckernel_sfpu_recip.h"
+#include "ckernel_sfpu_log.h"
 
 namespace ckernel {
 namespace sfpu {
@@ -13,6 +14,7 @@
 using namespace sfpi;
 
 #define PI (3.14159265358979323846f)
+#define LN2 (0.69314718055994530942f)
 
 template <bool APPROXIMATION_MODE, int ITERATIONS>
 inline void _sfpu_sin_cos_rad_() {
@@ -116,6 +118,7 @@
     }
 }
 
+// DEPRECATED: Kept for compatibility. Use _sfpu_asinh_log1p_ for new code.
 template <bool APPROXIMATION_MODE, int ITERATIONS>
 inline void _sfpu_asinh_() {
     // SFPU microcode
@@ -140,6 +143,7 @@
     }
 }
 
+// DEPRECATED: Kept for compatibility. Use _sfpu_acosh_log1p_ for new code.
 template <bool APPROXIMATION_MODE, int ITERATIONS>
 inline void _sfpu_acosh_() {
     // SFPU microcode
@@ -162,6 +166,7 @@
     }
 }
 
+// DEPRECATED: Kept for compatibility. Use _sfpu_atanh_log1p_ for new code.
 template <bool APPROXIMATION_MODE, int ITERATIONS>
 inline void _sfpu_atanh_() {
     // SFPU microcode
@@ -186,6 +191,168 @@
     }
 }
 
+// Numerically stable asinh using log1p
+// asinh(x) = sign(x) * log1p(x^2 / (1 + sqrt(1 + x^2)))
+// For |x| >= 1: asinh(x) = sign(x) * (log(|x|) + log1p(1/(2*x^2)) + 0.5*ln(2)) ... simplified to avoid overflow
+// For |x| < 1:  asinh(x) = sign(x) * log1p(|x| * (|x| / (1 + sqrt(1 + x^2))))
+template <bool APPROXIMATION_MODE, int ITERATIONS>
+inline void _sfpu_asinh_log1p_() {
+    for (int d = 0; d < ITERATIONS; d++) {
+        vFloat inp = dst_reg[0];
+        vFloat abs_inp = sfpi::abs(inp);
+        vFloat sign = 1.0f;
+        
+        // Extract sign
+        v_if(inp < 0.0f) {
+            sign = -1.0f;
+        }
+        v_endif;
+        
+        // For large |x|, use: asinh(x) ≈ sign(x) * (log(|x|) + log1p(1/(2*x^2)) + 0.5*ln(2))
+        // But we can simplify: asinh(x) = sign(x) * log(|x| + sqrt(x^2 + 1))
+        // For large x: sqrt(x^2 + 1) = |x| * sqrt(1 + 1/x^2) ≈ |x| * (1 + 1/(2*x^2))
+        // So |x| + sqrt(x^2+1) ≈ |x| * (2 + 1/(2*x^2)) = 2|x| * (1 + 1/(4*x^2))
+        // log(2|x| * (1 + 1/(4*x^2))) = log(2|x|) + log1p(1/(4*x^2))
+        // = log(|x|) + ln(2) + log1p(1/(4*x^2))
+        //
+        // For small |x|: asinh(x) = log1p(|x| * |x| / (1 + sqrt(1 + x^2)))
+        // This avoids catastrophic cancellation near 0
+        
+        // Threshold for "large" |x| to avoid overflow in x^2
+        // Use |x| >= 1.0 as threshold (safe, x^2 won't overflow for reasonable values)
+        vFloat result = 0.0f;
+        
+        // Compute x^2
+        vFloat x2 = abs_inp * abs_inp;
+        
+        // Check for large |x| where we need to avoid x^2 overflow
+        // For fp32, overflow happens around 1.84e19 for x^2, but we use a conservative threshold
+        v_if(abs_inp >= 1.0e9f) {
+            // Very large |x|: asinh(x) ≈ sign(x) * (log(|x|) + ln(2))
+            // More precisely: asinh(x) = sign(x) * log(2|x|) for very large x
+            // log(2|x|) = log(|x|) + ln(2)
+            result = _sfpu_log_<APPROXIMATION_MODE, 1>(&abs_inp);
+            result = result + LN2;
+        }
+        v_elseif(abs_inp >= 1.0f) {
+            // Medium large |x|: use log(|x|) + log1p(1/(2*x^2)) + 0.5*ln(2) approximation
+            // But more stably: compute via log1p
+            // asinh(x) = log(|x| + sqrt(x^2+1)) = log(|x| * (1 + sqrt(1 + 1/x^2)))
+            // = log(|x|) + log(1 + sqrt(1 + 1/x^2))
+            // For numerical stability, use:
+            // asinh(x) = log(|x|) + 0.5 * log1p(1/x^2) + correction
+            // Actually, let's use: log1p(|x| - 1 + sqrt(x^2+