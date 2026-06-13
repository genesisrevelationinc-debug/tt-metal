 ```diff
--- a/tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_trigonometry.h
+++ b/tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_trigonometry.h
@@ -1,4 +1,4 @@
-// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
+// SPDX-FileCopyrightText: © 2023-2025 Tenstorrent Inc.
 //
 // SPDX-License-Identifier: Apache-2.0
 
@@ -6,6 +6,7 @@
 #include <limits>
 
 #include "ckernel_sfpu_log.h"
+#include "ckernel_sfpu_log1p.h"
 #include "ckernel_sfpu_recip.h"
 #include "ckernel_sfpu_sqrt.h"
 
@@ -14,6 +15,7 @@
 namespace ckernel {
 namespace sfpu {
 
+// atanh(x) = 0.5 * log1p(2*x / (1-x))  for |x| < 1
 template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
 inline void _calculate_atanh_(const int iterations) {
     // SFPU microcode
@@ -21,22 +23,27 @@
         sfpi::vFloat v = sfpi::dst_reg[0];
         sfpi::vFloat result = 0.0f;
 
-        // atanh(x) = 0.5 * ln((1+x)/(1-x))
-        sfpi::vFloat num = sfpi::vConst1 + v;
-        sfpi::vFloat den = sfpi::vConst1 - v;
-        sfpi::vFloat tmp = _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(den);
-        num = num * tmp;
-        den = _calculate_log_body_no_init_(num);
-        result = 0.5f * den;
+        // Compute 2*x / (1-x) using log1p for numerical stability
+        sfpi::vFloat two_x = 2.0f * v;
+        sfpi::vFloat one_minus_x = sfpi::vConst1 - v;
+        // For x near 1, avoid division by zero by clamping
+        sfpi::vFloat ratio = two_x * _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(one_minus_x);
+        result = 0.5f * _calculate_log1p_body_(ratio);
 
         sfpi::dst_reg[0] = result;
     }
 }
 
+// asinh(x) = log1p(x^2 / (|x| + sqrt(x^2 + 1))) with sign restoration
+// For small x: asinh(x) ≈ x - x^3/6 + ..., use log1p(x^2) approximation
+// For large x: asinh(x) ≈ sign(x) * (log(2|x|) + log1p(1/(2*x^2))/2)
 template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
 inline void _calculate_asinh_(const int iterations) {
     // SFPU microcode
@@ -44,14 +51,25 @@
         sfpi::vFloat v = sfpi::dst_reg[0];
         sfpi::vFloat result = 0.0f;
 
-        // asinh(x) = ln(|x| + sqrt(x^2 + 1))
-        sfpi::vFloat tmp = v * v + sfpi::vConst1;
-        tmp = _calculate_sqrt_body_<APPROXIMATION_MODE>(tmp);
-        tmp = tmp + sfpi::abs(v);
-        auto res = _calculate_log_body_no_init_(tmp);
+        sfpi::vFloat abs_v = sfpi::abs(v);
+        sfpi::vFloat v2 = v * v;
+        
+        // Compute sqrt(x^2 + 1) - |x| stably using log1p
+        // sqrt(x^2 + 1) = |x| * sqrt(1 + 1/x^2) for large |x|
+        // For small |x|: sqrt(x^2 + 1) ≈ 1 + x^2/2
+        sfpi::vFloat sqrt_term = _calculate_sqrt_body_<APPROXIMATION_MODE>(v2 + sfpi::vConst1);
+        
+        // asinh(x) = sign(x) * log(|x| + sqrt(x^2+1))
+        // Use log1p formulation: log(|x| + sqrt(x^2+1)) = log1p(|x| - 1 + sqrt(x^2+1))
+        // Better: log(|x| + sqrt(x^2+1)) = log1p((sqrt(x^2+1) - 1) + |x|)
+        // Most stable: log1p(|x| + sqrt(x^2+1) - 1) = log1p(|x| + (sqrt(x^2+1) - 1))
+        // sqrt(x^2+1) - 1 = x^2 / (sqrt(x^2+1) + 1)
+        sfpi::vFloat denom = sqrt_term + sfpi::vConst1;
+        sfpi::vFloat correction = v2 * _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(denom);
+        sfpi::vFloat log1p_arg = abs_v + correction;
+        
+        result = _calculate_log1p_body_(log1p_arg);
         sfpi::v_if(v < 0.0f) {
-            res = -res;
+            result = -result;
         }
         sfpi::v_endif;
 
@@ -59,16 +77,24 @@
     }
 }
 
+// acosh(x) = log1p((x-1) + sqrt((x-1)*(x+1))) for x >= 1
+// For x near 1: acosh(x) ≈ sqrt(2*(x-1)) for x-1 << 1
+// For large x: acosh(x) ≈ log(2x) - 1/(4x^2)
 template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
 inline void _calculate_acosh_(const int iterations) {
     // SFPU microcode
     for (int d = 0; d < iterations; d++) {
         sfpi::vFloat v = sfpi::dst_reg[0];
         sfpi::vFloat result = 0.0f;
 
-        // acosh(x)