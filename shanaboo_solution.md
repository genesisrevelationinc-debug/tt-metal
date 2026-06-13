 ```diff
--- a/tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_trigonometry.h
+++ b/tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_trigonometry.h
@@ -1,4 +1,4 @@
-// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
+// SPDX-FileCopyrightText: © 2023-2025 Tenstorrent Inc.
 //
 // SPDX-License-Identifier: Apache-2.0
 
@@ -9,6 +9,7 @@
 #include "ckernel_sfpu_recip.h"
 #include "ckernel_sfpu_sqrt.h"
 #include "ckernel_sfpu_log.h"
+#include "ckernel_sfpu_log1p.h"
 
 using namespace sfpi;
 
@@ -116,6 +117,7 @@
     }
 }
 
+// Numerically stable atanh using log1p: atanh(x) = 0.5 * log1p(2*x/(1-x)) for x >= 0, with sign restoration
 template <bool APPROXIMATION_MODE, int ITERATIONS>
 inline void _calculate_atanh_(const int iterations)
 {
@@ -123,22 +125,32 @@
     {
         vFloat v = dst_reg[0];
 
-        // atanh(x) = 0.5 * ln((1+x)/(1-x))
-        vFloat num = vConst1 + v;
-        vFloat den = vConst1 - v;
-        vFloat tmp = _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(den);
-        num = num * den;
-        den = _calculate_log_body_no_init_(num);
-        v_if (v < 0.0f) {
-            den = -den;
-        }
-        v_endif;
-        v = 0.5f * den;
+        // Compute 2*x / (1 - x) in a numerically stable way
+        // For atanh, we use: atanh(x) = 0.5 * log1p(2*x/(1-x))
+        // Handle sign separately to avoid branch mispredictions
+        vFloat abs_v = sfpi::abs(v);
+        vFloat sign = 1.0f;
+        v_if (v < 0.0f) {
+            sign = -1.0f;
+        }
+        v_endif;
+
+        // For |x| near 1, 1-x underflows; use 1-abs_v for positive branch
+        vFloat one_minus_x = vConst1 - abs_v;
+        vFloat two_x = 2.0f * abs_v;
+        vFloat ratio = two_x / one_minus_x;
+
+        // log1p(ratio) gives us log(1 + 2|x|/(1-|x|)) = log((1+|x|)/(1-|x|))
+        vFloat log1p_result = _calculate_log1p_body_(ratio);
+
+        // Apply 0.5 * sign * result
+        v = 0.5f * sign * log1p_result;
+
         dst_reg[0] = v;
     }
 }
 
+// Numerically stable asinh using log1p: asinh(x) = sign(x) * log1p(|x| + x^2/(1+sqrt(1+x^2)))
 template <bool APPROXIMATION_MODE, int ITERATIONS>
 inline void _calculate_asinh_(const int iterations)
 {
@@ -146,16 +158,30 @@
     {
         vFloat v = dst_reg[0];
 
-        // asinh(x) = ln(|x| + sqrt(x^2 + 1))
-        vFloat tmp = v * v + vConst1;
-        tmp = _calculate_sqrt_body_<APPROXIMATION_MODE>(tmp);
-        tmp = tmp + sfpi::abs(v);
-        auto res = _calculate_log_body_no_init_(tmp);
-        v_if (v < 0.0f) {
-            res = -res;
-        }
-        v_endif;
+        // Extract sign
+        vFloat sign = 1.0f;
+        v_if (v < 0.0f) {
+            sign = -1.0f;
+        }
+        v_endif;
+
+        vFloat abs_v = sfpi::abs(v);
+
+        // asinh(x) = sign(x) * log(|x| + sqrt(x^2 + 1))
+        // For numerical stability, use:
+        // log(|x| + sqrt(x^2+1)) = log1p(|x| + sqrt(x^2+1) - 1)
+        // Better: use the identity asinh(x) = sign(x) * log1p(|x| * (|x| / (1 + sqrt(1+x^2))))
+        // Or more directly: asinh(x) = sign(x) * log1p(|x| + |x|^2/(1+sqrt(1+|x|^2)))
+
+        vFloat x2 = abs_v * abs_v;
+        vFloat sqrt_term = _calculate_sqrt_body_<APPROXIMATION_MODE>(x2 + vConst1);
+
+        // Compute |x| + sqrt(x^2+1) - 1 = |x| + (sqrt(x^2+1) - 1)
+        // sqrt(x^2+1) - 1 = x^2 / (1 + sqrt(x^2+1)) for stability
+        vFloat sqrt_minus_1 = x2 / (vConst1 + sqrt_term);
+        vFloat arg = abs_v + sqrt_minus_1;
+
+        vFloat res = sign * _calculate_log1p_body_(arg - abs_v);  // log1p of the small correction
 
         dst_reg[0] = res;
     }
@@ -167,14 +193,30 @@
     {
         vFloat v = dst_reg[0];
 
-        // acosh(x) = log(x + sqrt(x^2 - 1))
-        vFloat tmp = v * v;
-        tmp = tmp - vConst1;
-        tmp = _calculate_sqrt_body_<APPROXIMATION_MODE>(tmp);
-        tmp = tmp + v;
-        v = _calculate_log_body_no_init_(tmp);
-        dst_reg[0] = v;
+        // acosh(x) = log(x + sqrt(x^2 - 1)) for x >= 1
+        // For numerical stability near x = 1:
+        // sqrt(x^2 - 1) = sqrt((x-1)(x+1))
+       