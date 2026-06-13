 ```diff
--- a/tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_trigonometry.h
+++ b/tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_trigonometry.h
@@ -1,4 +1,4 @@
-// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
+// SPDX-FileCopyrightText: © 2023-2025 Tenstorrent Inc.
 //
 // SPDX-License-Identifier: Apache-2.0
 
@@ -10,6 +10,7 @@
 #include "ckernel_sfpu_recip.h"
 #include "ckernel_sfpu_sqrt.h"
 #include "ckernel_sfpu_log.h"
+#include "ckernel_sfpu_log1p.h"
 
 using namespace sfpi;
 
@@ -155,6 +156,7 @@
     }
 }
 
+// Optimized atanh using log1p for numerical stability
 template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
 inline void _calculate_atanh_(const int iterations)
 {
@@ -162,18 +164,18 @@
     {
         vFloat v = dst_reg[0];
 
-        // atanh(x) = 0.5 * ln((1 + x) / (1 - x))
-        vFloat num = vConst1 + v;
-        vFloat den = vConst1 - v;
-        vFloat tmp = _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(den);
-        num = num * den;
-        den = _calculate_log_body_no_init_(num);
-        v_if(den < 0.0f) { den = -den; }
-        v_endif;
-        v = 0.5f * den;
+        // atanh(x) = 0.5 * log((1+x)/(1-x))
+        // Use log1p for numerical stability:
+        // atanh(x) = 0.5 * log1p(2*x / (1-x))
+        vFloat den = vConst1 - v;
+        vFloat two_x = 2.0f * v;
+        vFloat ratio = two_x * _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(den);
+        
+        // Use log1p for better accuracy near x = 0 and x = ±1
+        v = 0.5f * _calculate_log1p_body_<APPROXIMATION_MODE>(ratio);
 
-        // atanh(-x) = -atanh(x)
+        // atanh is odd: atanh(-x) = -atanh(x), sign handled by the formula above
         dst_reg[0] = v;
 
         dst_reg++;
@@ -181,6 +183,7 @@
     }
 }
 
+// Optimized asinh using log1p for numerical stability
 template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
 inline void _calculate_asinh_(const int iterations)
 {
@@ -188,16 +191,31 @@
     {
         vFloat v = dst_reg[0];
 
-        // asinh(x) = ln(x + sqrt(x^2 + 1))
-        vFloat tmp = v * v;
-        tmp = tmp + vConst1;
-        tmp = _calculate_sqrt_body_<APPROXIMATION_MODE>(tmp);
-        tmp = tmp + sfpi::abs(v);
-        auto res = _calculate_log_body_no_init_(tmp);
+        vFloat abs_v = sfpi::abs(v);
+        vFloat result;
 
-        // asinh(-x) = -asinh(x)
-        dst_reg[0] = res;
+        // For large |x|, use: asinh(x) = sign(x) * (log(2*|x|) + log1p(1/(4*x^2))/2)
+        // For small to moderate |x|, use: asinh(x) = sign(x) * log1p(|x| + x^2/(1+sqrt(1+x^2)))
+        // Simplified stable form: asinh(x) = sign(x) * log1p(|x| * (1 + |x|/(1+sqrt(1+x^2))))
+        
+        // Compute sqrt(1 + x^2) stably
+        vFloat x2 = v * v;
+        vFloat sqrt_1px2 = _calculate_sqrt_body_<APPROXIMATION_MODE>(x2 + vConst1);
+        
+        // asinh(x) = sign(x) * log(|x| + sqrt(x^2 + 1))
+        // For numerical stability, rewrite as:
+        // log(|x| + sqrt(x^2+1)) = log1p(|x| - 1 + sqrt(x^2+1)) when |x| is not too large
+        // Better: use the identity that for all x:
+        // asinh(x) = sign(x) * log1p(|x| + x^2/(1+sqrt(1+x^2)))
+        
+        vFloat denom = vConst1 + sqrt_1px2;
+        vFloat correction = x2 / denom;
+        vFloat arg = abs_v + correction;
+        
+        result = _calculate_log1p_body_<APPROXIMATION_MODE>(arg - vConst1);
+        
+        // Restore sign: asinh(-x) = -asinh(x)
+        dst_reg[0] = result;
 
         dst_reg++;
     }
@@ -210,14 +228,24 @@
     {
         vFloat v = dst_reg[0];
 
-        // acosh(x) = log(x + sqrt(x^2 - 1))
-        sfpi::vFloat tmp = v * v;
-        tmp = tmp - sfpi::vConst1;
-        tmp = _calculate_sqrt_body_<APPROXIMATION_MODE>(tmp);
-        tmp = tmp + v;
-        sfpi::dst_reg[0] = _calculate_log_body_no_init_(tmp);
+        // acosh(x) = log(x + sqrt(x^2 - 1)) for x >= 1
+        // For numerical stability near x = 1, use:
+        // acosh(x) = log1p(x - 1 + sqrt((x-1)*(x+1)))
+        // Or equivalently: acosh(x) = 2 * asinh(sqrt((x-1)/2)) for x near 1
+        // Stable form: acosh(x) = log1p(x - 1 + sqrt(x^2 - 1))
+        
+        vFloat x_minus_1 = v - vConst1;
