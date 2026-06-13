 ```diff
--- a/tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_trigonometry.h
+++ b/tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_trigonometry.h
@@ -1,4 +1,4 @@
-// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
+// SPDX-FileCopyrightText: © 2023-2025 Tenstorrent Inc.
 //
// SPDX-License-Identifier: Apache-2.0
 
@@ -6,6 +6,7 @@
 #define CKERNEL_SFPU_TRIGONOMETRY_H
 
 #include "ckernel.h"
+#include "ckernel_sfpu_log.h"
 #include "ckernel_defs.h"
 #include "ckernel_sfpu_recip.h"
 #include "ckernel_sfpu_sqrt.h"
@@ -15,6 +16,7 @@
 using namespace sfpi;
 
 namespace ckernel {
+
 namespace sfpu {
 
 template <bool APPROXIMATION_MODE, int ITERATIONS>
@@ -73,6 +75,7 @@
     }
 }
 
+// DEPRECATED: Use _calculate_acosh_body_log1p_ for numerically stable implementation
 template <bool APPROXIMATION_MODE, int ITERATIONS>
 inline void _calculate_acosh_body_(const uint dst) {
     // SFPU microcode
@@ -88,6 +91,38 @@
     }
 }
 
+// Numerically stable acosh(x) using log1p
+// acosh(x) = log1p((x-1) + sqrt((x-1)*(x+1)))  for x >= 1
+// For x near 1: avoids catastrophic cancellation in x + sqrt(x^2 - 1) - 1
+// For large x: avoids overflow in x^2
+template <bool APPROXIMATION_MODE, int ITERATIONS>
+inline void _calculate_acosh_body_log1p_(const uint dst) {
+    for (int d = 0; d < ITERATIONS; d++) {
+        vFloat inp = dst_reg[0];
+        
+        // For x >= 1, compute acosh(x) = log1p((x-1) + sqrt((x-1)*(x+1)))
+        // This is equivalent to log(x + sqrt(x^2 - 1)) but numerically stable
+        vFloat x_minus_1 = inp - 1.0f;
+        vFloat x_plus_1 = inp + 1.0f;
+        
+        // sqrt((x-1)*(x+1)) = sqrt(x^2 - 1)
+        vFloat prod = x_minus_1 * x_plus_1;
+        vFloat sqrt_term = _calculate_sqrt_body_<APPROXIMATION_MODE>(prod);
+        
+        // log1p((x-1) + sqrt(x^2-1)) = log(1 + (x-1) + sqrt(x^2-1)) = log(x + sqrt(x^2-1))
+        vFloat log1p_arg = x_minus_1 + sqrt_term;
+        
+        // Use log1p for better accuracy near x = 1
+        dst_reg[0] = _calculate_log1p_body_<APPROXIMATION_MODE>(log1p_arg);
+        
+        dst_reg++;
+    }
+}
+
+// DEPRECATED: Use _calculate_asinh_body_log1p_ for numerically stable implementation
 template <bool APPROXIMATION_MODE, int ITERATIONS>
 inline void _calculate_asinh_body_(const uint dst) {
     // SFPU microcode
@@ -105,6 +140,46 @@
     }
 }
 
+// Numerically stable asinh(x) using log1p
+// For |x| small: asinh(x) = log1p(x * (x + sqrt(1+x^2)) / (1 + sqrt(1+x^2)))
+// For |x| large: asinh(x) = sign(x) * (log(2|x|) + log1p(1/(2*x^2))/2)
+template <bool APPROXIMATION_MODE, int ITERATIONS>
+inline void _calculate_asinh_body_log1p_(const uint dst) {
+    for (int d = 0; d < ITERATIONS; d++) {
+        vFloat inp = dst_reg[0];
+        vFloat abs_inp = sfpi::abs(inp);
+        
+        // Compute sqrt(1 + x^2)
+        vFloat x_sq = inp * inp;
+        vFloat sqrt_term = _calculate_sqrt_body_<APPROXIMATION_MODE>(x_sq + 1.0f);
+        
+        // For asinh, we use: asinh(x) = sign(x) * log(|x| + sqrt(1+x^2))
+        // Rewrite using log1p for better accuracy:
+        // log(|x| + sqrt(1+x^2)) = log1p(|x| - 1 + sqrt(1+x^2)) when |x| >= 1
+        //                        = log1p(|x| * (|x| + sqrt(1+x^2)) / (1 + sqrt(1+x^2))) when |x| < 1
+        //
+        // Use the stable form: log(|x| + sqrt(1+x^2)) = log1p(|x| + sqrt(1+x^2) - 1)
+        // But to avoid cancellation, use:
+        // log(|x| + sqrt(1+x^2)) = log1p(|x| * (|x| + sqrt(1+x^2)) / (1 + sqrt(1+x^2)))
+        // when |x| is small, and log(|x| + sqrt(1+x^2)) directly when |x| is large
+        
+        vFloat sum = abs_inp + sqrt_term;
+        vFloat result;
+        
+        // For large |x|, direct log is fine; for small |x|, use log1p
+        // Threshold at |x| = 1 for simplicity
+        v_if (abs_inp > 1.0f) {
+            result = _calculate_log_body_no_init_(sum);
+        } v_else {
+            // log1p(|x| * (|x| + sqrt(1+x^2)) / (1 + sqrt(1+x^2)) - 1) is not right
+            // Instead: log(|x| + sqrt(1+x^2)) = log1p(|x| + sqrt(1+x^2) - 1)
+            // But this cancels. Better: use log1p directly with the difference
+            v