 ```diff
--- a/tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_trigonometry.h
+++ b/tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_trigonometry.h
@@ -1,4 +1,4 @@
-// SPDX-FileCopyrightText: © 2024 Tenstorrent AI
+// SPDX-FileCopyrightText: © 2024 Tenstorrent AI
 //
 // SPDX-License-Identifier: Apache-2.0
 
@@ -6,6 +6,7 @@
 #include "ckernel.h"
 #include "ckernel_defs.h"
 #include "ckernel_sfpu_log.h"
+#include "ckernel_sfpu_log1p.h"
 #include "ckernel_sfpu_recip.h"
 #include "ckernel_sfpu_sqrt.h"
 
@@ -15,6 +16,7 @@
 using namespace sfpi;
 
 #include "sfpu/ckernel_sfpu_log.h"
+#include "sfpu/ckernel_sfpu_log1p.h"
 #include "sfpu/ckernel_sfpu_recip.h"
 #include "sfpu/ckernel_sfpu_sqrt.h"
 
@@ -22,6 +24,7 @@
 
 namespace ckernel {
 namespace sfpu {
+
 template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
 inline void calculate_atanh() {
     // SFPU microcode
@@ -29,20 +32,22 @@
         TTI_SFPLOAD(0, 3, 3, 0);
         TTI_SFPMUL(0, 0, 9, 1, 0);
         TTI_SFPSTORE(1, 0, 3, 0);  // Store intermediate result in lreg 0
-        // atanh(x) = 0.5 * ln((1 + x) / (1 - x))
-        // num = 1 + x
-        // den = 1 - x
-        // tmp = num / den
-        // result = 0.5 * ln(tmp)
-        sfpi::vFloat num = sfpi::vConst1 + sfpi::dst_reg[0];
-        sfpi::vFloat den = sfpi::vConst1 - sfpi::dst_reg[0];
-        sfpi::vFloat tmp = _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(den);
-        num = num * den;
-        den = _calculate_log_body_no_init_(num);
-        auto res = 0.5f * den;
-        sfpi::dst_reg[0] = res;
+        // Numerically stable atanh(x) using log1p:
+        // atanh(x) = 0.5 * log((1+x)/(1-x))
+        //          = 0.5 * log1p(2x/(1-x))
+        // For small x: 2x/(1-x) ≈ 2x, so log1p(2x/(1-x)) ≈ log1p(2x)
+        // This avoids catastrophic cancellation near x = 0
+        sfpi::vFloat x = sfpi::dst_reg[0];
+        sfpi::vFloat one_minus_x = sfpi::vConst1 - x;
+        // Compute 2x / (1 - x) using reciprocal
+        sfpi::vFloat recip = _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(one_minus_x);
+        sfpi::vFloat arg = 2.0f * x * recip;
+        // Use log1p for numerical stability
+        sfpi::vFloat result = _calculate_log1p_body_(arg);
+        // Fold 0.5 into the result
+        sfpi::dst_reg[0] = 0.5f * result;
         TTI_SFPSTORE(0, 3, 3, 0);
         sfpi::dst_reg++;
     }
@@ -53,14 +58,24 @@
     // SFPU microcode
     for (int d = 0; d < ITERATIONS; d++) {
         TTI_SFPLOAD(0, 3, 3, 0);
-        // asinh(x) = ln(x + sqrt(x^2 + 1))
-        sfpi::vFloat tmp = sfpi::dst_reg[0] * sfpi::dst_reg[0] + sfpi::vConst1;
-        tmp = _calculate_sqrt_body_<APPROXIMATION_MODE>(tmp);
-        tmp = tmp + sfpi::abs(sfpi::dst_reg[0]);
-        auto res = _calculate_log_body_no_init_(tmp);
-        // restore sign
-        v_if(sfpi::dst_reg[0] < 0.0f) { res = -res; }
+        // Numerically stable asinh(x) using log1p:
+        // asinh(x) = sign(x) * log(|x| + sqrt(x^2 + 1))
+        // For small x: use log1p-based formulation to avoid cancellation
+        // For large x: use sign(x) * (log(2|x|) + log1p(1/(2x^2)))
+        sfpi::vFloat x = sfpi::dst_reg[0];
+        sfpi::vFloat abs_x = sfpi::abs(x);
+        sfpi::vFloat x2 = x * x;
+        // sqrt(x^2 + 1) = |x| * sqrt(1 + 1/x^2) for |x| >= 1, or use direct for small
+        // Use the identity: asinh(x) = sign(x) * log1p(|x| + x^2 / (sqrt(x^2+1) + |x|))
+        // Simpler stable form: log(|x| + sqrt(x^2+1)) = log1p(|x| - 1 + sqrt(x^2+1)) for |x| near 1
+        // Most stable: log1p(x^2 / (|x| + sqrt(x^2+1))) + log(|x|) for large, but let's use:
+        // asinh(x) = sign(x) * log1p(|x| * (1 + sqrt(1 + 1/x^2))) for |x| >= 1
+        // For general case, use: log1p(x^2 / (1 + sqrt(1 + x^2))) + log(|x|) ... no
+        // Clean approach: compute sqrt(x^2+1), then use log1p(|x| + sqrt(x^2+1) - 1) when appropriate
