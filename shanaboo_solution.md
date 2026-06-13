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
 #include "sfpi.h"
+#include "ckernel_sfpu_log.h"

 using namespace sfpi;

@@ -17,6 +18,9 @@ namespace ckernel {
 namespace sfpu {

 //////////////////////////////////////////////////////////////////////////////
+// Forward declarations for log1p-based helpers
+//////////////////////////////////////////////////////////////////////////////
+template <bool APPROXIMATION_MODE, int ITERATIONS>
+sfpi_inline vFloat _calculate_log1p_body_(vFloat x);

 //////////////////////////////////////////////////////////////////////////////
 //                      Sine/Cosine (using range reduction)
@@ -168,6 +172,7 @@ sfpi_inline vFloat _calculate_tanh_body_(vFloat val)
 //                      ATANH
 //////////////////////////////////////////////////////////////////////////////
 template <bool APPROXIMATION_MODE, int ITERATIONS>
+[[deprecated("Use _calculate_atanh_log1p_body_ for numerically stable implementation")]]
 sfpi_inline vFloat _calculate_atanh_body_(vFloat in)
 {
     // atanh(x) = 0.5 * ln((1 + x) / (1 - x))
@@ -181,6 +186,24 @@ sfpi_inline vFloat _calculate_atanh_body_(vFloat in)
     return 0.5f * result;
 }

+template <bool APPROXIMATION_MODE, int ITERATIONS>
+sfpi_inline vFloat _calculate_atanh_log1p_body_(vFloat in)
+{
+    // Numerically stable atanh using log1p:
+    // atanh(x) = 0.5 * log((1+x)/(1-x))
+    //          = 0.5 * log1p(2x/(1-x))
+    // For |x| near 1: compute 2x/(1-x) directly, then log1p
+    // For x near 0: log1p handles the small argument accurately
+    vFloat num = in + in;                    // 2x
+    vFloat den = vConst1 - in;               // 1 - x
+    
+    // Compute 2x/(1-x) using reciprocal
+    vFloat recip_den = _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(den);
+    vFloat arg = num * recip_den;            // 2x/(1-x)
+    
+    // atanh(x) = 0.5 * log1p(2x/(1-x))
+    vFloat result = _calculate_log1p_body_<APPROXIMATION_MODE, ITERATIONS>(arg);
+    return 0.5f * result;
+}
+
 //////////////////////////////////////////////////////////////////////////////
 //                      ASINH
 //////////////////////////////////////////////////////////////////////////////
@@ -195,6 +218,7 @@ sfpi_inline vFloat _calculate_asinh_ body_(vFloat in)
     // asinh(x) = ln(x + sqrt(x^2 + 1))
     vFloat result = in * in;
     result = result + vConst1;
+    // For large |x|, x^2 + 1 overflows; use log1p-based formulation instead
     result = _calculate_sqrt_body_<APPROXIMATION_MODE>(result);
     result = result + in_abs;
     result = _calculate_log_body_no_init_<APPROXIMATION_MODE, ITERATIONS>(result);
@@ -206,6 +230,49 @@ sfpi_inline vFloat _calculate_asinh_ body_(vFloat in)
     return result;
 }

+template <bool APPROXIMATION_MODE, int ITERATIONS>
+sfpi_inline vFloat _calculate_asinh_log1p_body_(vFloat in)
+{
+    // Numerically stable asinh using log1p:
+    // asinh(x) = sign(x) * asinh(|x|)
+    // For |x| <= 1: asinh(x) = log1p(x^2 / (1 + sqrt(1 + x^2)))
+    // For |x| > 1:  asinh(x) = log(2|x|) + log1p(1/(4x^2)) / 2  (asymptotic)
+    // Simpler unified form using log1p for the small-argument case:
+    // asinh(x) = sign(x) * log1p(|x| * |x| / (1 + sqrt(1 + |x|^2)))  for moderate |x|
+    // For large |x|: asinh(x) ≈ sign(x) * (log(|x|) + log(2))
+    
+    vFloat in_abs = sfpi::abs(in);
+    vFloat result;
+    
+    // For |x| >= 1, use: asinh(x) = log(|x| + sqrt(|x|^2 + 1))
+    // Rewrite: sqrt(x^2 + 1) = |x| * sqrt(1 + 1/x^2)
+    // For large |x|: asinh(x) ≈ sign(x) * log(2|x|) = sign(x) * (log(|x|) + ln(2))
+    // Use threshold to avoid overflow: |x| > 1e18 is "large"
+    vFloat x2 = in_abs * in_abs;
+    
+    // Compute sqrt(1 + x^2) using log1p for accuracy when |x| is small
+    // For the main formula, we use:
+    // asinh(x) = log(|x| + sqrt(x^2 + 1))
+    // Let t = |x|, then:
+    // asinh(x) = log(t + sqrt(t^2 + 1))
+    //          = log1p(t + sqrt(t^2 + 1) - 1)
+    // For t near 0: use log1p(t^2 / (1 + sqrt(1 + t^2))) directly
+    // For general t: compute via log1p(t + sqrt(t^2+1) - 1)
+    
+    vFloat sqrt_term = _calculate_sqrt_body_<APPROXIMATION_MODE>(x2 + vConst1);
+    vFloat sum = in_abs + sqrt_term;
+    
+    // For accuracy near 0, use log1p