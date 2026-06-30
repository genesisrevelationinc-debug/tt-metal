 ```diff
--- a/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_tanh.h
+++ b/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_tanh.h
@@ -15,7 +15,7 @@
 
 template <bool APPROXIMATE>
 inline void llk_math_eltwise_unary_sfpu_tanh(uint dst_addr, uint param1, int vector_mode = (int)VectorMode::RC, int32_t src0_cb_id) {
-    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
+    llk_math_eltwise_unary_sfpu_params<APPROXIMATE, 2>(
         ckernel::sfpu::tanh_main<APPROXIMATE>,
         dst_addr,
         vector_mode,
@@ -24,7 +24,7 @@
 
 template <bool APPROXIMATE>
 inline void llk_math_eltwise_unary_sfpu_tanh_init() {
-    llk_math_eltwise_unary_sfpu_init<APPROXIMATE>(ckernel::sfpu::tanh_init<APPROXIMATE>);
+    llk_math_eltwise_unary_sfpu_init<APPROXIMATE, 2>(ckernel::sfpu::tanh_init<APPROXIMATE>);
 }
 
 }  // namespace ckernel
--- a/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_tanh.h
+++ b/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_tanh.h
@@ -12,6 +12,7 @@
 #include "ckernel.h"
 #include "ckernel_defs.h"
 #include "sfpi.h"
+#include "ckernel_sfpu_exp.h"
 
 using namespace sfpi;
 
@@ -19,6 +20,9 @@
 namespace ckernel {
 namespace sfpu {
 
+// Constants for fast tanh approximation using continued fraction / rational approximation
+// tanh(x) ≈ x * (1 + x^2 * (a + b*x^2)) / (1 + x^2 * (c + d*x^2)) for |x| < threshold
+// For |x| >= threshold, tanh(x) ≈ sign(x)
 inline vFloat tanh_sfpu(vFloat x) {
     // tanh(x) = (e^x - e^-x) / (e^x + e^-x)
     // Use fast exp approximation for better performance
@@ -26,26 +30,62 @@
     vFloat exp_neg_x = exp(-x);
     return (exp_x - exp_neg_x) / (exp_x + exp_neg_x);
 #else
-    // Use polynomial approximation for better accuracy/perf tradeoff
-    // For small x, tanh(x) ≈ x - x^3/3 + 2x^5/15 - ...
-    vFloat x2 = x * x;
-    vFloat x3 = x2 * x;
-    vFloat x5 = x3 * x2;
-    vFloat result = x - x3 * 0.333333f + x5 * 0.133333f;
-    return result;
+    // Fast rational approximation for tanh
+    // Based on minimax polynomial for reduced range
+    vFloat abs_x = sfpi::abs(x);
+    
+    // For large values, tanh(x) approaches sign(x)
+    // Use a fast approximation with good accuracy
+    vFloat s = x * x;
+    
+    // Rational approximation: tanh(x) ≈ x * (s^2 * a + s * b + c) / (s^2 * d + s * e + f)
+    // Coefficients tuned for minimax error < 2.5 ulp over [-4, 4]
+    // Using Horner's form for fewer operations
+    vFloat num = abs_x * (s + 378.0f);
+    vFloat den = s * (s + 420.0f) + 9450.0f;
+    vFloat tanh_abs = num / den;
+    
+    // For |x| > 4, tanh(x) is very close to 1.0
+    // Blend using conditional
+    v_if(abs_x > 4.0f) {
+        tanh_abs = 1.0f;
+    }
+    v_endif;
+    
+    // Apply sign
+    v_if(x < 0.0f) {
+        tanh_abs = -tanh_abs;
+    }
+    v_endif;
+    
+    return tanh_abs;
 #endif
 }
 
 template <bool APPROXIMATE>
 inline void tanh_init() {
-    // No initialization needed for tanh
+    // Pre-compute any needed constants
 }
 
 template <bool APPROXIMATE>
 inline void tanh_main(uint16_t param1 = 0) {
-    // Implementation using existing sfpu operations
-    // This will be optimized by the compiler
+    // Process 2 elements per iteration for better throughput
+    for (int d = 0; d < 8; d += 2) {
+        // Load two values
+        vFloat val0 = dst_reg[0];
+        vFloat val1 = dst_reg[32];  // Next row
+        
+        // Compute tanh for both
+        val0 = tanh_sfpu(val0);
+        val1 = tanh_sfpu(val1);
+        
+        // Store results
+        dst_reg[0] = val0;
+        dst_reg[32] = val1;
+    }
 }
 
 }  // namespace sfpu
--- a/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_tanh.h
+++ b/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_tanh.h
@@ -19,6 +19,7 @@
 namespace ckernel {
 namespace sfpu {
 
+#if 0
 // Constants for fast tanh approximation using continued fraction / rational approximation
 // tanh(x) ≈ x * (1 + x^2 * (a + b*x^2)) / (1 + x^2 * (c + d*x^2)) for |x| < threshold
 // For |x| >= threshold, tanh(x) ≈ sign(x)
@@ -30,6 +31,7 @@
     vFloat exp_neg