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
 #include "noc_nonblocking_api.h"
 #include <sfpi.h>
@@ -15,6 +16,7 @@ using namespace sfpi;
 namespace ckernel {
 namespace sfpu {
 
+// Forward declarations
 template <bool APPROXIMATION DOLPHIN, int ITERATIONS = 8>
 inline void _calculate_cosine_();
 template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
@@ -25,6 +27,8 @@ template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
 inline void _calculate_atan_();
 template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
 inline void _calculate_atan2_();
+template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
+inline void _calculate_log1p_();
 
 // Taylor series for sin/cos
 //  - use symmetery to get into range [0, pi/2]
@@ -261,6 +265,7 @@ sfpi_inline vFloat _calculate_sqrt_body_(vFloat val)
     return val;
 }
 
+// DEPRECATED: Use _calculate_log_body_no_init_ from ckernel_sfpu_log.h instead
 template <bool APPROXIMATION_MODE>
 sfpi_inline vFloat _calculate_log_body_no_init_(vFloat in)
 {
@@ -293,6 +298,7 @@ sfpi_inline vFloat _calculate_log_body_no_init_(vFloat in)
     return vConst1 + (x * (vConst1 + x * partial));
 }
 
+// DEPRECATED: Use _calculate_log_body_ from ckernel_sfpu_log.h instead
 template <bool APPROXIMATION_MODE>
 sfpi_inline vFloat _calculate_log_body_(vFloat in)
 {
@@ -302,6 +308,7 @@ sfpi_inline vFloat _calculate_log_body_(vFloat in)
     return _calculate_log_body_no_init_<APPROXIMATION_MODE>(in);
 }
 
+// DEPRECATED: Use _calculate_log_ from ckernel_sfpu_log.h instead
 template <bool APPROXIMATION_MODE, int ITERATIONS>
 inline void _calculate_log_()
 {
@@ -316,6 +323,7 @@ inline void _calculate_log_()
     }
 }
 
+// DEPRECATED: Use _calculate_log_with_base_ from ckernel_sfpu_log.h instead
 template <bool APPROXIMATION_MODE, int ITERATIONS>
 inline void _calculate_log_with_base_()
 {
@@ -333,6 +341,7 @@ inline void _calculate_log_with_base_()
     }
 }
 
+// DEPRECATED: Use _calculate_sigmoid_ from ckernel_sfpu_log.h instead
 template <bool APPROXIMATION_MODE, int ITERATIONS>
 inline void _calculate_sigmoid_()
 {
@@ -348,6 +357,7 @@ inline void _calculate_sigmoid_()
     }
 }
 
+// DEPRECATED: Use _calculate_tanh_ from ckernel_sfpu_log.h instead
 template <bool APPROXIMATION_MODE, int ITERATIONS>
 inline void _calculate_tanh_()
 {
@@ -363,6 +373,7 @@ inline void _calculate_tanh_()
     }
 }
 
+// DEPRECATED: Use _calculate_hardtanh_ from ckernel_sfpu_log.h instead
 template <bool APPROXIMATION_MODE, int ITERATIONS>
 inline void _calculate_hardtanh_()
 {
@@ -383,6 +394,7 @@ inline void _calculate_hardtanh_()
     }
 }
 
+// DEPRECATED: Use _calculate_sign_ from ckernel_sfpu_log.h instead
 template <bool APPROXIMATION_MODE, int ITERATIONS>
 inline void _calculate_sign_()
 {
@@ -399,6 +411,7 @@ inline void _calculate_sign_()
     }
 }
 
+// DEPRECATED: Use _calculate_signbit_ from ckernel_sfpu_log.h instead
 template <bool APPROXIMATION_MODE, int ITERATIONS>
 inline void _calculate_signbit_()
 {
@@ -415,6 +428,7 @@ inline void _calculate_signbit_()
     }
 }
 
+// DEPRECATED: Use _calculate_rsqrt_ from ckernel_sfpu_log.h instead
 template <bool APPROXIMATION_MODE, int ITERATIONS>
 inline void _calculate_rsqrt_()
 {
@@ -431,6 +445,7 @@ inline void _calculate_rsqrt_()
     }
 }
 
+// DEPRECATED: Use _calculate_relu_ from ckernel_sfpu_log.h instead
 template <bool APPROXIMATION_MODE, int ITERATIONS>
 inline void _calculate_relu_()
 {
@@ -447,6 +462,7 @@ inline void _calculate_relu_()
     }
 }
 
+// DEPRECATED: Use _calculate_leaky_relu_ from ckernel_sfpu_log.h instead
 template <bool APPROXIMATION_MODE, int ITERATIONS>
 inline void _calculate_leaky_relu_()
 {
@@ -464,6 +480,7 @@ inline void _calculate_leaky_relu_()
     }
 }
 
+// DEPRECATED: Use _calculate_elu_ from ckernel_sfpu_log.h instead
 template <bool APPROXIMATION_MODE, int ITERATIONS>
 inline void _calculate_elu_()
 {
@@ -483,6 +500,7 @@ inline void _calculate_elu_()
     }
 }
 
+// DEPRECATED: Use _calculate_erf_ from ckernel_sfpu_log.h instead
 template <bool APPROXIMATION_MODE, int ITERATIONS>
 inline void _calculate_erf_()
 {
@@ -499,6 +517,7 @@ inline void _calculate_erf_()
     }
 }
 
+// DEPRECATED: Use _calculate_erfc_ from ckernel_sfpu_log.h instead
 template <bool APPROXIMATION_MODE, int ITERATIONS>
 inline void _calculate_erfc_()
 {
@@ -515,6 +534,7 @@ inline void _calculate_erfc_()
     }
 }
 
+// DEPRECATED: Use _calculate_gelu_ from ckernel_sfpu_log.h instead
 template <bool APPROXIMATION_MODE, int ITERATIONS>
 inline void _calculate_g