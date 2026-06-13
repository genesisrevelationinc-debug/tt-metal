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
 #include "ckernel_sfpu_recip.h"
 #include "ckernel_sfpu_sqrt.h"
 #include "sfpi.h"
@@ -15,6 +16,7 @@
 using namespace sfpi;
 
 namespace ckernel {
+
 namespace sfpu {
 
 template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
@@ -22,6 +24,7 @@
     // SFPU microcode
     // New version uses a polynomial which is more accurate
     // than the original Pade approximant.
+    // This is kept for backward compatibility; prefer _calculate_log1p_body_ for small inputs.
     vFloat a = dst_reg[0];
 
     // using 4th order polynomial, max error: 1.5e-5%
@@ -41,6 +44,7 @@
     // SFPU microcode
     // New version uses a polynomial which is more accurate
     // than the original Pade approximant.
+    // This is kept for backward compatibility; prefer _calculate_log1p_body_ for small inputs.
     vFloat a = dst_reg[0];
 
     // using 4th order polynomial, max error: 1.5e-5%
@@ -58,6 +62,7 @@
     // SFPU microcode
     // New version uses a polynomial which is more accurate
     // than the original Pade approximant.
+    // This is kept for backward compatibility; prefer _calculate_log1p_body_ for small inputs.
     vFloat a = dst_reg[0];
 
     // using 4th order polynomial, max error: 1.5e-5%
@@ -75,6 +80,7 @@
     // SFPU microcode
     // New version uses a polynomial which is more accurate
     // than the original Pade approximant.
+    // This is kept for backward compatibility; prefer _calculate_log1p_body_ for small inputs.
     vFloat a = dst_reg[0];
 
     // using 4th order polynomial, max error: 1.5e-5%
@@ -92,6 +98,7 @@
     // SFPU microcode
     // New version uses a polynomial which is more accurate
     // than the original Pade approximant.
+    // This is kept for backward compatibility; prefer _calculate_log1p_body_ for small inputs.
     vFloat a = dst_reg[0];
 
     // using 4th order polynomial, max error: 1.5e-5%
@@ -109,6 +116,7 @@
     // SFPU microcode
     // New version uses a polynomial which is more accurate
     // than the original Pade approximant.
+    // This is kept for backward compatibility; prefer _calculate_log1p_body_ for small inputs.
     vFloat a = dst_reg[0];
 
     // using 4th order polynomial, max error: 1.5e-5%
@@ -126,6 +134,7 @@
     // SFPU microcode
     // New version uses a polynomial which is more accurate
     // than the original Pade approximant.
+    // This is kept for backward compatibility; prefer _calculate_log1p_body_ for small inputs.
     vFloat a = dst_reg[0];
 
     // using 4th order polynomial, max error: 1.5e-5%
@@ -143,6 +152,7 @@
     // SFPU microcode
     // New version uses a polynomial which is more accurate
     // than the original Pade approximant.
+    // This is kept for backward compatibility; prefer _calculate_log1p_body_ for small inputs.
     vFloat a = dst_reg[0];
 
     // using 4th order polynomial, max error: 1.5e-5%
@@ -160,6 +170,7 @@
     // SFPU microcode
     // New version uses a polynomial which is more accurate
     // than the original Pade approximant.
+    // This is kept for backward compatibility; prefer _calculate_log1p_body_ for small inputs.
     vFloat a = dst_reg[0];
 
     // using 4th order polynomial, max error: 1.5e-5%
@@ -177,6 +188,7 @@
     // SFPU microcode
     // New version neighboring a polynomial which is more accurate
     // than the original Pade approximant.
+    // This is kept for backward compatibility; prefer _calculate_log1p_body_ for small inputs.
     vFloat a = dst_reg[0];
 
     // using 4th order polynomial, max error: 1.5e-5%
@@ -194,6 +206,7 @@
     // SFPU microcode
     // New version uses a polynomial which is more accurate
     // than the original Pade approximant.
+    // This is kept for backward compatibility; prefer _calculate_log1p_body_ for small inputs.
     vFloat a = dst_reg[0];
 
     // using 4th order polynomial, max error: 1.5e-5%
@@ -211,6 +224,7 @@
     // SFPU microcode
     // New version uses a polynomial which is more accurate
     // than the original Pade approximant.
+    // This is kept for backward compatibility; prefer _calculate_log1p_body_ for small inputs.
     vFloat a = dst_reg[0];
 
     // using 4th order polynomial, max error: 1.5e-5%
@@ -228,6 +242,7 @@
     // SFPU microcode
     // New version uses a polynomial which is more accurate
     // than the original Pade approximant.
+    // This is kept for backward compatibility; prefer _calculate_log1p_body_ for small inputs.
     vFloat a = dst_reg[0];
 
     // using 4th order polynomial, max error: 1.5e-5%
@@ -245,6 +260,7 @@
     // SFPU microcode
     // New version uses a polynomial which is more accurate
     // than the original