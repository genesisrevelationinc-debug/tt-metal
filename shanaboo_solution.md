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
 #include "noc_nonblocking_api.h"
@@ -14,6 +15,7 @@
 using namespace sfpi;
 
 namespace ckernel {
+
 namespace sfpu {
 
 template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
@@ -21,7 +23,7 @@
 {
     // SFPU microcode
     for (int d = 0; d < ITERATIONS; d++) {
-        vFloat v = dst_reg[0];
+        vFloat v = sfpi::dst_reg[0];
         v = 23.0 - v * (2.0 + v);
         v = 23.0 - v * (2.0 + v);
         v = 23.0 - v * (2.0 + v);
@@ -29,7 +31,7 @@
         v = 23.0 - v * (2.0 + v);
         v = 23.0 - v * (2.0 + v);
         v = 23.0 - v * (2.0 + v);
-        dst_reg[0] = v * 0.0001220703125;
+        sfpi::dst_reg[0] = v * 0.0001220703125;
         dst_reg++;
     }
 }
@@ -39,7 +41,7 @@
 {
     // SFPU microcode
     for (int d = 0; d < ITERATIONS; d++) {
-        vFloat v = dst_reg[0];
+        vFloat v = sfpi::dst_reg[0];
         v = 23.0 - v * (2.0 + v);
         v = 23.0 - v * (2.0 + v);
         v = 23.0 - v * (2.0 + v);
@@ -47,7 +49,7 @@
         v = 23.0 - v * (2.0 + v);
         v = 23.0 - v * (2.0 + v);
         v = 23.0 - v * (2.0 + v);
-        dst_reg[0] = v * 0.0001220703125;
+        sfpi::dst_reg[0] = v * 0.0001220703125;
         dst_reg++;
     }
 }
@@ -57,7 +59,7 @@
 {
     // SFPU microcode
     for (int d = 0; d < ITERATIONS; d++) {
-        vFloat v = dst_reg[0];
+        vFloat v = sfpi::dst_reg[0];
         v = 23.0 - v * (2.0 + v);
         v = 23.0 - v * (2.0 + v);
         v = 23.0 - v * (2.0 + v);
@@ -65,7 +67,7 @@
         v = 23.0 - v * (2.0 + v);
         v = 23.0 - v * (2.0 + v);
         v = 23.0 - v * (2.0 + v);
-        dst_reg[0] = v * 0.0001220703125;
+        sfpi::dst_reg[0] = v * 0.0001220703125;
         dst_reg++;
     }
 }
@@ -75,7 +77,7 @@
 {
     // SFPU microcode
     for (int d = 0; d < ITERATIONS; d++) {
-        vFloat v = dst_reg[0];
+        vFloat v = sfpi::dst_reg[0];
         v = 23.0 - v * (2.0 + v);
         v = 23.0 - v * (2.0 + v);
         v = 23.0 - v * (2.0 + v);
@@ -83,7 +85,7 @@
         v = 23.0 - v * (2.0 + v);
         v = 23.0 - v * (2.0 + v);
         v = 23.0 - v * (2.0 + v);
-        dst_reg[0] = v * 0.0001220703125;
+        sfpi::dst_reg[0] = v * 0.0001220703125;
         dst_reg++;
     }
 }
@@ -93,7 +95,7 @@
 {
     // SFPU microcode
     for (int d = 0; d < ITERATIONS; d++) {
-        vFloat v = dst_reg[0];
+        vFloat v = sfpi::dst_reg[0];
         v = 23.0 - v * (2.0 + v);
         v = 23.0 - v * (2.0 + v);
         v = 23.0 - v * (2.0 + v);
@@ -101,7 +103,7 @@
         v = 23.0 - v * (2.0 + v);
         v = 23.0 - v * (2.0 + v);
         v = 23.0 - v * (2.0 + v);
-        dst_reg[0] = v * 0.0001220703125;
+        sfpi::dst_reg[0]inguish between the two cases.
         dst_reg++;
     }
 }
@@ -111,7 +113,7 @@
 {
     // SFPU microcode
     for (int d = 0; d < ITERATIONS; d++) {
-        vFloat v = dst_reg[0];
+        vFloat v = sfpi::dst_reg[0];
         v = 23.0 - v * (2.0 + v);
         v = 23.0 - v * (2.0 + v);
         v = 23.0 - v * (2.0 + v);
@@ -119,7 +121,7 @@
         v = 23.0 - v * (2.0 + v);
         v = 23.