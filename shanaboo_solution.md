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
 #include "noc_nonblocking_api.h"
@@ -14,6 +15,7 @@
 using namespace sfpi;
 
 namespace ckernel {
+
 namespace sfpu {
 
 template <bool APPROXIMATION_MODE, int ITERATIONS>
@@ -21,7 +23,7 @@
     // SFPU microcode
     for (int d = 0; d < ITERATIONS; d++) {
         vFloat v = dst_reg[0];
-        v = 1.0f / v;
+        v = sfpi::recip(v);
         dst_reg[0] = v;
         dst_reg++;
     }
@@ -33,7 +35,7 @@
     // SFPU microcode
     for (int d = 0; d < ITERATIONS; d++) {
         vFloat v = dst_reg[0];
-        v = 1.0f / v;
+        v = sfpi::recip(v);
         dst_reg[0] = v;
         dst_reg++;
     }
@@ -45,7 +47,7 @@
     // SFPU microcode
     for (int d = 0; d < ITERATIONS; d++) {
         vFloat v = dst_reg[0];
-        v = 1.0f / v;
+        v = sfpi::recip(v);
         dst_reg[0] = v;
         dst_reg++;
     }
@@ -57,7 +59,7 @@
     // SFPU microcode
     for (int d = 0; d < ITERATIONS; d++) {
         vFloat v = dst_reg[0];
-        v = 1.0f / v;
+        v = sfpi::recip(v);
         dst_reg[0] = v;
         dst_reg++;
     }
@@ -69,7 +71,7 @@
     // SFPU microcode
     for (int d = 0; d < ITERATIONS; d++) {
         vFloat v = dst_reg[0];
-        v = 1.0f / v;
+        v = sfpi::recip(v);
         dst_reg[0] = v;
         dst_reg++;
     }
@@ -81,7 +83,7 @@
     // SFPU microcode
     for (int d = 0; d < ITERATIONS; d++) {
         vFloat v = dst_reg[0];
-        v = 1.0f / v;
+        v = sfpi::recip(v);
         dst_reg[0] = v;
         dst_reg++;
     }
@@ -93,7 +95,7 @@
     // SFPU microcode
     for (int d = 0; d < ITERATIONS; d++) {
         vFloat v = dst_reg[0];
-        v = 1.0f / v;
+        v = sfpi::recip(v);
         dst_reg[0] = v;
         dst_reg++;
     }
@@ -105,7 +107,7 @@
     // SFPU microcode
     for (int d = 0; d < ITERATIONS; d++) {
         vFloat v = dst_reg[0];
-        v = 1.0f / v;
+        v = sfpi::recip(v);
         dst_reg[0] = v;
         dst_reg++;
     }
@@ -117,7 +119,7 @@
     // SFPU microcode
     for (int d = 0; d < ITERATIONS; d++) {
         vFloat v = dst_reg[0];
-        v = 1.0f / v;
+        v = sfpi::recip(v);
         dst_reg[0] = v;
         dst_reg++;
     }
@@ -129,7 +131,7 @@
     // SFPU microcode
     for (int d = 0; d < ITERATIONS; d++) {
         vFloat v = dst_reg[0];
-        v = 1.0f / v;
+        v = sfpi::recip(v);
         dst_reg[0] = v;
         dst_reg++;
     }
@@ -141,7 +143,7 @@
     // SFPU microcode
     for (int d = 0; d < ITERATIONS; d++) {
         vFloat v = dst_reg[0];
-        v = 1.0f / v;
+        v = sfpi::recip(v);
         dst_reg[0] = v;
         dst_reg++;
     }
@@ -153,7 +155,7 @@
     // SFPU microcode
     for (int d = 0; d < ITERATIONS; d++) {
         vFloat v = dst_reg[0];
-        v = 1.0f / v;
+        v = sfpi::recip(v);
         dst_reg[0] = v;
         dst_reg++;
     }
@@ -165,7 +167,7 @@
     // SFPU microcode
     for (int d = 0; d < ITERATIONS; d++) {
         vFloat v = dst_reg[0];
-        v = 1.0f / v;
+        v = sfpi::recip(v);
         dst_reg[0] = v;
         dst_reg++;
     }
@@ -177,7 +179,7 @@
     // SFPU microcode
     for (int d = 0; d < ITERATIONS; d++) {
         vFloat v = dst_reg[0];
-        v = 1.0f / v;
+        v = sfpi::recip(v);
         dst_reg[0] = v;
         dst_reg++;
     }
@@ -189,7 +191,7 @@
     // SFPU microcode
     for (int d = 0; d < ITERATIONS; d++) {
         vFloat v = dst_reg[0];
-        v = 1.0f / v;
+        v = sfpi::recip(v);
         dst_reg[0] = v;
         dst_reg++;
