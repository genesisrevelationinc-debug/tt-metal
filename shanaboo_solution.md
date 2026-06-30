 ```diff
--- a/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_tanh.h
+++ b/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_tanh.h
@@ -18,7 +18,7 @@
 template <bool APPROXIMATE>
 inline void llk_math_eltwise_unary_sfpu_tanh(uint dst_index, int vector_mode = (int)VectorMode::RC, int param0 = 0, int param1 = 0, int param2 = 0, int param3 = 0, int param4 = 0, int param5 = 0) {
     if constexpr Math::is_fp32_mode_enabled() {
-        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
+        llk_math_eltwise_unary_sfpu_params<APPROXIMATE, 0, 0, 0, 0, 0, 0