// RDNA2 has neither MFMA nor WMMA, and TritonAMDGPUAccelerateMatmul registers no
// matrix-instruction pattern for ISAFamily::RDNA2 at all, so a dot that RDNA3 lowers
// to v_wmma keeps its blocked operands here. gfx1010 (which classifies as Unknown) is
// the control run line, and gfx1100 shows the negative assertions are not vacuous.

// RUN: triton-opt %s --tritonamdgpu-accelerate-matmul="gfx-arch=gfx1100" | FileCheck %s --check-prefix=RDNA3
// RUN: triton-opt %s --tritonamdgpu-accelerate-matmul="gfx-arch=gfx1030" | FileCheck %s --check-prefix=NO-MATRIX --implicit-check-not=#ttg.amd_wmma --implicit-check-not=#ttg.amd_mfma
// RUN: triton-opt %s --tritonamdgpu-accelerate-matmul="gfx-arch=gfx1031" | FileCheck %s --check-prefix=NO-MATRIX --implicit-check-not=#ttg.amd_wmma --implicit-check-not=#ttg.amd_mfma
// RUN: triton-opt %s --tritonamdgpu-accelerate-matmul="gfx-arch=gfx1010" | FileCheck %s --check-prefix=NO-MATRIX --implicit-check-not=#ttg.amd_wmma --implicit-check-not=#ttg.amd_mfma

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // RDNA3: #[[WMMA:.+]] = #ttg.amd_wmma<{version = 1, isTranspose = true, ctaLayout = {warp = {{\[\[0, 1\], \[0, 2\]\]}}}}>
  // RDNA3: tt.func public @dot_fp16_wave32
  // RDNA3: %[[DOT:.+]] = tt.dot
  // RDNA3-SAME: -> tensor<128x256xf32, #[[WMMA]]>

  // NO-MATRIX-LABEL: tt.func public @dot_fp16_wave32
  // NO-MATRIX: %[[DOT:.+]] = tt.dot
  // NO-MATRIX-SAME: -> tensor<128x256xf32, #blocked>
  tt.func public @dot_fp16_wave32(
      %arg0: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<128x256x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked>
    %dot = tt.dot %arg0, %arg1, %cst : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x256xf32, #blocked>
    tt.store %arg2, %dot : tensor<128x256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
