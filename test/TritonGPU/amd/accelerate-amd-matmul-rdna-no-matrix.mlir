// RDNA2 has neither MFMA nor WMMA, and TritonAMDGPUAccelerateMatmul registers no
// matrix-instruction pattern for ISAFamily::RDNA2 at all, so a dot that RDNA3 lowers
// to v_wmma keeps its blocked operands here. gfx1010 (which classifies as Unknown) is
// the control run line, and gfx1100 shows the negative assertions are not vacuous.

// RUN: triton-opt %s --tritonamdgpu-accelerate-matmul="gfx-arch=gfx1100" | FileCheck %s --check-prefix=RDNA3
// RUN: triton-opt %s --tritonamdgpu-accelerate-matmul="gfx-arch=gfx1030" | FileCheck %s --check-prefix=NO-MATRIX --implicit-check-not=#ttg.amd_wmma --implicit-check-not=#ttg.amd_mfma
// RUN: triton-opt %s --tritonamdgpu-accelerate-matmul="gfx-arch=gfx1031" | FileCheck %s --check-prefix=NO-MATRIX --implicit-check-not=#ttg.amd_wmma --implicit-check-not=#ttg.amd_mfma
// RUN: triton-opt %s --tritonamdgpu-accelerate-matmul="gfx-arch=gfx1010" | FileCheck %s --check-prefix=NO-MATRIX --implicit-check-not=#ttg.amd_wmma --implicit-check-not=#ttg.amd_mfma
// RUN: triton-opt %s --tritonamdgpu-accelerate-matmul="gfx-arch=gfx1030" | FileCheck %s --check-prefix=SCALED-RDNA2

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blockedA = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blockedB = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blockedS = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blockedO = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
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

  // RDNA2 has neither MFMA nor WMMA, so the only scaled-dot pattern registered for it is the software
  // decomposition: tl.dot_scaled becomes an explicit amdg.scaled_upcast_fp4 plus an ordinary tt.dot whose
  // operands stay dot_op-over-blocked, which is the shape the FMA lowering handles. Without that arm the
  // op survived into ConvertTritonGPUToLLVM and the process died on report_fatal_error "Unsupported
  // DotScaleOp found when converting TritonGPU to LLVM" (third_party/amd/lib/TritonAMDGPUToLLVM/
  // DotOpToLLVM.cpp) -- an uncatchable abort, not a diagnosable error.
  // SCALED-RDNA2-LABEL: tt.func public @rdna2_dot_scaled_e2m1
  // The op has to be gone, replaced by the upcast and a plain dot -- asserting both, because "no
  // dot_scaled" alone would also be satisfied by the op being dropped.
  // SCALED-RDNA2-NOT: tt.dot_scaled
  // SCALED-RDNA2: amdg.scaled_upcast_fp4
  // SCALED-RDNA2: tt.dot {{.*}} : tensor<32x64xf32, #ttg.dot_op<{{.*}}>> * tensor<64x32xf32, #ttg.dot_op<{{.*}}>> -> tensor<32x32xf32, #
  tt.func public @rdna2_dot_scaled_e2m1(
      %arg0: tensor<32x64x!tt.ptr<f8E4M3FN>, #blockedA>,
      %arg1: tensor<32x32x!tt.ptr<i8>, #blockedB>,
      %arg2: tensor<32x2x!tt.ptr<i8>, #blockedS>,
      %arg3: tensor<32x32x!tt.ptr<f32>, #blockedO>
    ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blockedO>
    %1 = tt.load %arg0 : tensor<32x64x!tt.ptr<f8E4M3FN>, #blockedA>
    %2 = tt.load %arg1 : tensor<32x32x!tt.ptr<i8>, #blockedB>
    %3 = tt.load %arg2 : tensor<32x2x!tt.ptr<i8>, #blockedS>
    %4 = tt.dot_scaled %1, %2 scale %3, %cst lhs = e4m3 rhs = e2m1 {fastMath = false} : tensor<32x64xf8E4M3FN, #blockedA> * tensor<32x32xi8, #blockedB>, tensor<32x2xi8, #blockedS> -> tensor<32x32xf32, #blockedO>
    tt.store %arg3, %4 : tensor<32x32x!tt.ptr<f32>, #blockedO>
    tt.return
  }
}
