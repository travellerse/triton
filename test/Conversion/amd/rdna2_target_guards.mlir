// RUN: triton-opt %s --triton-amdgpu-membar="gfx-arch=gfx1030" | FileCheck %s
// RUN: triton-opt %s --triton-amdgpu-membar="gfx-arch=gfx1031" | FileCheck %s
// RUN: triton-opt %s --convert-warp-pipeline="gfx-arch=gfx1030" | FileCheck %s
// RUN: triton-opt %s --convert-warp-pipeline="gfx-arch=gfx1031" | FileCheck %s

// Direct coverage of the two AMD passes that abort on an unclassified ISA.
// TritonAMDGPUMembar and ConvertWarpPipeline each build their TargetInfo from
// the gfx-arch option and emit "unsupported target" when computeISAFamily()
// returns Unknown, so these RUN lines are red until gfx103x is classified.
// Checking real printed IR (rather than using `not`) keeps a failing pass from
// looking like a passing pipeline.
//
// The arch comes exclusively from the pass option, so the module deliberately
// carries no ttg.target attribute: one input then serves both gfx1030 and
// gfx1031 without contradicting itself. With no memory or warp ops the
// post-guard work is vacuous by design, which is what makes the expected
// output an unchanged round-trip.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: @rdna2_target_guard_func
  tt.func public @rdna2_target_guard_func() {
    // tt.func rejects a body that holds only the terminator, so this constant
    // exists to make the function non-empty; nothing reads it.
    %unused = arith.constant 0 : i32
    // CHECK: tt.return
    tt.return
  }
}
