// RUN: triton-opt %s --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx90a" | FileCheck %s --check-prefix=GENERIC
// RUN: triton-opt %s --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx942" | FileCheck %s --check-prefix=BUFFER
// RUN: triton-opt %s --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx950" | FileCheck %s --check-prefix=BUFFER
// RUN: triton-opt %s --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx1030" | FileCheck %s --check-prefix=GENERIC
// RUN: triton-opt %s --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx1031" | FileCheck %s --check-prefix=GENERIC
// RUN: triton-opt %s --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx1100" | FileCheck %s --check-prefix=BUFFER
// RUN: triton-opt %s --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx1170" | FileCheck %s --check-prefix=BUFFER
// RUN: triton-opt %s --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx1200" | FileCheck %s --check-prefix=BUFFER
// RUN: triton-opt %s --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx1250" | FileCheck %s --check-prefix=BUFFER

// Verify the target-level buffer atomic RMW capability independently of the
// floating-point type-specific predicates.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // BUFFER-LABEL: atomic_add_i32
  // GENERIC-LABEL: atomic_add_i32
  tt.func public @atomic_add_i32(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %values: tensor<1024xi32, #blocked>) {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<i32>, i32
    %4 = tt.splat %3 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
    %5 = tt.addptr %4, %2 : tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>
    // BUFFER: amdg.buffer_atomic_rmw add
    // BUFFER-NOT: tt.atomic_rmw
    // GENERIC-NOT: amdg.buffer_atomic_rmw
    // GENERIC: tt.atomic_rmw add
    // GENERIC-NOT: amdg.buffer_atomic_rmw
    %6 = tt.atomic_rmw add, acq_rel, gpu, %5, %values : (tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>) -> tensor<1024xi32, #blocked>
    tt.return
  }

  // The gate is per-target, not per-op, so bitwise and exchange RMWs obey the same
  // family list as `add`. Those are exactly the ops a future RDNA2 enablement would
  // change: on the generic path LLVM has to CAS-expand and/or/xor (and signed/unsigned
  // integer min/max) on every AMD target, while `add` and f32 `exch` stay native.
  // Evidence: docs/rdna2/m1-implementation-report.md section 22.
  // BUFFER-LABEL: atomic_and_i32
  // GENERIC-LABEL: atomic_and_i32
  tt.func public @atomic_and_i32(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %values: tensor<1024xi32, #blocked>) {
    %c1024_i32_1 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32_1 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<i32>, i32
    %4 = tt.splat %3 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
    %5 = tt.addptr %4, %2 : tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>
    // BUFFER: amdg.buffer_atomic_rmw and
    // BUFFER-NOT: tt.atomic_rmw
    // GENERIC-NOT: amdg.buffer_atomic_rmw
    // GENERIC: tt.atomic_rmw and
    // GENERIC-NOT: amdg.buffer_atomic_rmw
    %6 = tt.atomic_rmw and, acq_rel, gpu, %5, %values : (tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>) -> tensor<1024xi32, #blocked>
    tt.return
  }

  // BUFFER-LABEL: atomic_exch_f32
  // GENERIC-LABEL: atomic_exch_f32
  tt.func public @atomic_exch_f32(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %values: tensor<1024xf32, #blocked>) {
    %c1024_i32_2 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32_2 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %4 = tt.splat %3 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %5 = tt.addptr %4, %2 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // BUFFER: amdg.buffer_atomic_rmw exch
    // BUFFER-NOT: tt.atomic_rmw
    // GENERIC-NOT: amdg.buffer_atomic_rmw
    // GENERIC: tt.atomic_rmw exch
    // GENERIC-NOT: amdg.buffer_atomic_rmw
    %6 = tt.atomic_rmw exch, acq_rel, gpu, %5, %values : (tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
    tt.return
  }
}
