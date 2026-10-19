// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1100 --convert-builtin-func-to-llvm | FileCheck %s
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1170 --convert-builtin-func-to-llvm | FileCheck %s
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1030 --convert-builtin-func-to-llvm | FileCheck %s
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1031 --convert-builtin-func-to-llvm | FileCheck %s
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1030 --convert-builtin-func-to-llvm | FileCheck %s --check-prefix=FP8-SW
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1031 --convert-builtin-func-to-llvm | FileCheck %s --check-prefix=FP8-SW
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1100 --convert-builtin-func-to-llvm | FileCheck %s --check-prefix=FP8-SW

#blocked3 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: reduce_dpp_max
  tt.func @reduce_dpp_max(%arg0: tensor<32xf32, #blocked3>) {
    // CHECK: rocdl.update.dpp
    // CHECK-SAME: with 280, 15, 15, true : f32
    // CHECK-NEXT: llvm.intr.maxnum

    // CHECK-NEXT: rocdl.update.dpp
    // CHECK-SAME: with 276, 15, 15, true : f32
    // CHECK-NEXT: llvm.intr.maxnum

    // CHECK-NEXT: rocdl.update.dpp
    // CHECK-SAME: with 274, 15, 15, true : f32
    // CHECK-NEXT: llvm.intr.maxnum

    // CHECK-NEXT: rocdl.update.dpp
    // CHECK-SAME: with 273, 15, 15, true : f32
    // CHECK-NEXT: llvm.intr.maxnum

    // CHECK: rocdl.permlanex16
    // CHECK: llvm.intr.maxnum
    // CHECK: rocdl.readlane
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.maxnumf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<32xf32, #blocked3>) -> f32
    tt.return
  }
}

#linear = #ttg.linear<{register = [[16, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1]], warp = [], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @reduce_linear_layout
tt.func private @reduce_linear_layout(%arg0: tensor<32x2xi32, #linear>) -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #linear}>> {
  // This tensor has 64 elements with the last dimension across the lower and upper 16 lanes.
  // Therefore, we can reduce it with a 16 element butterfly shuffle.

  // CHECK-DAG: [[result0:%.*]] = llvm.mlir.undef
  // CHECK-DAG: [[select_lo:%.*]] = llvm.mlir.constant(1985229328 : i32)
  // CHECK-DAG: [[select_hi:%.*]] = llvm.mlir.constant(-19088744 : i32)
  // CHECK-DAG: [[reg0:%.*]] = llvm.extractvalue %arg0[0]
  // CHECK-DAG: [[reg1:%.*]] = llvm.extractvalue %arg0[1]
  // CHECK: [[permlane0:%.*]] = rocdl.permlanex16 [[reg0]], [[reg0]], [[select_lo]], [[select_hi]], true, false
  // CHECK: [[sum0:%.*]] = llvm.add [[reg0]], [[permlane0]]
  // CHECK: [[permlane1:%.*]] = rocdl.permlanex16 [[reg1]], [[reg1]], [[select_lo]], [[select_hi]], true, false
  // CHECK: [[sum1:%.*]] = llvm.add [[reg1]], [[permlane1]]
  // CHECK: [[result1:%.*]] = llvm.insertvalue [[sum0]], [[result0]][0]
  // CHECK: [[result2:%.*]] = llvm.insertvalue [[sum1]], [[result1]][1]

  %0 = "tt.reduce"(%arg0) ({
  ^bb0(%arg1: i32, %arg2: i32):
    %1 = arith.addi %arg1, %arg2 : i32
    tt.reduce.return %1 : i32
  }) {axis = 1 : i32} : (tensor<32x2xi32, #linear>) -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #linear}>>

  // CHECK: llvm.return [[result2]]
  tt.return %0 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #linear}>>
}
}

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @bf16_mulf
tt.func private @bf16_mulf(%arg0: tensor<64xbf16, #blocked>, %arg1: tensor<64xbf16, #blocked>) -> tensor<64xbf16, #blocked> {
  // v_dot2_bf16_bf16 truncates the bf16 result, so the multiply has to go
  // through fp32 and round to nearest even explicitly.
  // CHECK-NOT: llvm.amdgcn.fdot2.bf16.bf16
  // CHECK-COUNT-2: llvm.fmul {{.*}} : f32
  // CHECK-NOT: llvm.amdgcn.fdot2.bf16.bf16
  %0 = arith.mulf %arg0, %arg1 : tensor<64xbf16, #blocked>
  tt.return %0 : tensor<64xbf16, #blocked>
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // RDNA2 has neither MFMA nor WMMA, so a dot whose operands are dot_op-over-blocked lowers to
  // the packed 2-element half-precision dot intrinsic. Nothing pinned this for RDNA2 before: the
  // only fdot2 assertion in the suite was on the gfx942 (CDNA) arm of tritongpu_to_llvm.mlir, so
  // the fp16 matmul path this card depends on could regress silently.
  // All four RUN lines (gfx1030/1031/1100/1170) take this path, hence one shared CHECK.
  // 64 = 8 elements per thread (16x16 over 32 lanes at sizePerThread [1,1]) x 8 f16 pairs along K.
  // CHECK-COUNT is a lower bound, which is the direction that matters here: if the FMA path stops
  // emitting the packed dot (or silently falls back to scalar mul/add), the count collapses and
  // this fails. Verified by mutation: halving K to 8 drops emission to 32 and the case fails.
  // CHECK-LABEL: dot_fp16_fma
  tt.func private @dot_fp16_fma(%a: tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %b: tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>, %acc: tensor<16x16xf32, #blocked>) -> tensor<16x16xf32, #blocked> {
    // CHECK-COUNT-64: llvm.call_intrinsic "llvm.amdgcn.fdot2"
    %0 = tt.dot %a, %b, %acc, inputPrecision = ieee : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x16xf32, #blocked>
    tt.return %0 : tensor<16x16xf32, #blocked>
  }
}

// -----

#scaled = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // E4M3FN encodes NaN as 0x7F (and 0xFF with the sign set), but the software upcast's bias
  // multiply would turn those patterns into finite +/-480. The lowering must select a NaN first:
  // the plain fp8->f16 software path in this backend already does it
  // (Fp8E4M3fnToFp16OneValue selects 0x7E00 for the same byte), so the scaled path must not turn
  // the encoding into a number.
  // All four RUN lines take the software path (neither RDNA2 nor RDNA3 has a hardware scaled
  // upcast), hence one shared CHECK.
  // CHECK-LABEL: scaled_upcast_fp8_sw_nan
  // CHECK: %[[ABS:.+]] = llvm.and %{{.+}}, %{{.+}} : i8
  // CHECK: %[[ISNAN:.+]] = llvm.icmp "eq" %[[ABS]], %{{.+}} : i8
  // CHECK: llvm.select %[[ISNAN]], %{{.+}}, %{{.+}} : i1, f32
  tt.func @scaled_upcast_fp8_sw_nan(%in: tensor<32x128xf8E4M3FN, #scaled>, %scale: tensor<32x128xbf16, #scaled>, %out: tensor<32x128x!tt.ptr<bf16>, #scaled>) {
    %0 = amdg.scaled_upcast_fp8 %in scale %scale : tensor<32x128xf8E4M3FN, #scaled>, tensor<32x128xbf16, #scaled> -> tensor<32x128xbf16, #scaled>
    tt.store %out, %0 : tensor<32x128x!tt.ptr<bf16>, #scaled>
    tt.return
  }
}

// -----

#fp8blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // The plain fp8->bf16 cast is another route where a NaN encoding must not become a number.
  // E4M3FN 0x7F/0xFF and E5M2 0x7D..0x7F are the NaN patterns; the software bit trick maps them
  // to finite values, so the lowering selects a canonical bf16 NaN. The FP8-SW prefix covers the
  // three RUN lines whose families take this software path (gfx1030/1031/1100); gfx1170 has the
  // unscaled hardware convert and is deliberately not asserted here.
  // FP8-SW-LABEL: fp8_e4m3_bf16_sw_nan
  // FP8-SW: %[[ABS:.+]] = llvm.and %{{.+}}, %{{.+}} : i8
  // FP8-SW: %[[ISNAN:.+]] = llvm.icmp "eq" %[[ABS]], %{{.+}} : i8
  // FP8-SW: llvm.select %[[ISNAN]], %{{.+}}, %{{.+}} : i1, bf16
  tt.func @fp8_e4m3_bf16_sw_nan(%in: tensor<32x64xf8E4M3FN, #fp8blocked>, %out: tensor<32x64x!tt.ptr<bf16>, #fp8blocked>) {
    %0 = tt.fp_to_fp %in : tensor<32x64xf8E4M3FN, #fp8blocked> -> tensor<32x64xbf16, #fp8blocked>
    tt.store %out, %0 : tensor<32x64x!tt.ptr<bf16>, #fp8blocked>
    tt.return
  }

  // E5M2 has no dedicated NaN bit pattern: magnitudes 0x7D..0x7F are NaNs and 0x7C is infinity,
  // so the guard is an unsigned "greater than 0x7C" on the sign-cleared byte.
  // FP8-SW-LABEL: fp8_e5m2_bf16_sw_nan
  // FP8-SW: %[[ABS5:.+]] = llvm.and %{{.+}}, %{{.+}} : i8
  // FP8-SW: %[[NAN5:.+]] = llvm.icmp "ugt" %[[ABS5]], %{{.+}} : i8
  // FP8-SW: llvm.select %[[NAN5]], %{{.+}}, %{{.+}} : i1, bf16
  tt.func @fp8_e5m2_bf16_sw_nan(%in: tensor<32x64xf8E5M2, #fp8blocked>, %out: tensor<32x64x!tt.ptr<bf16>, #fp8blocked>) {
    %0 = tt.fp_to_fp %in : tensor<32x64xf8E5M2, #fp8blocked> -> tensor<32x64xbf16, #fp8blocked>
    tt.store %out, %0 : tensor<32x64x!tt.ptr<bf16>, #fp8blocked>
    tt.return
  }
}
