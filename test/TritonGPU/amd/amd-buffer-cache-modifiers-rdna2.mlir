// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1030 | FileCheck %s
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1031 | FileCheck %s

// Verify that RDNA2 cache modifier qualifiers emit the GFX10 cache-policy aux values in
// rocdl.raw.ptr.buffer.load/store operations.
//
// GFX10 uses the pre-GFX12 cache-control layout, so the aux bits are named by their CDNA
// roles (LLVM aliases them SC0=GLC=0x1, NT=SLC=0x2, SC1=SCC=0x10 in SIDefines.h and masks the
// aux operand with ALL_pregfx12 for RDNA parts too):
//
//   Load  .cg -> 3   (SC0|NT:  bypass L1, do not expect reuse)
//   Load  .cs -> 3   (SC0|NT:  streaming)
//   Load  .cv -> 17  (SC0|SC1: do not cache, device-coherent refetch)
//   Store .cs -> 3   (SC0|NT)
//   Store .wt -> 17  (SC0|SC1: write through)
//
// RDNA3's encodings are deliberately different and must not be copied here: RDNA3 uses 1 and
// 7, and its 7 sets bit 2, which on GFX10 is CLAUSE (instruction fetch grouping), not the DLC
// far-cache hint. amd-buffer-cache-modifiers-rdna3.mlir pins the RDNA3 side.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: buffer_load_cg
// CHECK-COUNT-4: rocdl.raw.ptr.buffer.load {{.*}}, 3 :
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @buffer_load_cg(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    // .cg load on RDNA2: aux = 3 (SC0|NT)
    %ret = amdg.buffer_load %arg0[%offset] {cachePolicy = #tt.cache_policy<cache_modifier = cg, eviction_policy = evict_normal>} : !tt.ptr<f32> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: buffer_load_cs
// CHECK-COUNT-4: rocdl.raw.ptr.buffer.load {{.*}}, 3 :
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @buffer_load_cs(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    // .cs load on RDNA2: aux = 3 (SC0|NT)
    %ret = amdg.buffer_load %arg0[%offset] {cachePolicy = #tt.cache_policy<cache_modifier = cs, eviction_policy = evict_normal>} : !tt.ptr<f32> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: buffer_load_cv
// CHECK-COUNT-4: rocdl.raw.ptr.buffer.load {{.*}}, 17 :
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @buffer_load_cv(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    // .cv load on RDNA2: aux = 17 (SC0|SC1)
    %ret = amdg.buffer_load %arg0[%offset] {cachePolicy = #tt.cache_policy<cache_modifier = cv, eviction_policy = evict_normal>} : !tt.ptr<f32> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: buffer_store_cs
// CHECK-COUNT-4: rocdl.raw.ptr.buffer.store {{.*}}, 3 :
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @buffer_store_cs(%value: tensor<128xf32, #blocked>, %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    %c256_i32 = arith.constant 256 : i32
    // .cs store on RDNA2: aux = 3 (SC0|NT)
    amdg.buffer_store %value, %arg0[%offset] stride = %c256_i32 {cachePolicy = #tt.cache_policy<cache_modifier = cs, eviction_policy = evict_normal>} : !tt.ptr<f32> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: buffer_store_wt
// CHECK-COUNT-4: rocdl.raw.ptr.buffer.store {{.*}}, 17 :
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @buffer_store_wt(%value: tensor<128xf32, #blocked>, %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    %c256_i32 = arith.constant 256 : i32
    // .wt store on RDNA2: aux = 17 (SC0|SC1)
    amdg.buffer_store %value, %arg0[%offset] stride = %c256_i32 {cachePolicy = #tt.cache_policy<cache_modifier = wt, eviction_policy = evict_normal>} : !tt.ptr<f32> -> tensor<128xf32, #blocked>
    tt.return
  }
}
