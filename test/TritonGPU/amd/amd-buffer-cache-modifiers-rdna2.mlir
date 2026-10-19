// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1030 | FileCheck %s
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1031 | FileCheck %s

// Record the current RDNA2 cache-modifier behaviour: getCtrlBitsForCacheModifierOnTarget
// has no GFX10 mapping, so every qualifier lowers to aux = 0 and the modifier is
// dropped. That is numerically safe (aux = 0 is the default caching policy), but the
// requested hint never reaches the instruction -- unlike RDNA3, where the same requests
// emit 1 or 7 (see amd-buffer-cache-modifiers-rdna3.mlir).
//
// aux is the final operand of rocdl.raw.ptr.buffer.load/store, so pinning every lowered
// access to 0 is also the negative assertion: any nonzero encoding fails the match.
// Wiring up a GFX10 policy therefore has to update this file, which is the review
// signal that a mapping was added and hardware-validated.
//
// Each case lowers a 128-element f32 tensor at sizePerThread = 4 into four accesses.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: buffer_load_cg
// CHECK-COUNT-4: rocdl.raw.ptr.buffer.load {{.*}}, 0 :
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @buffer_load_cg(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    %ret = amdg.buffer_load %arg0[%offset] {cachePolicy = #tt.cache_policy<cache_modifier = cg, eviction_policy = evict_normal>} : !tt.ptr<f32> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: buffer_load_cs
// CHECK-COUNT-4: rocdl.raw.ptr.buffer.load {{.*}}, 0 :
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @buffer_load_cs(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    %ret = amdg.buffer_load %arg0[%offset] {cachePolicy = #tt.cache_policy<cache_modifier = cs, eviction_policy = evict_normal>} : !tt.ptr<f32> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: buffer_load_cv
// CHECK-COUNT-4: rocdl.raw.ptr.buffer.load {{.*}}, 0 :
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @buffer_load_cv(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    %ret = amdg.buffer_load %arg0[%offset] {cachePolicy = #tt.cache_policy<cache_modifier = cv, eviction_policy = evict_normal>} : !tt.ptr<f32> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: buffer_store_cs
// CHECK-COUNT-4: rocdl.raw.ptr.buffer.store {{.*}}, 0 :
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @buffer_store_cs(%value: tensor<128xf32, #blocked>, %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    %c256_i32 = arith.constant 256 : i32
    amdg.buffer_store %value, %arg0[%offset] stride = %c256_i32 {cachePolicy = #tt.cache_policy<cache_modifier = cs, eviction_policy = evict_normal>} : !tt.ptr<f32> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: buffer_store_wt
// CHECK-COUNT-4: rocdl.raw.ptr.buffer.store {{.*}}, 0 :
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @buffer_store_wt(%value: tensor<128xf32, #blocked>, %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    %c256_i32 = arith.constant 256 : i32
    amdg.buffer_store %value, %arg0[%offset] stride = %c256_i32 {cachePolicy = #tt.cache_policy<cache_modifier = wt, eviction_policy = evict_normal>} : !tt.ptr<f32> -> tensor<128xf32, #blocked>
    tt.return
  }
}
