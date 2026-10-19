// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1030 | FileCheck %s
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1031 | FileCheck %s

// Verify that RDNA2 cache modifier qualifiers emit GFX10 cache-policy aux values in
// rocdl.raw.ptr.buffer.load/store operations.
//
// GFX10 buffer ops expose three cache fields. Bit positions per the AMD "RDNA 2" ISA Reference
// Guide section 13.6.1 (MTBUF), confirmed by llvm-objdump of the words Triton's LLVM emits
// (test/LLVMIR/rdna2-buffer-load-cache-policy.ll pins that side):
//   aux bit 0 -> GLC (instr bit 14)  load: miss L0 and refetch, invalidating a matching L0 line
//   aux bit 1 -> SLC (instr bit 54)  L2 policy, only meaningful together with DLC
//   aux bit 2 -> DLC (instr bit 15)  load: L1 miss-evict; store: L2 bypass
//   aux bits 3 and 4 -> no field on these ops, dropped by the encoder (which is why the CDNA
//   table, whose third field lives at bit 4, cannot be reused here)
//
// Per-modifier values follow the ISA guide's RDNA2 behaviour tables (section 8.1.10, tables 38
// for loads and 39 for stores/atomics):
//   Load  .cg -> 5 (GLC|DLC: L0 bypassed, L1 miss-evict, L2 stays LRU)
//   Load  .cs -> 6 (SLC|DLC: L1 miss-evict, L2 hit-no-allocate)
//   Load  .cv -> 5 (identical to .cg: no load encoding can bypass L2, and the ordering half of
//                   .cv is a s_waitcnt that aux cannot express)
//   Store .cs -> 2 (SLC: L2 stream)
//   Store .wt -> 5 (GLC|DLC: L2 bypass + no L0 persistence; write-around, not architectural
//                   write-through-to-system-memory)
//   Load  .ca -> 0, Store .cg -> 0 (controls: the default rows really do emit nothing)
//
// aux is the final operand of rocdl.raw.ptr.buffer.load/store, so pinning every lowered access
// is also the negative assertion: any other encoding fails the match. Each case lowers a
// 128-element f32 tensor at sizePerThread = 4 into four accesses.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: buffer_load_cg
// CHECK-COUNT-4: rocdl.raw.ptr.buffer.load {{.*}}, 5 :
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @buffer_load_cg(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    // .cg load on RDNA2: aux = 5 (GLC|DLC)
    %ret = amdg.buffer_load %arg0[%offset] {cachePolicy = #tt.cache_policy<cache_modifier = cg, eviction_policy = evict_normal>} : !tt.ptr<f32> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: buffer_load_cs
// CHECK-COUNT-4: rocdl.raw.ptr.buffer.load {{.*}}, 6 :
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @buffer_load_cs(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    // .cs load on RDNA2: aux = 6 (SLC|DLC)
    %ret = amdg.buffer_load %arg0[%offset] {cachePolicy = #tt.cache_policy<cache_modifier = cs, eviction_policy = evict_normal>} : !tt.ptr<f32> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: buffer_load_cv
// CHECK-COUNT-4: rocdl.raw.ptr.buffer.load {{.*}}, 5 :
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @buffer_load_cv(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    // .cv load on RDNA2: aux = 5, the same encoding .cg gets (see header)
    %ret = amdg.buffer_load %arg0[%offset] {cachePolicy = #tt.cache_policy<cache_modifier = cv, eviction_policy = evict_normal>} : !tt.ptr<f32> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: buffer_load_ca
// CHECK-COUNT-4: rocdl.raw.ptr.buffer.load {{.*}}, 0 :
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @buffer_load_ca(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    // .ca load on RDNA2: aux = 0 (L0 hit, L1 hit-LRU, L2 LRU)
    %ret = amdg.buffer_load %arg0[%offset] {cachePolicy = #tt.cache_policy<cache_modifier = ca, eviction_policy = evict_normal>} : !tt.ptr<f32> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: buffer_store_cs
// CHECK-COUNT-4: rocdl.raw.ptr.buffer.store {{.*}}, 2 :
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @buffer_store_cs(%value: tensor<128xf32, #blocked>, %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    %c256_i32 = arith.constant 256 : i32
    // .cs store on RDNA2: aux = 2 (SLC: L2 stream)
    amdg.buffer_store %value, %arg0[%offset] stride = %c256_i32 {cachePolicy = #tt.cache_policy<cache_modifier = cs, eviction_policy = evict_normal>} : !tt.ptr<f32> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: buffer_store_wt
// CHECK-COUNT-4: rocdl.raw.ptr.buffer.store {{.*}}, 5 :
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @buffer_store_wt(%value: tensor<128xf32, #blocked>, %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    %c256_i32 = arith.constant 256 : i32
    // .wt store on RDNA2: aux = 5 (GLC|DLC: L2 bypass, write-around not write-through)
    amdg.buffer_store %value, %arg0[%offset] stride = %c256_i32 {cachePolicy = #tt.cache_policy<cache_modifier = wt, eviction_policy = evict_normal>} : !tt.ptr<f32> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: buffer_store_cg
// CHECK-COUNT-4: rocdl.raw.ptr.buffer.store {{.*}}, 0 :
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @buffer_store_cg(%value: tensor<128xf32, #blocked>, %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    %c256_i32 = arith.constant 256 : i32
    // .cg store on RDNA2: aux = 0 -- L1 is already bypassed-but-coherent for stores, so the
    // default L2 LRU encoding is the exact match
    amdg.buffer_store %value, %arg0[%offset] stride = %c256_i32 {cachePolicy = #tt.cache_policy<cache_modifier = cg, eviction_policy = evict_normal>} : !tt.ptr<f32> -> tensor<128xf32, #blocked>
    tt.return
  }
}
