// RUN: triton-opt %s --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx1030" | FileCheck %s --check-prefix=RDNA2
// RUN: triton-opt %s --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx1031" | FileCheck %s --check-prefix=RDNA2
// RUN: triton-opt %s --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx1100" | FileCheck %s --check-prefix=RDNA3
// RUN: triton-opt %s --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx942" | FileCheck %s --check-prefix=CDNA3
// RUN: triton-opt %s --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx90a" | FileCheck %s --check-prefix=GENERIC

// Operation- and type-scoped RDNA2 buffer atomic RMW (M1M candidate). gfx103x has no
// supportsBufferAtomicRMW() entry; instead the rewrite registers for RDNA2 and only accepts
// integer and/or/xor/umin/umax at 32 and 64 bits when every lane is provably active. The reason for
// the lane requirement is arithmetic, not missing documentation: a masked lane is sent to offset
// 0x8000_0000 and the RDNA2 guide (8.1, Table 35 note 3) says buffer atomics are range-checked
// all-or-nothing, but that offset only exceeds numRecords = 0x7FFF_FFFE under an unsigned compare,
// which the guide does not state -- so until a device probe settles it, no masked lane may take the
// buffer path. RDNA3 and CDNA3 are the gate-open controls, gfx90a the gate-closed control.
// `add`, `sub` and `exch` stay on the generic path on purpose (they are native there too, and the
// pinned per-target behaviour is kept in amd-convert-buffer-ops-atomic-rmw.mlir).
// Evidence: docs/rdna2/buffer-atomic-capability-audit-rdna2.md,
// docs/rdna2/m1-implementation-report.md section 24.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked_small = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {

  // ---- positive: the allowlisted integer ops, unmasked, one element per lane ----

  // RDNA2-LABEL: @and_i32
  // RDNA3-LABEL: @and_i32
  // CDNA3-LABEL: @and_i32
  // GENERIC-LABEL: @and_i32
  tt.func public @and_i32(%ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %values: tensor<1024xi32, #blocked>) {
    %base = tt.splat %ptr : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
    %offs = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %addrs = tt.addptr %base, %offs : tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>
    // RDNA2: amdg.buffer_atomic_rmw and
    // RDNA2-NOT: tt.atomic_rmw
    // RDNA3: amdg.buffer_atomic_rmw and
    // CDNA3: amdg.buffer_atomic_rmw and
    // GENERIC-NOT: amdg.buffer_atomic_rmw
    // GENERIC: tt.atomic_rmw and
    %res = tt.atomic_rmw and, acq_rel, gpu, %addrs, %values : (tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>) -> tensor<1024xi32, #blocked>
    tt.return
  }

  // RDNA2-LABEL: @or_i32
  // RDNA3-LABEL: @or_i32
  // CDNA3-LABEL: @or_i32
  // GENERIC-LABEL: @or_i32
  tt.func public @or_i32(%ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %values: tensor<1024xi32, #blocked>) {
    %base = tt.splat %ptr : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
    %offs = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %addrs = tt.addptr %base, %offs : tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>
    // RDNA2: amdg.buffer_atomic_rmw or
    // RDNA2-NOT: tt.atomic_rmw
    // RDNA3: amdg.buffer_atomic_rmw or
    // CDNA3: amdg.buffer_atomic_rmw or
    // GENERIC-NOT: amdg.buffer_atomic_rmw
    // GENERIC: tt.atomic_rmw or
    %res = tt.atomic_rmw or, acq_rel, gpu, %addrs, %values : (tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>) -> tensor<1024xi32, #blocked>
    tt.return
  }

  // RDNA2-LABEL: @xor_i32
  // RDNA3-LABEL: @xor_i32
  // CDNA3-LABEL: @xor_i32
  // GENERIC-LABEL: @xor_i32
  tt.func public @xor_i32(%ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %values: tensor<1024xi32, #blocked>) {
    %base = tt.splat %ptr : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
    %offs = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %addrs = tt.addptr %base, %offs : tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>
    // RDNA2: amdg.buffer_atomic_rmw xor
    // RDNA2-NOT: tt.atomic_rmw
    // RDNA3: amdg.buffer_atomic_rmw xor
    // CDNA3: amdg.buffer_atomic_rmw xor
    // GENERIC-NOT: amdg.buffer_atomic_rmw
    // GENERIC: tt.atomic_rmw xor
    %res = tt.atomic_rmw xor, acq_rel, gpu, %addrs, %values : (tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>) -> tensor<1024xi32, #blocked>
    tt.return
  }

  // RDNA2-LABEL: @umin_i32
  // RDNA3-LABEL: @umin_i32
  // CDNA3-LABEL: @umin_i32
  // GENERIC-LABEL: @umin_i32
  tt.func public @umin_i32(%ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %values: tensor<1024xi32, #blocked>) {
    %base = tt.splat %ptr : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
    %offs = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %addrs = tt.addptr %base, %offs : tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>
    // RDNA2: amdg.buffer_atomic_rmw umin
    // RDNA2-NOT: tt.atomic_rmw
    // RDNA3: amdg.buffer_atomic_rmw umin
    // CDNA3: amdg.buffer_atomic_rmw umin
    // GENERIC-NOT: amdg.buffer_atomic_rmw
    // GENERIC: tt.atomic_rmw umin
    %res = tt.atomic_rmw umin, acq_rel, gpu, %addrs, %values : (tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>) -> tensor<1024xi32, #blocked>
    tt.return
  }

  // RDNA2-LABEL: @umax_i32
  // RDNA3-LABEL: @umax_i32
  // CDNA3-LABEL: @umax_i32
  // GENERIC-LABEL: @umax_i32
  tt.func public @umax_i32(%ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %values: tensor<1024xi32, #blocked>) {
    %base = tt.splat %ptr : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
    %offs = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %addrs = tt.addptr %base, %offs : tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>
    // RDNA2: amdg.buffer_atomic_rmw umax
    // RDNA2-NOT: tt.atomic_rmw
    // RDNA3: amdg.buffer_atomic_rmw umax
    // CDNA3: amdg.buffer_atomic_rmw umax
    // GENERIC-NOT: amdg.buffer_atomic_rmw
    // GENERIC: tt.atomic_rmw umax
    %res = tt.atomic_rmw umax, acq_rel, gpu, %addrs, %values : (tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>) -> tensor<1024xi32, #blocked>
    tt.return
  }

  // RDNA2-LABEL: @and_i64
  // RDNA3-LABEL: @and_i64
  // CDNA3-LABEL: @and_i64
  // GENERIC-LABEL: @and_i64
  tt.func public @and_i64(%ptr: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %values: tensor<1024xi64, #blocked>) {
    %base = tt.splat %ptr : !tt.ptr<i64> -> tensor<1024x!tt.ptr<i64>, #blocked>
    %offs = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %addrs = tt.addptr %base, %offs : tensor<1024x!tt.ptr<i64>, #blocked>, tensor<1024xi32, #blocked>
    // RDNA2: amdg.buffer_atomic_rmw and
    // RDNA2-NOT: tt.atomic_rmw
    // RDNA3: amdg.buffer_atomic_rmw and
    // CDNA3: amdg.buffer_atomic_rmw and
    // GENERIC-NOT: amdg.buffer_atomic_rmw
    // GENERIC: tt.atomic_rmw and
    %res = tt.atomic_rmw and, acq_rel, gpu, %addrs, %values : (tensor<1024x!tt.ptr<i64>, #blocked>, tensor<1024xi64, #blocked>) -> tensor<1024xi64, #blocked>
    tt.return
  }

  // RDNA2-LABEL: @umax_i64
  // RDNA3-LABEL: @umax_i64
  // CDNA3-LABEL: @umax_i64
  // GENERIC-LABEL: @umax_i64
  tt.func public @umax_i64(%ptr: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %values: tensor<1024xi64, #blocked>) {
    %base = tt.splat %ptr : !tt.ptr<i64> -> tensor<1024x!tt.ptr<i64>, #blocked>
    %offs = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %addrs = tt.addptr %base, %offs : tensor<1024x!tt.ptr<i64>, #blocked>, tensor<1024xi32, #blocked>
    // RDNA2: amdg.buffer_atomic_rmw umax
    // RDNA2-NOT: tt.atomic_rmw
    // RDNA3: amdg.buffer_atomic_rmw umax
    // CDNA3: amdg.buffer_atomic_rmw umax
    // GENERIC-NOT: amdg.buffer_atomic_rmw
    // GENERIC: tt.atomic_rmw umax
    %res = tt.atomic_rmw umax, acq_rel, gpu, %addrs, %values : (tensor<1024x!tt.ptr<i64>, #blocked>, tensor<1024xi64, #blocked>) -> tensor<1024xi64, #blocked>
    tt.return
  }

  // ---- negative: excluded operations must stay on the generic atomic path ----

  // RDNA2-LABEL: @add_i32
  // RDNA3-LABEL: @add_i32
  // CDNA3-LABEL: @add_i32
  // GENERIC-LABEL: @add_i32
  tt.func public @add_i32(%ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %values: tensor<1024xi32, #blocked>) {
    %base = tt.splat %ptr : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
    %offs = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %addrs = tt.addptr %base, %offs : tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>
    // RDNA2-NOT: amdg.buffer_atomic_rmw
    // RDNA2: tt.atomic_rmw add
    // RDNA3: amdg.buffer_atomic_rmw add
    // CDNA3: amdg.buffer_atomic_rmw add
    %res = tt.atomic_rmw add, acq_rel, gpu, %addrs, %values : (tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>) -> tensor<1024xi32, #blocked>
    tt.return
  }

  // RDNA2-LABEL: @exch_i32
  // RDNA3-LABEL: @exch_i32
  // CDNA3-LABEL: @exch_i32
  // GENERIC-LABEL: @exch_i32
  tt.func public @exch_i32(%ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %values: tensor<1024xi32, #blocked>) {
    %base = tt.splat %ptr : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
    %offs = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %addrs = tt.addptr %base, %offs : tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>
    // RDNA2-NOT: amdg.buffer_atomic_rmw
    // RDNA2: tt.atomic_rmw exch
    // RDNA3: amdg.buffer_atomic_rmw exch
    // CDNA3: amdg.buffer_atomic_rmw exch
    %res = tt.atomic_rmw exch, acq_rel, gpu, %addrs, %values : (tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>) -> tensor<1024xi32, #blocked>
    tt.return
  }

  // RDNA2-LABEL: @fadd_f32
  // RDNA3-LABEL: @fadd_f32
  // CDNA3-LABEL: @fadd_f32
  // GENERIC-LABEL: @fadd_f32
  tt.func public @fadd_f32(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %values: tensor<1024xf32, #blocked>) {
    %base = tt.splat %ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %offs = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %addrs = tt.addptr %base, %offs : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // RDNA2-NOT: amdg.buffer_atomic_rmw
    // RDNA2: tt.atomic_rmw fadd
    // RDNA3: amdg.buffer_atomic_rmw fadd
    // CDNA3: amdg.buffer_atomic_rmw fadd
    %res = tt.atomic_rmw fadd, acq_rel, gpu, %addrs, %values : (tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
    tt.return
  }

  // f64 exchange cannot select on any target today; RDNA2 must keep the working generic path.
  // The pass itself still converts it on gate-open targets, which is where the codegen abort
  // comes from (pre-existing, reported in section 22.4).
  // RDNA2-LABEL: @exch_f64
  // RDNA3-LABEL: @exch_f64
  // CDNA3-LABEL: @exch_f64
  // GENERIC-LABEL: @exch_f64
  tt.func public @exch_f64(%ptr: !tt.ptr<f64> {tt.divisibility = 16 : i32}, %values: tensor<1024xf64, #blocked>) {
    %base = tt.splat %ptr : !tt.ptr<f64> -> tensor<1024x!tt.ptr<f64>, #blocked>
    %offs = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %addrs = tt.addptr %base, %offs : tensor<1024x!tt.ptr<f64>, #blocked>, tensor<1024xi32, #blocked>
    // RDNA2-NOT: amdg.buffer_atomic_rmw
    // RDNA2: tt.atomic_rmw exch
    // RDNA3: amdg.buffer_atomic_rmw exch
    // CDNA3: amdg.buffer_atomic_rmw exch
    %res = tt.atomic_rmw exch, acq_rel, gpu, %addrs, %values : (tensor<1024x!tt.ptr<f64>, #blocked>, tensor<1024xf64, #blocked>) -> tensor<1024xf64, #blocked>
    tt.return
  }

  // Signed integer min/max already use the generic path on every target, including gate-open ones.
  // RDNA2-LABEL: @smax_smin_i32
  // RDNA3-LABEL: @smax_smin_i32
  // CDNA3-LABEL: @smax_smin_i32
  // GENERIC-LABEL: @smax_smin_i32
  tt.func public @smax_smin_i32(%ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %values: tensor<1024xi32, #blocked>) {
    %base = tt.splat %ptr : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
    %offs = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %addrs = tt.addptr %base, %offs : tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>
    // RDNA2-NOT: amdg.buffer_atomic_rmw
    // RDNA2: tt.atomic_rmw max
    // RDNA3-NOT: amdg.buffer_atomic_rmw
    // CDNA3-NOT: amdg.buffer_atomic_rmw
    %res0 = tt.atomic_rmw max, acq_rel, gpu, %addrs, %values : (tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>) -> tensor<1024xi32, #blocked>
    // RDNA2: tt.atomic_rmw min
    %res1 = tt.atomic_rmw min, acq_rel, gpu, %addrs, %values : (tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>) -> tensor<1024xi32, #blocked>
    tt.return
  }

  // Floating-point min/max has no RDNA2 buffer atomic and no native fallback here.
  // RDNA2-LABEL: @fmax_f32
  // RDNA3-LABEL: @fmax_f32
  // CDNA3-LABEL: @fmax_f32
  // GENERIC-LABEL: @fmax_f32
  tt.func public @fmax_f32(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %values: tensor<1024xf32, #blocked>) {
    %base = tt.splat %ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %offs = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %addrs = tt.addptr %base, %offs : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // RDNA2-NOT: amdg.buffer_atomic_rmw
    // RDNA2: tt.atomic_rmw max
    // RDNA3: amdg.buffer_atomic_rmw max
    // CDNA3-NOT: amdg.buffer_atomic_rmw
    // CDNA3: tt.atomic_rmw max
    %res = tt.atomic_rmw max, acq_rel, gpu, %addrs, %values : (tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
    tt.return
  }

  // ---- negative: lane-suppression cases, which is why the subset is not op-only ----

  // A user mask makes inactive lanes depend on descriptor bounds suppression.
  // RDNA2-LABEL: @and_i32_masked
  // RDNA3-LABEL: @and_i32_masked
  // CDNA3-LABEL: @and_i32_masked
  // GENERIC-LABEL: @and_i32_masked
  tt.func public @and_i32_masked(%ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %values: tensor<1024xi32, #blocked>, %mask: tensor<1024xi1, #blocked>) {
    %base = tt.splat %ptr : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
    %offs = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %addrs = tt.addptr %base, %offs : tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>
    // RDNA2-NOT: amdg.buffer_atomic_rmw
    // RDNA2: tt.atomic_rmw and
    // RDNA3: amdg.buffer_atomic_rmw and
    // CDNA3: amdg.buffer_atomic_rmw and
    %res = tt.atomic_rmw and, acq_rel, gpu, %addrs, %values, %mask : (tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>, tensor<1024xi1, #blocked>) -> tensor<1024xi32, #blocked>
    tt.return
  }

  // Same for a masked unsigned min, the other shape the runtime path would take.
  // RDNA2-LABEL: @umin_i32_masked
  // RDNA3-LABEL: @umin_i32_masked
  // CDNA3-LABEL: @umin_i32_masked
  // GENERIC-LABEL: @umin_i32_masked
  tt.func public @umin_i32_masked(%ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %values: tensor<1024xi32, #blocked>, %mask: tensor<1024xi1, #blocked>) {
    %base = tt.splat %ptr : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
    %offs = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %addrs = tt.addptr %base, %offs : tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>
    // RDNA2-NOT: amdg.buffer_atomic_rmw
    // RDNA2: tt.atomic_rmw umin
    // RDNA3: amdg.buffer_atomic_rmw umin
    // CDNA3: amdg.buffer_atomic_rmw umin
    %res = tt.atomic_rmw umin, acq_rel, gpu, %addrs, %values, %mask : (tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>, tensor<1024xi1, #blocked>) -> tensor<1024xi32, #blocked>
    tt.return
  }

  // Redundant lanes: four elements spread over a 4-warp CTA leaves most threads with no element
  // of their own, so the emitter predicates lanes off and again needs the unverified
  // suppression. Gate-open targets do not check this; the RDNA2 subset is deliberately stricter.
  // RDNA2-LABEL: @and_i32_redundant
  // RDNA3-LABEL: @and_i32_redundant
  // CDNA3-LABEL: @and_i32_redundant
  // GENERIC-LABEL: @and_i32_redundant
  tt.func public @and_i32_redundant(%ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %values: tensor<4xi32, #blocked_small>) {
    %base = tt.splat %ptr : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>, #blocked_small>
    %offs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #blocked_small>
    %addrs = tt.addptr %base, %offs : tensor<4x!tt.ptr<i32>, #blocked_small>, tensor<4xi32, #blocked_small>
    // RDNA2-NOT: amdg.buffer_atomic_rmw
    // RDNA2: tt.atomic_rmw and
    // RDNA3: amdg.buffer_atomic_rmw and
    // CDNA3: amdg.buffer_atomic_rmw and
    %res = tt.atomic_rmw and, acq_rel, gpu, %addrs, %values : (tensor<4x!tt.ptr<i32>, #blocked_small>, tensor<4xi32, #blocked_small>) -> tensor<4xi32, #blocked_small>
    tt.return
  }

  // A splat pointer has no addptr to describe, so no target reaches the buffer rewrite.
  // RDNA2-LABEL: @and_i32_splat_ptr
  // RDNA3-LABEL: @and_i32_splat_ptr
  // CDNA3-LABEL: @and_i32_splat_ptr
  // GENERIC-LABEL: @and_i32_splat_ptr
  tt.func public @and_i32_splat_ptr(%ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %values: tensor<1024xi32, #blocked>) {
    %base = tt.splat %ptr : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
    // RDNA2-NOT: amdg.buffer_atomic_rmw
    // RDNA3-NOT: amdg.buffer_atomic_rmw
    // CDNA3-NOT: amdg.buffer_atomic_rmw
    // GENERIC-NOT: amdg.buffer_atomic_rmw
    %res = tt.atomic_rmw and, acq_rel, gpu, %base, %values : (tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>) -> tensor<1024xi32, #blocked>
    tt.return
  }

  // ---- control: compare-and-swap is registered separately and must not move ----
  // RDNA2-LABEL: @cas_i32
  // RDNA3-LABEL: @cas_i32
  // CDNA3-LABEL: @cas_i32
  // GENERIC-LABEL: @cas_i32
  tt.func public @cas_i32(%ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32},
                          %cmp: tensor<1024xi32, #blocked>, %val: tensor<1024xi32, #blocked>) {
    %base = tt.splat %ptr : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
    %offs = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %addrs = tt.addptr %base, %offs : tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>
    // RDNA2: amdg.buffer_atomic_cas
    // RDNA3: amdg.buffer_atomic_cas
    // CDNA3: amdg.buffer_atomic_cas
    // GENERIC: amdg.buffer_atomic_cas
    %res = tt.atomic_cas acq_rel, gpu, %addrs, %cmp, %val : (tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>, tensor<1024xi32, #blocked>) -> tensor<1024xi32, #blocked>
    tt.return
  }
}
