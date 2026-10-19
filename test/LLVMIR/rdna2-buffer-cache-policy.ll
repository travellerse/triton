; ISA-level regression for the GFX10 buffer cache-policy operand.
;
; The MLIR-side test (test/TritonGPU/amd/amd-buffer-cache-modifiers-rdna2.mlir) pins the aux
; constant that Triton emits. This pins what LLVM encodes from that constant on gfx10, so an
; ROCDL lowering change or an LLVM bump that silently re-assigns or drops a bit has to fail here
; instead of quietly altering cache behaviour:
;
;   aux bit 0 -> glc   (instr bit 14)
;   aux bit 1 -> slc   (instr bit 54)
;   aux bit 2 -> dlc   (instr bit 15)
;   aux bits >= 3 -> dropped (the aux=16 case encodes identically to aux=0)
;
; The bit numbers on the right are from the AMD "RDNA 2" ISA Reference Guide, section 13.6.2
; (MUBUF field table; the MTBUF table in 13.6.1 agrees); they were confirmed against llvm-objdump output
; for these same aux values, and the words Triton emits start 0xE038, whose bits [31:26] = 111000
; is the MUBUF encoding field (MTBUF would be 111010).
; The last point above is why RDNA2 must not reuse the CDNA3/CDNA4 table: CDNA's SC1 lives at
; bit 4, which GFX10 discards even though LLVM's pre-GFX12 validity mask admits it.
; Kernel names carry no cache-field substrings so the CHECK-NOTs below can only match
; instruction text.

; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 -filetype=asm -o - %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1031 -filetype=asm -o - %s | FileCheck %s

declare <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.v4f32(ptr addrspace(8), i32, i32, i32)
declare void @llvm.amdgcn.raw.ptr.buffer.store.v4f32(<4 x float>, ptr addrspace(8), i32, i32, i32)

; The first four cases establish the bit-to-field map one field at a time; the pair cases pin the
; order the AsmPrinter/encoder emits them in, which is what the MLIR-side expectations quote.
; CHECK-LABEL: ld_a1:
; CHECK: buffer_load_dwordx4 v[0:3], off, s[0:3], 0 glc
; CHECK-NOT: slc
; CHECK-NOT: dlc
define amdgpu_kernel void @ld_a1(<4 x float> addrspace(1)* %out, ptr addrspace(8) %rsrc) {
  %v = call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.v4f32(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 1)
  store <4 x float> %v, ptr addrspace(1) %out, align 16
  ret void
}

; CHECK-LABEL: ld_a2:
; CHECK: buffer_load_dwordx4 v[0:3], off, s[0:3], 0 slc
; CHECK-NOT: glc
; CHECK-NOT: dlc
define amdgpu_kernel void @ld_a2(<4 x float> addrspace(1)* %out, ptr addrspace(8) %rsrc) {
  %v = call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.v4f32(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 2)
  store <4 x float> %v, ptr addrspace(1) %out, align 16
  ret void
}

; CHECK-LABEL: ld_a4:
; CHECK: buffer_load_dwordx4 v[0:3], off, s[0:3], 0 dlc
; CHECK-NOT: glc
; CHECK-NOT: slc
define amdgpu_kernel void @ld_a4(<4 x float> addrspace(1)* %out, ptr addrspace(8) %rsrc) {
  %v = call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.v4f32(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 4)
  store <4 x float> %v, ptr addrspace(1) %out, align 16
  ret void
}

; CHECK-LABEL: ld_a3:
; CHECK: buffer_load_dwordx4 v[0:3], off, s[0:3], 0 glc slc
; CHECK-NOT: dlc
define amdgpu_kernel void @ld_a3(<4 x float> addrspace(1)* %out, ptr addrspace(8) %rsrc) {
  %v = call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.v4f32(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 3)
  store <4 x float> %v, ptr addrspace(1) %out, align 16
  ret void
}

; All three fields at once, in the order the encoder prints them.
; CHECK-LABEL: ld_a7:
; CHECK: buffer_load_dwordx4 v[0:3], off, s[0:3], 0 glc slc dlc
define amdgpu_kernel void @ld_a7(<4 x float> addrspace(1)* %out, ptr addrspace(8) %rsrc) {
  %v = call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.v4f32(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 7)
  store <4 x float> %v, ptr addrspace(1) %out, align 16
  ret void
}

; aux=16 is CDNA's SC1 position; GFX10 has no field there, so nothing is emitted. This is the
; case that makes the CDNA table wrong for RDNA2.
; CHECK-LABEL: ld_a16:
; CHECK: buffer_load_dwordx4 v[0:3], off, s[0:3], 0{{$}}
; CHECK-NOT: glc
; CHECK-NOT: slc
; CHECK-NOT: dlc
define amdgpu_kernel void @ld_a16(<4 x float> addrspace(1)* %out, ptr addrspace(8) %rsrc) {
  %v = call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.v4f32(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 16)
  store <4 x float> %v, ptr addrspace(1) %out, align 16
  ret void
}

; Store path carries the same three fields.
; CHECK-LABEL: st_a7:
; CHECK: buffer_store_dwordx4 v[0:3], off, s[0:3], 0 glc slc dlc
define amdgpu_kernel void @st_a7(<4 x float> addrspace(1)* %in, ptr addrspace(8) %rsrc) {
  %v = load <4 x float>, ptr addrspace(1) %in, align 16
  call void @llvm.amdgcn.raw.ptr.buffer.store.v4f32(<4 x float> %v, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 7)
  ret void
}

; The values Triton actually emits on RDNA2 (see getCtrlBitsForCacheModifierOnRDNA2), each checked
; to survive into a distinct instruction encoding:
;   5 = GLC|DLC  -> .cg and .cv loads
;   2 = SLC      -> .cs loads and stores
;   4 = DLC      -> .wt stores
; CHECK-LABEL: ld_a5:
; CHECK: buffer_load_dwordx4 v[0:3], off, s[0:3], 0 glc dlc
; CHECK-NOT: slc
define amdgpu_kernel void @ld_a5(<4 x float> addrspace(1)* %out, ptr addrspace(8) %rsrc) {
  %v = call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.v4f32(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 5)
  store <4 x float> %v, ptr addrspace(1) %out, align 16
  ret void
}

; CHECK-LABEL: st_a2:
; CHECK: buffer_store_dwordx4 v[0:3], off, s[0:3], 0 slc
; CHECK-NOT: glc
; CHECK-NOT: dlc
define amdgpu_kernel void @st_a2(<4 x float> addrspace(1)* %in, ptr addrspace(8) %rsrc) {
  %v = load <4 x float>, ptr addrspace(1) %in, align 16
  call void @llvm.amdgcn.raw.ptr.buffer.store.v4f32(<4 x float> %v, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 2)
  ret void
}

; CHECK-LABEL: st_a4:
; CHECK: buffer_store_dwordx4 v[0:3], off, s[0:3], 0 dlc
; CHECK-NOT: glc
; CHECK-NOT: slc
define amdgpu_kernel void @st_a4(<4 x float> addrspace(1)* %in, ptr addrspace(8) %rsrc) {
  %v = load <4 x float>, ptr addrspace(1) %in, align 16
  call void @llvm.amdgcn.raw.ptr.buffer.store.v4f32(<4 x float> %v, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 4)
  ret void
}
