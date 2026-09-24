/* Handwritten alternative schedule: keep path masks in 32-bit GPRs and
 * transfer only final masks to AVX-512 k registers. This exposes independent
 * integer Boolean work alongside floating-point accumulation.
 * Same ABI, inputs, outputs and temporal arithmetic as score8_asm_mask.
 * zmm0=signal_count,1=equity,2..5=counts,6..9=sums,10..13=squares;
 * zmm14=threshold,15=xor sign,16=abs and,17=FP one,18=int one,19=zero;
 * zmm20=prediction,21=a,22=a*a. k7=inclusive.
 */
.text
.p2align 5
.globl score8_asm_gpr
.type score8_asm_gpr, @function
score8_asm_gpr:
    push rbx
    push r12
    push r13
    push r14
    push r15
    mov r8, [rsi]
    mov r9, [rsi+8]
    mov r10, [rsi+16]
    mov r11, [rsi+24]
    vmovupd zmm14, [rdx]
    vmovdqu64 zmm15, [rdx+64]
    vmovdqu64 zmm16, [rdx+128]
    kmovw k7, [rdx+192]
    mov eax, 1
    vpbroadcastq zmm18, rax
    mov rax, 0x3ff0000000000000
    vpbroadcastq zmm17, rax
    vpxord zmm0, zmm0, zmm0
    vmovapd zmm1, zmm17
    vpxord zmm2, zmm2, zmm2
    vpxord zmm3, zmm3, zmm3
    vpxord zmm4, zmm4, zmm4
    vpxord zmm5, zmm5, zmm5
    vpxord zmm6, zmm6, zmm6
    vpxord zmm7, zmm7, zmm7
    vpxord zmm8, zmm8, zmm8
    vpxord zmm9, zmm9, zmm9
    vpxord zmm10, zmm10, zmm10
    vpxord zmm11, zmm11, zmm11
    vpxord zmm12, zmm12, zmm12
    vpxord zmm13, zmm13, zmm13
    vpxord zmm19, zmm19, zmm19
    xor r12d, r12d                 /* eligible bars */
    xor r13d, r13d                 /* index */
    xor r14d, r14d                 /* last signal */
    xor r15d, r15d                 /* previous position */
    test r8, r8
    jz .Lstore_g
.p2align 5
.Lbar_g:
    vbroadcastsd zmm20, [rdi+r13*8]
    vpternlogq zmm20, zmm15, zmm16, 0x28   /* (prediction XOR sign) AND abs */
    vcmppd k1, zmm20, zmm14, 30
    vcmppd k0{k7}, zmm20, zmm14, 0
    korw k1, k1, k0
    kmovw eax, k1
    vpaddq zmm0{k1}, zmm0, zmm18
    mov edx, r14d
    mov r14d, eax
    cmp byte ptr [r9+r13], 0
    jne .Lpos_g
    xor edx, edx
.Lpos_g:
    andn esi, r15d, edx            /* entry */
    and r15d, edx                  /* continuation */
    kmovw k2, r15d
    mov r15d, edx
    lea rbx, [r13+1]
    cmp rbx, r8
    jae .Llast_g
    cmp byte ptr [r9+rbx], 0
    je .Llast_g
    andn ebx, eax, edx             /* exit */
    jmp .Lpath_g
.Llast_g:
    mov ebx, edx
.Lpath_g:
    kmovw k1, esi
    vaddpd zmm21{k1}{z}, zmm19, [r10+r13*8]{1to8}
    vaddpd zmm21{k2}, zmm21, [r11+r13*8]{1to8}
    vaddpd zmm21, zmm17, zmm21
    vmulpd zmm1, zmm1, zmm21
    test r13, r13
    jz .Lnext_g
    cmp byte ptr [r9+r13], 0
    je .Lnext_g
    inc r12
    vmulpd zmm22, zmm21, zmm21
    mov edx, esi
    or edx, ebx
    not edx
    kmovw k1, edx
    vpaddq zmm2{k1}, zmm2, zmm18
    vaddpd zmm6{k1}, zmm6, zmm21
    vaddpd zmm10{k1}, zmm10, zmm22
    andn edx, ebx, esi
    kmovw k2, edx
    vpaddq zmm3{k2}, zmm3, zmm18
    vaddpd zmm7{k2}, zmm7, zmm21
    vaddpd zmm11{k2}, zmm11, zmm22
    andn edx, esi, ebx
    kmovw k3, edx
    vpaddq zmm4{k3}, zmm4, zmm18
    vaddpd zmm8{k3}, zmm8, zmm21
    vaddpd zmm12{k3}, zmm12, zmm22
    and esi, ebx
    kmovw k4, esi
    vpaddq zmm5{k4}, zmm5, zmm18
    vaddpd zmm9{k4}, zmm9, zmm21
    vaddpd zmm13{k4}, zmm13, zmm22
.Lnext_g:
    inc r13
    cmp r13, r8
    jb .Lbar_g
.Lstore_g:
    vmovdqu64 [rcx], zmm0
    vmovupd [rcx+64], zmm1
    vmovdqu64 [rcx+128], zmm2
    vmovdqu64 [rcx+192], zmm3
    vmovdqu64 [rcx+256], zmm4
    vmovdqu64 [rcx+320], zmm5
    vmovupd [rcx+384], zmm6
    vmovupd [rcx+448], zmm7
    vmovupd [rcx+512], zmm8
    vmovupd [rcx+576], zmm9
    vmovupd [rcx+640], zmm10
    vmovupd [rcx+704], zmm11
    vmovupd [rcx+768], zmm12
    vmovupd [rcx+832], zmm13
    mov [rcx+896], r12
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
.size score8_asm_gpr, .-score8_asm_gpr
.section .note.GNU-stack,"",@progbits
