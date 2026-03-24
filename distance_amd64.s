//go:build amd64

#include "textflag.h"

// func l2Float32(a, b []float32) float32
TEXT ·l2Float32(SB), NOSPLIT, $0-56
    MOVQ a_base+0(FP), AX
    MOVQ a_len+8(FP), CX
    MOVQ b_base+24(FP), DX

    VZEROUPPER

    PXOR X0, X0

    // Process 8 elements at a time
    SUBQ $8, CX
    JL   tail

loop:
    VMOVUPS (AX), Y1
    VMOVUPS (DX), Y2
    VSUBPS  Y2, Y1, Y1
    VFMADD231PS Y1, Y1, Y0

    ADDQ $32, AX
    ADDQ $32, DX

    SUBQ $8, CX
    JGE  loop

tail:
    ADDQ $8, CX
    JZ   done

tail_loop:
    MOVSS (AX), X1
    MOVSS (DX), X2
    SUBSS X2, X1
    MULSS X1, X1
    ADDSS X1, X0

    ADDQ $4, AX
    ADDQ $4, DX

    DECQ CX
    JNE  tail_loop

done:
    VEXTRACTF128 $1, Y0, X1
    VADDPS  X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0
    VMOVSS X0, ret+48(FP)
    VZEROUPPER
    RET
