//go:build amd64
// +build amd64

#include "textflag.h"

// func euclideanSquaredAVX2(a, b []float64) float64
TEXT ·euclideanSquaredAVX2(SB), NOSPLIT, $0-56
    MOVQ a_base+0(FP), R8
    MOVQ a_len+8(FP), R9
    MOVQ b_base+24(FP), R10

    // Y0 will hold the running sums
    VXORPD Y0, Y0, Y0

    XORQ R11, R11 // i = 0

loop:
    MOVQ R9, R12
    SUBQ $4, R12
    CMPQ R11, R12
    JG tail

    // Process 4 float64s at once
    VMOVUPD (R8)(R11*8), Y1
    VMOVUPD (R10)(R11*8), Y2
    VSUBPD Y2, Y1, Y1
    VMULPD Y1, Y1, Y1
    VADDPD Y1, Y0, Y0

    ADDQ $4, R11
    JMP loop

tail:
    CMPQ R11, R9
    JGE done

    VMOVSD (R8)(R11*8), X1
    VMOVSD (R10)(R11*8), X2
    VSUBSD X2, X1, X1
    VMULSD X1, X1, X1
    VADDSD X1, X0, X0

    INCQ R11
    JMP tail

done:
    // Extract upper 128-bits into X1
    VEXTRACTF128 $1, Y0, X1
    VADDPD X1, X0, X0
    
    // Horizontal add of the adjacent 64-bit values in X0
    VHADDPD X0, X0, X0

    VMOVSD X0, ret+48(FP)
    VZEROUPPER
    RET
