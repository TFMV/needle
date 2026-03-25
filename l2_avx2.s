// l2_avx2.s
#include "textflag.h"

// func l2Float32AVX2(a, b *float32, dim int) float32
TEXT ·l2Float32AVX2(SB), NOSPLIT, $0-32
    MOVQ a_base+0(FP), R8      // R8 = a
    MOVQ b_base+8(FP), R9      // R9 = b
    MOVQ dim+16(FP), CX       // CX = dim
    VXORPS Y0, Y0, Y0         // Y0 = accumulator, zero it out

    CMPQ CX, $8
    JL   tail                 // if dim < 8, jump to tail

loop:
    VMOVUPS (R8), Y1          // load 8 floats from a
    VMOVUPS (R9), Y2          // load 8 floats from b
    VSUBPS  Y2, Y1, Y1        // Y1 = a - b
    VMULPS  Y1, Y1, Y1        // Y1 = (a - b)^2
    VADDPS  Y1, Y0, Y0        // Y0 += (a - b)^2

    ADDQ $32, R8              // move pointers forward by 8*4 bytes
    ADDQ $32, R9
    SUBQ $8, CX               // decrement counter
    CMPQ CX, $8
    JGE  loop                 // if counter >= 8, loop again

tail:
    CMPQ CX, $0
    JE   done                 // if dim is 0, we're done

    // Scalar tail
scalar_loop:
    MOVSS (R8), X1
    MOVSS (R9), X2
    SUBSS X2, X1
    MULSS X1, X1
    ADDSS X1, X0
    ADDQ $4, R8
    ADDQ $4, R9
    SUBQ $1, CX
    JNE  scalar_loop

done:
    // Horizontal sum of Y0
    VEXTRACTF128 $1, Y0, X1   // extract upper 128 bits of Y0 into X1
    VADDPS       X1, X0, X0   // add lower 128 of Y0 and X1
    VHADDPS      X0, X0, X0   // horizontal add
    VHADDPS      X0, X0, X0   // horizontal add again
    VMOVSS       X0, ret+24(FP) // move result to return value
    VZEROUPPER
    RET
