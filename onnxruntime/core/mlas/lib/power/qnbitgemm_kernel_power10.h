/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qnbitgemm_kernel_power10.h

Abstract:

    This module includes function declarations and common helpers for the
    POWER10 (VSX + MMA) implementation of SQNBitGemm.

    Packed B layout for SQNBIT_CompInt8 (4-bit weights):

    The columns of B are grouped into column groups of 4 (the trailing group may
    hold 1-3 columns). A group with `w` columns occupies `w * ldb` bytes, where
    `ldb = BlockCountK * BlkLen / 2`, so the shared `n * ldb` column offsets stay
    valid for any `n` that is a multiple of 4. Within a group the data is ordered
    by [k_blk][k8 step]. Each k8 step holds `4 * w` bytes, and byte `4 * j + t`
    stores B[n_j][k + t] in its low nibble and B[n_j][k + t + 4] in its high
    nibble. After `vec_and(v, 0x0F)` and `vec_sr(v, 4)` the two halves are
    directly usable as the unsigned (XB) operand of `xvi8ger4pp`.

    The scales are reordered to the same [colgroup][k_blk][w] order. The block
    sums (-scale * zero_point) use the standard [N/16][BlockCountK][16] layout.

--*/

#pragma once

#include <altivec.h>

#include "qnbitgemm.h"

namespace sqnbitgemm_power10
{

typedef __vector unsigned char MLAS_VUC;
typedef __vector signed char MLAS_VSC;
typedef __vector signed short MLAS_VSS;
typedef __vector signed int MLAS_VSI;
typedef __vector float MLAS_VF;

constexpr size_t kColGroupWidth = 4;

size_t
SQ4BitGemmKernel_BlkSum_CompInt8(
    size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ABlockSum,
    const float* QuantBBlkSum
);

}  // namespace sqnbitgemm_power10
