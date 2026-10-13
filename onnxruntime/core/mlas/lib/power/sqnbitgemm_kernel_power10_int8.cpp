/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_power10_int8.cpp

Abstract:

    This module implements the SQNBIT_CompInt8 kernel for 4-bit weights on
    POWER10. Signed int8 A is multiplied by unsigned 4-bit B (0..15) with the
    MMA instruction xvi8ger4pp (signed x unsigned bytes, the equivalent of x86
    vpdpbusd). The zero point is applied afterwards by an SGEMM pass over the
    per-block sums, exactly like the x86 BlkSum kernels:

        C = sum_blk sa * sb * dot(qa, qb) + Bias + ABlockSum x QuantBBlkSum

    where QuantBBlkSum = -sb * zp and ABlockSum = sa * sum(qa).

    See qnbitgemm_kernel_power10.h for the packed B layout.

--*/

#include <algorithm>
#include <cassert>
#include <vector>

#include "qnbitgemm.h"
#include "qnbitgemm_kernel_power10.h"

namespace sqnbitgemm_power10
{

namespace
{

constexpr size_t kRowPanel = 4;   // rows per MMA A operand
constexpr size_t kTileCols = 16;  // columns per tile (4 column groups)

MLAS_FORCEINLINE MLAS_VUC
LoadBytes(const uint8_t* p, size_t count)
{
    if (count == 16) {
        return vec_xl(0, p);
    }
    return vec_xl_len(const_cast<uint8_t*>(p), count);
}

MLAS_FORCEINLINE MLAS_VF
LoadFloats(const float* p, size_t count)
{
    if (count == 4) {
        return vec_xl(0, p);
    }
    return vec_xl_len(const_cast<float*>(p), count * sizeof(float));
}

MLAS_FORCEINLINE void
StoreFloats(float* p, MLAS_VF v, size_t count)
{
    if (count == 4) {
        vec_xst(v, 0, p);
    } else {
        vec_xst_len(v, p, count * sizeof(float));
    }
}

// Packs rows of quantized A into panels of 4 rows. For every 8 values of K a panel holds two 16-byte
// vectors: {r0 k0-3, r1 k0-3, r2 k0-3, r3 k0-3} followed by {r0 k4-7, ..., r3 k4-7}, which are the XA
// operands paired with the low and high nibbles of the packed B data. Missing rows are zero filled.
void
PackQuantA(const int8_t* QuantA, size_t lda, size_t CountM, size_t CountKPadded, int8_t* PackedA)
{
    typedef __vector unsigned long long VUL;
    const MLAS_VSI zero = vec_splats(0);

    for (size_t m = 0; m < CountM; m += kRowPanel) {
        const size_t rows = std::min(CountM - m, kRowPanel);
        const int8_t* a[kRowPanel];
        for (size_t r = 0; r < kRowPanel; ++r) {
            a[r] = r < rows ? QuantA + (m + r) * lda : nullptr;
        }

        for (size_t k = 0; k < CountKPadded; k += 16) {
            MLAS_VSI v[kRowPanel];
            for (size_t r = 0; r < kRowPanel; ++r) {
                v[r] = a[r] != nullptr ? reinterpret_cast<MLAS_VSI>(vec_xl(0, a[r] + k)) : zero;
            }
            // Word w of row r holds k values 4w .. 4w+3.
            const VUL m01h = reinterpret_cast<VUL>(vec_mergeh(v[0], v[1]));  // r0w0 r1w0 r0w1 r1w1
            const VUL m23h = reinterpret_cast<VUL>(vec_mergeh(v[2], v[3]));  // r2w0 r3w0 r2w1 r3w1
            const VUL m01l = reinterpret_cast<VUL>(vec_mergel(v[0], v[1]));  // r0w2 r1w2 r0w3 r1w3
            const VUL m23l = reinterpret_cast<VUL>(vec_mergel(v[2], v[3]));  // r2w2 r3w2 r2w3 r3w3
            vec_xst(reinterpret_cast<MLAS_VUC>(vec_mergeh(m01h, m23h)), 0, reinterpret_cast<uint8_t*>(PackedA));
            vec_xst(reinterpret_cast<MLAS_VUC>(vec_mergel(m01h, m23h)), 16, reinterpret_cast<uint8_t*>(PackedA));
            vec_xst(reinterpret_cast<MLAS_VUC>(vec_mergeh(m01l, m23l)), 32, reinterpret_cast<uint8_t*>(PackedA));
            vec_xst(reinterpret_cast<MLAS_VUC>(vec_mergel(m01l, m23l)), 48, reinterpret_cast<uint8_t*>(PackedA));
            PackedA += 64;
        }
    }
}

int8_t*
GetPackedABuffer(size_t Size)
{
    static thread_local std::vector<int8_t> buffer;
    if (buffer.size() < Size) {
        buffer.resize(Size);
    }
    return buffer.data();
}

//
// Computes a tile of up to (4 * RowPanels) rows x (4 * ColGroups) columns of C with MMA.
// Only the last column group may be narrower than 4 columns (LastGroupWidth).
//
template <size_t RowPanels, size_t ColGroups>
MLAS_FORCEINLINE void
ComputeTileMMA(
    size_t BlkLen,
    size_t BlockCountK,
    const int8_t* PackedA,
    size_t PackedAPanelStride,
    const float* QuantAScale,
    const uint8_t* QuantBData,
    size_t ldb,
    const float* QuantBScale,
    size_t LastGroupWidth,
    const float* Bias,
    float* C,
    size_t ldc,
    size_t CountM,
    size_t CountN
)
{
    constexpr size_t TileRows = RowPanels * kRowPanel;
    const MLAS_VUC low_mask = vec_splats(static_cast<unsigned char>(0x0F));
    const MLAS_VUC shift4 = vec_splats(static_cast<unsigned char>(4));
    const size_t StepsPerBlk = BlkLen / 8;

    size_t width[ColGroups];
    const uint8_t* b_ptr[ColGroups];
    const float* b_scale[ColGroups];
    for (size_t g = 0; g < ColGroups; ++g) {
        width[g] = (g + 1 == ColGroups) ? LastGroupWidth : kColGroupWidth;
        b_ptr[g] = QuantBData + g * kColGroupWidth * ldb;
        b_scale[g] = QuantBScale + g * kColGroupWidth * BlockCountK;
    }

    // Kept in memory: while the 8 accumulators are live they occupy VSR0-31.
    MLAS_VF c_acc[TileRows][ColGroups];
    for (size_t r = 0; r < TileRows; ++r) {
        for (size_t g = 0; g < ColGroups; ++g) {
            c_acc[r][g] = vec_splats(0.0f);
        }
    }

    const int8_t* a_ptr = PackedA;

    for (size_t k_blk = 0; k_blk < BlockCountK; ++k_blk) {
        __vector_quad acc[RowPanels][ColGroups];
        for (size_t p = 0; p < RowPanels; ++p) {
            for (size_t g = 0; g < ColGroups; ++g) {
                __builtin_mma_xxsetaccz(&acc[p][g]);
            }
        }

        for (size_t step = 0; step < StepsPerBlk; ++step) {
            MLAS_VUC b_lo[ColGroups], b_hi[ColGroups];
            for (size_t g = 0; g < ColGroups; ++g) {
                const MLAS_VUC bv = LoadBytes(b_ptr[g], 4 * width[g]);
                b_ptr[g] += 4 * width[g];
                b_lo[g] = vec_and(bv, low_mask);
                b_hi[g] = vec_sr(bv, shift4);
            }
            for (size_t p = 0; p < RowPanels; ++p) {
                const MLAS_VUC a_lo = vec_xl(0, reinterpret_cast<const uint8_t*>(a_ptr + p * PackedAPanelStride));
                const MLAS_VUC a_hi = vec_xl(16, reinterpret_cast<const uint8_t*>(a_ptr + p * PackedAPanelStride));
                for (size_t g = 0; g < ColGroups; ++g) {
                    __builtin_mma_xvi8ger4pp(&acc[p][g], a_lo, b_lo[g]);
                    __builtin_mma_xvi8ger4pp(&acc[p][g], a_hi, b_hi[g]);
                }
            }
            a_ptr += 32;
        }

        MLAS_VF scale_b[ColGroups];
        for (size_t g = 0; g < ColGroups; ++g) {
            scale_b[g] = LoadFloats(b_scale[g] + k_blk * width[g], width[g]);
        }

        for (size_t p = 0; p < RowPanels; ++p) {
            for (size_t g = 0; g < ColGroups; ++g) {
                MLAS_VSI result[4];
                __builtin_mma_disassemble_acc(result, &acc[p][g]);
                for (size_t i = 0; i < kRowPanel; ++i) {
                    const size_t row = p * kRowPanel + i;
                    // Rows past CountM are zero-padded in PackedA; skip reading their scales.
                    const float scale_a = row < CountM ? QuantAScale[row * BlockCountK + k_blk] : 0.0f;
                    const MLAS_VF scale = vec_mul(vec_splats(scale_a), scale_b[g]);
                    c_acc[row][g] = vec_madd(vec_ctf(result[i], 0), scale, c_acc[row][g]);
                }
            }
        }
    }

    for (size_t row = 0; row < std::min(CountM, TileRows); ++row) {
        float* c_row = C + row * ldc;
        for (size_t g = 0; g < ColGroups; ++g) {
            const size_t col = g * kColGroupWidth;
            if (col >= CountN) {
                break;
            }
            const size_t count = std::min(CountN - col, kColGroupWidth);
            MLAS_VF v = c_acc[row][g];
            if (Bias != nullptr) {
                v = vec_add(v, LoadFloats(Bias + col, count));
            }
            StoreFloats(c_row + col, v, count);
        }
    }
}

template <size_t RowPanels>
void
ComputeRowsMMA(
    size_t BlkLen,
    size_t BlockCountK,
    const int8_t* PackedA,
    size_t PackedAPanelStride,
    const float* QuantAScale,
    const uint8_t* QuantBData,
    const float* QuantBScale,
    const float* Bias,
    float* C,
    size_t ldc,
    size_t CountM,
    size_t CountN
)
{
    const size_t ldb = BlockCountK * MlasQNBitBlkDataSizeInBytes(4, BlkLen);

    for (size_t n = 0; n < CountN; n += kTileCols) {
        const size_t cols = std::min(CountN - n, kTileCols);
        const size_t groups = MlasDivRoundup(cols, kColGroupWidth);
        const size_t last_width = cols - (groups - 1) * kColGroupWidth;

        const uint8_t* b = QuantBData + n * ldb;
        const float* b_scale = QuantBScale + n * BlockCountK;
        const float* bias = Bias != nullptr ? Bias + n : nullptr;
        float* c = C + n;

#define MLAS_P10_TILE(G) \
    ComputeTileMMA<RowPanels, G>(BlkLen, BlockCountK, PackedA, PackedAPanelStride, QuantAScale, b, ldb, b_scale, last_width, bias, c, ldc, CountM, cols)
        switch (groups) {
            case 4:
                MLAS_P10_TILE(4);
                break;
            case 3:
                MLAS_P10_TILE(3);
                break;
            case 2:
                MLAS_P10_TILE(2);
                break;
            default:
                MLAS_P10_TILE(1);
                break;
        }
#undef MLAS_P10_TILE
    }
}

//
// M == 1 (decode): VSX vmsummbm (signed x unsigned bytes, 4-way dot product per word). The 16 bytes
// of a packed B step hold 4 columns x 4 k values, so one splatted A word yields 4 column partial sums.
//
template <size_t ColGroups>
MLAS_FORCEINLINE void
ComputeM1Tile(
    size_t BlkLen,
    size_t BlockCountK,
    const int8_t* QuantA,
    const float* QuantAScale,
    const uint8_t* QuantBData,
    size_t ldb,
    const float* QuantBScale,
    size_t LastGroupWidth,
    const float* Bias,
    float* C,
    size_t CountN
)
{
    const MLAS_VUC low_mask = vec_splats(static_cast<unsigned char>(0x0F));
    const MLAS_VUC shift4 = vec_splats(static_cast<unsigned char>(4));

    size_t width[ColGroups];
    const uint8_t* b_ptr[ColGroups];
    const float* b_scale[ColGroups];
    MLAS_VF c_acc[ColGroups];
    for (size_t g = 0; g < ColGroups; ++g) {
        width[g] = (g + 1 == ColGroups) ? LastGroupWidth : kColGroupWidth;
        b_ptr[g] = QuantBData + g * kColGroupWidth * ldb;
        b_scale[g] = QuantBScale + g * kColGroupWidth * BlockCountK;
        c_acc[g] = vec_splats(0.0f);
    }

    const int8_t* a_ptr = QuantA;

    for (size_t k_blk = 0; k_blk < BlockCountK; ++k_blk) {
        MLAS_VSI sum[ColGroups];
        for (size_t g = 0; g < ColGroups; ++g) {
            sum[g] = vec_splats(0);
        }

        // BlkLen is a multiple of 16: two packed B steps per 16 bytes of A.
        for (size_t k = 0; k < BlkLen; k += 16) {
            const MLAS_VSI av = reinterpret_cast<MLAS_VSI>(vec_xl(0, a_ptr));
            a_ptr += 16;
            const MLAS_VSC a0 = reinterpret_cast<MLAS_VSC>(vec_splat(av, 0));
            const MLAS_VSC a1 = reinterpret_cast<MLAS_VSC>(vec_splat(av, 1));
            const MLAS_VSC a2 = reinterpret_cast<MLAS_VSC>(vec_splat(av, 2));
            const MLAS_VSC a3 = reinterpret_cast<MLAS_VSC>(vec_splat(av, 3));

            for (size_t g = 0; g < ColGroups; ++g) {
                const size_t step_bytes = 4 * width[g];
                const MLAS_VUC b0 = LoadBytes(b_ptr[g], step_bytes);
                const MLAS_VUC b1 = LoadBytes(b_ptr[g] + step_bytes, step_bytes);
                b_ptr[g] += 2 * step_bytes;
                sum[g] = vec_msum(a0, vec_and(b0, low_mask), sum[g]);
                sum[g] = vec_msum(a1, vec_sr(b0, shift4), sum[g]);
                sum[g] = vec_msum(a2, vec_and(b1, low_mask), sum[g]);
                sum[g] = vec_msum(a3, vec_sr(b1, shift4), sum[g]);
            }
        }

        const MLAS_VF scale_a = vec_splats(QuantAScale[k_blk]);
        for (size_t g = 0; g < ColGroups; ++g) {
            const MLAS_VF scale = vec_mul(scale_a, LoadFloats(b_scale[g] + k_blk * width[g], width[g]));
            c_acc[g] = vec_madd(vec_ctf(sum[g], 0), scale, c_acc[g]);
        }
    }

    for (size_t g = 0; g < ColGroups; ++g) {
        const size_t col = g * kColGroupWidth;
        const size_t count = std::min(CountN - col, kColGroupWidth);
        MLAS_VF v = c_acc[g];
        if (Bias != nullptr) {
            v = vec_add(v, LoadFloats(Bias + col, count));
        }
        StoreFloats(C + col, v, count);
    }
}

void
ComputeM1(
    size_t BlkLen,
    size_t BlockCountK,
    const int8_t* QuantA,
    const float* QuantAScale,
    const uint8_t* QuantBData,
    const float* QuantBScale,
    const float* Bias,
    float* C,
    size_t CountN
)
{
    const size_t ldb = BlockCountK * MlasQNBitBlkDataSizeInBytes(4, BlkLen);

    for (size_t n = 0; n < CountN; n += kTileCols) {
        const size_t cols = std::min(CountN - n, kTileCols);
        const size_t groups = MlasDivRoundup(cols, kColGroupWidth);
        const size_t last_width = cols - (groups - 1) * kColGroupWidth;

        const uint8_t* b = QuantBData + n * ldb;
        const float* b_scale = QuantBScale + n * BlockCountK;
        const float* bias = Bias != nullptr ? Bias + n : nullptr;

#define MLAS_P10_M1_TILE(G) \
    ComputeM1Tile<G>(BlkLen, BlockCountK, QuantA, QuantAScale, b, ldb, b_scale, last_width, bias, C + n, cols)
        switch (groups) {
            case 4:
                MLAS_P10_M1_TILE(4);
                break;
            case 3:
                MLAS_P10_M1_TILE(3);
                break;
            case 2:
                MLAS_P10_M1_TILE(2);
                break;
            default:
                MLAS_P10_M1_TILE(1);
                break;
        }
#undef MLAS_P10_M1_TILE
    }
}

}  // namespace

size_t
SQ4BitGemmKernel_BlkSum_CompInt8(
    size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* /*QuantBZeroPoint*/,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t /*CountK*/,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ABlockSum,
    const float* QuantBBlkSum
)
{
    assert(BlkLen >= 16 && BlkLen % 16 == 0);

    const int8_t* quant_a = reinterpret_cast<const int8_t*>(QuantA);
    const uint8_t* quant_b = reinterpret_cast<const uint8_t*>(QuantBData);
    const size_t CountKPadded = BlockCountK * BlkLen;

    if (CountM == 1) {
        ComputeM1(BlkLen, BlockCountK, quant_a, QuantAScale, quant_b, QuantBScale, Bias, C, CountN);
    } else {
        const size_t PanelCount = MlasDivRoundup(CountM, kRowPanel);
        const size_t PanelStride = kRowPanel * CountKPadded;
        int8_t* packed_a = GetPackedABuffer(PanelCount * PanelStride);
        PackQuantA(quant_a, CountKPadded, CountM, CountKPadded, packed_a);

        for (size_t m = 0; m < CountM; m += 2 * kRowPanel) {
            const size_t rows = std::min(CountM - m, 2 * kRowPanel);
            const int8_t* a = packed_a + (m / kRowPanel) * PanelStride;
            const float* a_scale = QuantAScale + m * BlockCountK;
            float* c = C + m * ldc;
            if (rows > kRowPanel) {
                ComputeRowsMMA<2>(BlkLen, BlockCountK, a, PanelStride, a_scale, quant_b, QuantBScale, Bias, c, ldc, rows, CountN);
            } else {
                ComputeRowsMMA<1>(BlkLen, BlockCountK, a, PanelStride, a_scale, quant_b, QuantBScale, Bias, c, ldc, rows, CountN);
            }
        }
    }

    // Zero point correction: C += ABlockSum x QuantBBlkSum, with QuantBBlkSum in SGEMM packed B layout.
    float* c_blk = C;
    const float* a_blksum_row = ABlockSum;
    size_t RowsRemaining = CountM;
    while (RowsRemaining > 0) {
        const size_t RowsHandled = GetMlasPlatform().GemmFloatKernel(
            a_blksum_row, QuantBBlkSum, c_blk, BlockCountK, RowsRemaining, CountN, BlockCountK, ldc, 1.f, false
        );
        c_blk += ldc * RowsHandled;
        a_blksum_row += BlockCountK * RowsHandled;
        RowsRemaining -= RowsHandled;
    }

    return CountM;
}

}  // namespace sqnbitgemm_power10
