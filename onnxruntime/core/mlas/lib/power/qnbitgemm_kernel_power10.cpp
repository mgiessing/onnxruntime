/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qnbitgemm_kernel_power10.cpp

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication kernel dispatch for POWER10 (VSX + MMA): B packing,
    workspace sizing, A quantization, and the SQNBIT_CompFp32 kernels.

    The SQNBIT_CompInt8 compute kernel is in sqnbitgemm_kernel_power10_int8.cpp.

--*/

#include "qnbitgemm_kernel_power10.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

#include "qnbitgemm.h"
#include "sqnbitgemm_q8_block.h"

namespace sqnbitgemm_power10
{

namespace
{

constexpr size_t kBlkBitWidth = 4;

// PackedQuantBDataStruct aligns the packed data to 128 bytes on POWER.
constexpr size_t kPackedQuantBDataAlignment = 128;

//
// Vector helpers.
//

MLAS_FORCEINLINE MLAS_VF
LoadFloats(const float* p, ptrdiff_t count)
{
    if (count >= 4) {
        return vec_xl(0, p);
    }
    if (count <= 0) {
        return vec_splats(0.0f);
    }
    return vec_xl_len(const_cast<float*>(p), static_cast<size_t>(count) * sizeof(float));
}

MLAS_FORCEINLINE void
Transpose4x4(MLAS_VF& a0, MLAS_VF& a1, MLAS_VF& a2, MLAS_VF& a3)
{
    typedef __vector unsigned long long VUL;
    const MLAS_VF t0 = vec_mergeh(a0, a1);  // a0_0 a1_0 a0_1 a1_1
    const MLAS_VF t1 = vec_mergel(a0, a1);  // a0_2 a1_2 a0_3 a1_3
    const MLAS_VF t2 = vec_mergeh(a2, a3);  // a2_0 a3_0 a2_1 a3_1
    const MLAS_VF t3 = vec_mergel(a2, a3);  // a2_2 a3_2 a2_3 a3_3
    a0 = reinterpret_cast<MLAS_VF>(vec_mergeh(reinterpret_cast<VUL>(t0), reinterpret_cast<VUL>(t2)));
    a1 = reinterpret_cast<MLAS_VF>(vec_mergel(reinterpret_cast<VUL>(t0), reinterpret_cast<VUL>(t2)));
    a2 = reinterpret_cast<MLAS_VF>(vec_mergeh(reinterpret_cast<VUL>(t1), reinterpret_cast<VUL>(t3)));
    a3 = reinterpret_cast<MLAS_VF>(vec_mergel(reinterpret_cast<VUL>(t1), reinterpret_cast<VUL>(t3)));
}

MLAS_FORCEINLINE float
HorizontalSum(MLAS_VF v)
{
    return v[0] + v[1] + v[2] + v[3];
}

MLAS_FORCEINLINE uint8_t
GetZeroPoint(const std::byte* QuantBZeroPointCol, size_t k_blk)
{
    const std::byte zp_packed = QuantBZeroPointCol[k_blk / 2];
    return std::to_integer<uint8_t>((k_blk & 1) ? (zp_packed >> 4) : (zp_packed & std::byte{0x0F}));
}

// Converts 16 nibbles stored in natural order (value 2i in the low nibble of byte i, value 2i+1 in the
// high nibble) in the first 8 bytes of `packed` to 4 float vectors holding values 0-3, 4-7, 8-11, 12-15.
MLAS_FORCEINLINE void
Unpack16NibblesToFloat(MLAS_VUC packed, bool high_half, MLAS_VF out[4])
{
    const MLAS_VUC low_mask = vec_splats(static_cast<unsigned char>(0x0F));
    const MLAS_VUC shift = vec_splats(static_cast<unsigned char>(4));
    const MLAS_VUC zero = vec_splats(static_cast<unsigned char>(0));
    const MLAS_VUC lo = vec_and(packed, low_mask);
    const MLAS_VUC hi = vec_sr(packed, shift);
    // Interleave low and high nibbles back into value order.
    const MLAS_VUC values = high_half ? vec_mergel(lo, hi) : vec_mergeh(lo, hi);

    // Zero-extend byte 4q+e into word e.
    static const MLAS_VUC perm[4] = {
        {0, 16, 16, 16, 1, 16, 16, 16, 2, 16, 16, 16, 3, 16, 16, 16},
        {4, 16, 16, 16, 5, 16, 16, 16, 6, 16, 16, 16, 7, 16, 16, 16},
        {8, 16, 16, 16, 9, 16, 16, 16, 10, 16, 16, 16, 11, 16, 16, 16},
        {12, 16, 16, 16, 13, 16, 16, 16, 14, 16, 16, 16, 15, 16, 16, 16},
    };
    for (size_t q = 0; q < 4; ++q) {
        out[q] = vec_ctf(reinterpret_cast<MLAS_VSI>(vec_perm(values, zero, perm[q])), 0);
    }
}

//
// Quantized B data packing.
//

size_t
Q4BitGemmPackQuantBDataSize(
    size_t N,
    size_t K,
    size_t BlkLen,
    bool /*HasZeroPoint*/,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*BackendKernelSelectorConfig*/
)
{
    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t PackedQuantBDataSize = N * BlockCountK * MlasQNBitBlkDataSizeInBytes(kBlkBitWidth, BlkLen);
    if (ComputeType != SQNBIT_CompInt8) {
        return PackedQuantBDataSize;
    }

    const size_t ScaleSize = N * BlockCountK * sizeof(float);
    const size_t BlkSumSize = MlasDivRoundup(N, 16) * BlockCountK * 16 * sizeof(float);
    // Slack for the alignment applied by PackedQuantBDataStruct.
    return (PackedQuantBDataSize + kPackedQuantBDataAlignment - 1) + ScaleSize +
           (BlkSumSize + MlasQNBitQuantBBlkSumAlignment() - 1);
}

// SQNBIT_CompFp32 keeps the source layout: the CompFp32 kernels unpack nibbles in natural order.
void
SQ4BitGemmPackQuantBData(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE /*ComputeType*/,
    const std::byte* QuantBDataBegin,
    std::byte* PackedQuantBDataBegin,
    MLAS_THREADPOOL* ThreadPool,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*BackendKernelSelectorConfig*/
)
{
    assert(BlkLen >= 16 && BlkLen % 16 == 0);

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t ColBytes = BlockCountK * MlasQNBitBlkDataSizeInBytes(kBlkBitWidth, BlkLen);

    MlasTrySimpleParallel(ThreadPool, static_cast<ptrdiff_t>(N), [&](ptrdiff_t n) {
        const std::byte* src = QuantBDataBegin + n * ColBytes;
        std::copy(src, src + ColBytes, PackedQuantBDataBegin + n * ColBytes);
    });
}

// Packs B into the MMA compute order described in qnbitgemm_kernel_power10.h.
void
PackQuantBDataCompInt8(
    size_t N,
    size_t BlockCountK,
    size_t BlkLen,
    const std::byte* QuantBDataBegin,
    std::byte* PackedQuantBDataBegin,
    MLAS_THREADPOOL* ThreadPool
)
{
    const size_t BlkDataSize = MlasQNBitBlkDataSizeInBytes(kBlkBitWidth, BlkLen);
    const size_t ldb = BlockCountK * BlkDataSize;
    const size_t StepsPerBlk = BlkLen / 8;
    const size_t GroupCount = MlasDivRoundup(N, kColGroupWidth);

    MlasTrySimpleParallel(ThreadPool, static_cast<ptrdiff_t>(GroupCount), [&](ptrdiff_t group) {
        const size_t n0 = static_cast<size_t>(group) * kColGroupWidth;
        const size_t width = std::min(kColGroupWidth, N - n0);
        std::byte* dst = PackedQuantBDataBegin + n0 * ldb;

        for (size_t k_blk = 0; k_blk < BlockCountK; ++k_blk) {
            for (size_t step = 0; step < StepsPerBlk; ++step) {
                for (size_t j = 0; j < width; ++j) {
                    // 8 values k .. k+7 of this step are stored in 4 source bytes.
                    const uint8_t* src = reinterpret_cast<const uint8_t*>(
                        QuantBDataBegin + (n0 + j) * ldb + k_blk * BlkDataSize + step * 4
                    );
                    const uint8_t v[8] = {
                        static_cast<uint8_t>(src[0] & 0x0F),
                        static_cast<uint8_t>(src[0] >> 4),
                        static_cast<uint8_t>(src[1] & 0x0F),
                        static_cast<uint8_t>(src[1] >> 4),
                        static_cast<uint8_t>(src[2] & 0x0F),
                        static_cast<uint8_t>(src[2] >> 4),
                        static_cast<uint8_t>(src[3] & 0x0F),
                        static_cast<uint8_t>(src[3] >> 4),
                    };
                    for (size_t t = 0; t < 4; ++t) {
                        dst[4 * j + t] = static_cast<std::byte>(v[t] | (v[t + 4] << 4));
                    }
                }
                dst += 4 * width;
            }
        }
    });
}

// Computes the block sums (-scale * zero_point) and reorders the scales to the packed data order.
// PackedQuantBScale must hold the scales in the source [N][BlockCountK] order on entry.
void
ComputePackBlkSum(
    size_t N,
    size_t BlockCountK,
    float* PackedQuantBScale,
    const std::byte* QuantBZPBegin,
    float* BlockSumBegin,
    MLAS_THREADPOOL* ThreadPool
)
{
    if (N == 0 || BlockCountK == 0) {
        return;
    }

    const std::vector<float> scales(PackedQuantBScale, PackedQuantBScale + N * BlockCountK);
    const size_t ZPCountK = MlasQNBitZeroPointsForBlksSizeInBytes<kBlkBitWidth>(BlockCountK);

    MlasTrySimpleParallel(ThreadPool, static_cast<ptrdiff_t>(N), [&](ptrdiff_t tid) {
        const size_t n = static_cast<size_t>(tid);
        const size_t n0 = n & ~(kColGroupWidth - 1);
        const size_t width = std::min(kColGroupWidth, N - n0);
        const size_t blksum_offset = (n / 16) * BlockCountK * 16 + n % 16;

        for (size_t k_blk = 0; k_blk < BlockCountK; ++k_blk) {
            const float scale = scales[n * BlockCountK + k_blk];
            const uint8_t zp = QuantBZPBegin ? GetZeroPoint(QuantBZPBegin + n * ZPCountK, k_blk) : 8;

            BlockSumBegin[blksum_offset + k_blk * 16] = -scale * zp;
            PackedQuantBScale[n0 * BlockCountK + k_blk * width + (n - n0)] = scale;
        }
    });
}

void
SQ4BitGemmPackQuantBDataAndBlkSum(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE /*ComputeType*/,
    const std::byte* QuantBDataBegin,
    const float* QuantBScaleBegin,
    bool HasZeroPoint,
    const std::byte* QuantBZPBegin,
    PackedQuantBDataStruct<float, 4>& PackedQuantB,
    MLAS_THREADPOOL* ThreadPool,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*BackendKernelSelectorConfig*/
)
{
    assert(BlkLen >= 16 && BlkLen % 16 == 0);

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);

    // Same call protocol as x86: data, scales and zero points may arrive in separate calls. The scales
    // are reordered exactly once, together with the block sum computation.
    if (QuantBDataBegin) {
        PackQuantBDataCompInt8(N, BlockCountK, BlkLen, QuantBDataBegin, PackedQuantB.PackedQuantBData, ThreadPool);
    }

    if (QuantBScaleBegin) {
        std::copy(QuantBScaleBegin, QuantBScaleBegin + N * BlockCountK, PackedQuantB.PackedQuantBScale);
    }

    if ((QuantBScaleBegin && !HasZeroPoint) || QuantBZPBegin) {
        ComputePackBlkSum(N, BlockCountK, PackedQuantB.PackedQuantBScale, QuantBZPBegin, PackedQuantB.QuantBBlkSum, ThreadPool);
    }
}

//
// Workspace size calculation.
//

size_t
QNBitGemmPerGemmWorkspaceSize(
    size_t M,
    size_t /*N*/,
    size_t K,
    size_t BlkLen,
    bool /*HasZeroPoint*/,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    size_t BlkBitWidth,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*BackendKernelSelectorConfig*/
)
{
    if (ComputeType != SQNBIT_CompInt8 || BlkBitWidth != 4) {
        return 0;
    }
    // Quantized A data + scales + block sums, see PerGemmQuantAWorkspace.
    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    return M * BlockCountK * (Q8BlkSize(BlkLen) + sizeof(float));
}

size_t
QNBitGemmPerGemmWorkspaceAlignment(
    size_t /*BlkLen*/,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    return ComputeType == SQNBIT_CompInt8 ? Q8BlkAlignment() : 1;
}

//
// A quantization for SQNBIT_CompInt8.
//

void
QuantizeARowComputeBlkSum_CompInt8(
    size_t BlkLen,
    const float* A,
    size_t CountK,
    std::byte* QuantA,
    float* QuantAScale,
    float* AScaledBlkSum  // scale_k * Sum_blklen(a_i)
)
{
    assert(BlkLen % 16 == 0);

    int8_t* dst = reinterpret_cast<int8_t*>(QuantA);

    for (size_t k = 0; k < CountK; k += BlkLen) {
        const ptrdiff_t step = static_cast<ptrdiff_t>(std::min(BlkLen, CountK - k));
        const float* a = A + k;

        MLAS_VF max_abs = vec_splats(0.0f);
        for (ptrdiff_t kk = 0; kk < step; kk += 4) {
            max_abs = vec_max(max_abs, vec_abs(LoadFloats(a + kk, step - kk)));
        }
        const float max_scalar = std::max(std::max(max_abs[0], max_abs[1]), std::max(max_abs[2], max_abs[3]));

        const float scale = max_scalar / 127.f;
        const float inverse_scale = (max_scalar != 0.0f) ? 127.f / max_scalar : 0.0f;
        const MLAS_VF mul = vec_splats(inverse_scale);

        // Writes the full block; values past CountK load as zero and quantize to zero.
        MLAS_VSI sum = vec_splats(0);
        for (ptrdiff_t kk = 0; kk < static_cast<ptrdiff_t>(BlkLen); kk += 16) {
            MLAS_VSI q[4];
            for (ptrdiff_t i = 0; i < 4; ++i) {
                const MLAS_VF v = vec_mul(LoadFloats(a + kk + 4 * i, step - kk - 4 * i), mul);
                // vec_rint rounds to nearest even, matching the x86 implementation.
                q[i] = vec_cts(vec_rint(v), 0);
            }
            const MLAS_VSC q8 = vec_packs(vec_packs(q[0], q[1]), vec_packs(q[2], q[3]));
            vec_xst(q8, 0, dst + kk);
            sum = vec_sum4s(q8, sum);
        }

        *QuantAScale++ = scale;
        *AScaledBlkSum++ = scale * static_cast<float>(sum[0] + sum[1] + sum[2] + sum[3]);
        dst += BlkLen;
    }
}

//
// SQNBIT_CompFp32 kernels.
//

// Computes C[0..NCols) = A . B[:, 0..NCols) for one row of A. Uses the identity
// sum(a * (q - zp) * s) = s * (sum(a * q) - zp * sum(a)) per block.
template <size_t NCols, bool HasZeroPoint>
MLAS_FORCEINLINE void
ComputeDotProducts_CompFp32(
    size_t BlkLen,
    const float* A,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountK,
    size_t BlockCountK,
    const float* Bias
)
{
    const size_t StrideData = BlockCountK * MlasQNBitBlkDataSizeInBytes(kBlkBitWidth, BlkLen);
    const size_t StrideZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<kBlkBitWidth>(BlockCountK);

    MLAS_VF total[NCols];
    for (size_t i = 0; i < NCols; ++i) {
        total[i] = vec_splats(0.0f);
    }

    for (size_t k = 0, k_blk = 0; k < CountK; k += BlkLen, ++k_blk) {
        const ptrdiff_t k_blk_len = static_cast<ptrdiff_t>(std::min(CountK - k, BlkLen));
        const float* a_blk = A + k;
        const std::byte* b_blk = QuantBData + k_blk * MlasQNBitBlkDataSizeInBytes(kBlkBitWidth, BlkLen);

        MLAS_VF dot[NCols];
        for (size_t i = 0; i < NCols; ++i) {
            dot[i] = vec_splats(0.0f);
        }
        MLAS_VF a_sum = vec_splats(0.0f);

        // 32 values (16 bytes) per iteration, or 16 values (8 bytes) for the BlkLen 16 tail.
        for (ptrdiff_t kk = 0; kk < k_blk_len; kk += 32) {
            const bool full = static_cast<size_t>(kk) + 32 <= BlkLen;
            const size_t value_count = full ? 32 : 16;

            MLAS_VF av[8];
            for (size_t q = 0; q < value_count / 4; ++q) {
                av[q] = LoadFloats(a_blk + kk + 4 * q, k_blk_len - kk - static_cast<ptrdiff_t>(4 * q));
                a_sum = vec_add(a_sum, av[q]);
            }

            for (size_t i = 0; i < NCols; ++i) {
                const unsigned char* b = reinterpret_cast<const unsigned char*>(b_blk + i * StrideData + kk / 2);
                const MLAS_VUC packed = full ? vec_xl(0, b) : vec_xl_len(const_cast<unsigned char*>(b), 8);
                MLAS_VF bv[8];
                Unpack16NibblesToFloat(packed, false, bv);
                if (full) {
                    Unpack16NibblesToFloat(packed, true, bv + 4);
                }
                for (size_t q = 0; q < value_count / 4; ++q) {
                    dot[i] = vec_madd(av[q], bv[q], dot[i]);
                }
            }
        }

        for (size_t i = 0; i < NCols; ++i) {
            const float scale = QuantBScale[i * BlockCountK + k_blk];
            float zp = 8.0f;
            if constexpr (HasZeroPoint) {
                zp = static_cast<float>(GetZeroPoint(QuantBZeroPoint + i * StrideZeroPoint, k_blk));
            }
            total[i] = vec_madd(dot[i], vec_splats(scale), total[i]);
            total[i] = vec_madd(a_sum, vec_splats(-scale * zp), total[i]);
        }
    }

    if constexpr (NCols == 4) {
        Transpose4x4(total[0], total[1], total[2], total[3]);
        MLAS_VF sum = vec_add(vec_add(total[0], total[1]), vec_add(total[2], total[3]));
        if (Bias != nullptr) {
            sum = vec_add(sum, vec_xl(0, Bias));
        }
        vec_xst(sum, 0, C);
    } else {
        for (size_t i = 0; i < NCols; ++i) {
            C[i] = HorizontalSum(total[i]) + (Bias != nullptr ? Bias[i] : 0.0f);
        }
    }
}

template <bool HasZeroPoint>
void
SQ4BitGemmM1Kernel_CompFp32_Impl(
    size_t BlkLen,
    const float* A,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    const float* Bias
)
{
    constexpr size_t NCols = 4;
    const size_t StrideData = BlockCountK * MlasQNBitBlkDataSizeInBytes(kBlkBitWidth, BlkLen);
    const size_t StrideZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<kBlkBitWidth>(BlockCountK);

    size_t n = 0;
    for (; n + NCols <= CountN; n += NCols) {
        ComputeDotProducts_CompFp32<NCols, HasZeroPoint>(
            BlkLen, A, QuantBData + n * StrideData, QuantBScale + n * BlockCountK,
            HasZeroPoint ? QuantBZeroPoint + n * StrideZeroPoint : nullptr, C + n, CountK, BlockCountK,
            Bias != nullptr ? Bias + n : nullptr
        );
    }
    for (; n < CountN; ++n) {
        ComputeDotProducts_CompFp32<1, HasZeroPoint>(
            BlkLen, A, QuantBData + n * StrideData, QuantBScale + n * BlockCountK,
            HasZeroPoint ? QuantBZeroPoint + n * StrideZeroPoint : nullptr, C + n, CountK, BlockCountK,
            Bias != nullptr ? Bias + n : nullptr
        );
    }
}

void
SQ4BitGemmM1Kernel_CompFp32(
    size_t BlkLen,
    const float* A,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    const float* Bias
)
{
    if (QuantBZeroPoint != nullptr) {
        SQ4BitGemmM1Kernel_CompFp32_Impl<true>(
            BlkLen, A, QuantBData, QuantBScale, QuantBZeroPoint, C, CountN, CountK, BlockCountK, Bias
        );
    } else {
        SQ4BitGemmM1Kernel_CompFp32_Impl<false>(
            BlkLen, A, QuantBData, QuantBScale, QuantBZeroPoint, C, CountN, CountK, BlockCountK, Bias
        );
    }
}

// Dequantizes B into the 16-column panel layout consumed by MlasSgemmKernelPOWER10: element (k, n) of
// panel p = n / 16 is stored at FpData[(p * CountK + k) * 16 + n % 16]. Columns past CountN are zero.
template <bool HasZeroPoint>
void
Q4BitBlkDequantBForSgemm_CompFp32_Impl(
    size_t BlkLen,
    float* FpData,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK
)
{
    const size_t StrideData = BlockCountK * MlasQNBitBlkDataSizeInBytes(kBlkBitWidth, BlkLen);
    const size_t StrideZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<kBlkBitWidth>(BlockCountK);

    float* dst = FpData;

    for (size_t n = 0; n < CountN; n += 16) {
        const size_t cols = std::min(CountN - n, size_t{16});

        for (size_t k = 0, k_blk = 0; k < CountK; k += BlkLen, ++k_blk) {
            const size_t k_blk_len = std::min(CountK - k, BlkLen);

            MLAS_VF scale[16];
            MLAS_VF neg_scaled_zp[16];
            for (size_t c = 0; c < cols; ++c) {
                const float s = QuantBScale[(n + c) * BlockCountK + k_blk];
                float zp = 8.0f;
                if constexpr (HasZeroPoint) {
                    zp = static_cast<float>(GetZeroPoint(QuantBZeroPoint + (n + c) * StrideZeroPoint, k_blk));
                }
                scale[c] = vec_splats(s);
                neg_scaled_zp[c] = vec_splats(-s * zp);
            }

            for (size_t kk = 0; kk < k_blk_len; kk += 16) {
                const size_t rows = std::min(k_blk_len - kk, size_t{16});

                for (size_t cg = 0; cg < 16; cg += 4) {
                    // bv[j][q]: column cg + j, values kk + 4q .. kk + 4q + 3.
                    MLAS_VF bv[4][4];
                    for (size_t j = 0; j < 4; ++j) {
                        const size_t c = cg + j;
                        if (c >= cols) {
                            for (size_t q = 0; q < 4; ++q) {
                                bv[j][q] = vec_splats(0.0f);
                            }
                            continue;
                        }
                        const unsigned char* b = reinterpret_cast<const unsigned char*>(
                            QuantBData + (n + c) * StrideData +
                            (k_blk * BlkLen + kk) * kBlkBitWidth / 8
                        );
                        Unpack16NibblesToFloat(vec_xl_len(const_cast<unsigned char*>(b), 8), false, bv[j]);
                        for (size_t q = 0; q < 4; ++q) {
                            bv[j][q] = vec_madd(bv[j][q], scale[c], neg_scaled_zp[c]);
                        }
                    }

                    for (size_t q = 0; q < 4; ++q) {
                        Transpose4x4(bv[0][q], bv[1][q], bv[2][q], bv[3][q]);
                        for (size_t i = 0; i < 4; ++i) {
                            const size_t row = 4 * q + i;
                            if (row < rows) {
                                vec_xst(bv[i][q], 0, dst + row * 16 + cg);
                            }
                        }
                    }
                }

                dst += 16 * rows;
            }
        }
    }
}

void
SQ4BitBlkDequantBForSgemm_CompFp32(
    size_t BlkLen,
    float* FpData,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK
)
{
    if (QuantBZeroPoint != nullptr) {
        Q4BitBlkDequantBForSgemm_CompFp32_Impl<true>(
            BlkLen, FpData, QuantBData, QuantBScale, QuantBZeroPoint, CountN, CountK, BlockCountK
        );
    } else {
        Q4BitBlkDequantBForSgemm_CompFp32_Impl<false>(
            BlkLen, FpData, QuantBData, QuantBScale, QuantBZeroPoint, CountN, CountK, BlockCountK
        );
    }
}

}  // namespace

}  // namespace sqnbitgemm_power10

//
// Kernel dispatch structure definition.
//
const MLAS_QNBIT_GEMM_DISPATCH MlasSQNBitGemmDispatchPOWER10 = []() {
    using namespace sqnbitgemm_power10;

    MLAS_QNBIT_GEMM_DISPATCH d;

    d.Q4BitGemmPackQuantBDataSize = Q4BitGemmPackQuantBDataSize;
    d.SQ4BitGemmPackQuantBData = SQ4BitGemmPackQuantBData;
    d.SQ4BitGemmPackQuantBDataAndBlkSum = SQ4BitGemmPackQuantBDataAndBlkSum;

    d.QNBitGemmPerGemmWorkspaceSize = QNBitGemmPerGemmWorkspaceSize;
    d.QNBitGemmPerGemmWorkspaceAlignment = QNBitGemmPerGemmWorkspaceAlignment;

    d.SQ4BitGemmM1Kernel_CompFp32 = SQ4BitGemmM1Kernel_CompFp32;
    d.SQ4BitBlkDequantBForSgemm_CompFp32 = SQ4BitBlkDequantBForSgemm_CompFp32;

    // SQ4BitGemmKernel_CompInt8 / QuantizeARow_CompInt8 stay unset so the shared code takes the
    // separate-scale BlkSum path (PerGemmQuantAWorkspace) instead of the interleaved Q8 block path.
    d.SQ4BitGemmKernel_BlkSum_CompInt8 = SQ4BitGemmKernel_BlkSum_CompInt8;
    d.QuantizeARowComputeBlkSum_CompInt8 = QuantizeARowComputeBlkSum_CompInt8;

    return d;
}();
