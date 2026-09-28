#pragma once

// NVFP4 codes multiplied by their raw E4M3 G16 scales are exactly representable
// in BF16. The global weight divisor is applied to the complete FP32 reduction.
// Activations remain the represented public BF16 inputs.

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/nvfp4/nvfp4_operands.h"
#include "ops/linear/nvfp4/nvfp4_shared.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// One CTA reduces kBlockRows = 16 * kRowTiles weight rows over K. Warp w owns the 64 columns
// [g * kBlockK + 64 * w, +64) of every K group g and accumulates them in group order, one m16n8k16
// MMA per 16 columns; the kKWarps partial sums are then added in pairs (w, w + 1), and warp 0 adds
// the pairs in ascending order. An output element follows this sequence for every kRowTiles,
// kStageTokens and kStages, so these select speed and not results.
template <class Schedule, class Output, class Epilogue, class RowPolicy>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void nvfp4_a16_sliced_k_mma_kernel(
    Nvfp4A16Operands operands, Output output, Epilogue epilogue, RowPolicy row_policy,
    int token_offset) {
    const auto* __restrict__ x             = operands.x;
    const auto* __restrict__ weight_codes  = operands.codes;
    const auto* __restrict__ weight_scales = operands.scales;
    const int kHidden                      = Schedule::kStaticK ? Schedule::kStaticK : operands.k;
    constexpr int ActiveTokens =
        Schedule::kTokenCapacity ? Schedule::kTokenCapacity : Schedule::kBlockTokens;
    constexpr bool MaskedColumns = !Schedule::kExactTokens;
    constexpr int kTileK         = Schedule::kTileKPerWarp;
    constexpr int kWarps         = Schedule::kKWarps;
    constexpr int kRowTiles      = Schedule::kRowTiles;
    constexpr int kBlockRows     = Schedule::kBlockRows;
    constexpr int kBlockK        = Schedule::kBlockK;
    const int kGroups            = kHidden / kBlockK;
    constexpr int kBlockTokens   = Schedule::kBlockTokens;
    constexpr int kTokenMmas     = kBlockTokens / 8;
    // Output rows of one 16-row MMA tile: a paired tile holds 8 gate rows over their 8 up rows.
    constexpr int kTileOutputRows = RowPolicy::kPaired ? 8 : 16;
    constexpr bool kPadded        = Schedule::kActivationStage == Nvfp4ActivationStage::PaddedZero;
    constexpr int kSharedTokens   = Schedule::kStageTokens;
    constexpr int kStagedTokens   = kPadded ? kSharedTokens : ActiveTokens;
    static_assert(ActiveTokens >= 1 && ActiveTokens <= kSharedTokens);
    static_assert((kWarps & 1) == 0);

    union SharedStorage {
        struct {
            std::uint8_t codes[Schedule::kStages][kBlockRows][kBlockK / 2];
            std::uint8_t scales[Schedule::kStages][kBlockRows][kBlockK / 16];
            __nv_bfloat16 activations[Schedule::kStages][kWarps][kSharedTokens * kTileK];
        } staging;

        float partial[kWarps * kRowTiles * kTokenMmas * 32 * 4];
    };

    static_assert(sizeof(SharedStorage) == Schedule::kSharedBytes);
    auto& shared =
        *reinterpret_cast<SharedStorage*>(nvfp4_shared_storage<Schedule::kSharedBytes>());
    auto& code_shared  = shared.staging.codes;
    auto& scale_shared = shared.staging.scales;
    auto& x_shared     = shared.staging.activations;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
    // First output row of MMA tile `tile` of this CTA.
    const auto tile_row0 = [](int tile) {
        return (static_cast<int>(blockIdx.x) * kRowTiles + tile) * kTileOutputRows;
    };
    const int token_begin = token_offset + static_cast<int>(blockIdx.y) * ActiveTokens;
    const int live_columns =
        MaskedColumns ? min(ActiveTokens, operands.tokens - token_begin) : ActiveTokens;

    const auto stage_activation = [&](int stage, int group_k0) {
        constexpr auto kActivationCache = Schedule::kActivationCache;
        constexpr int kItems            = kStagedTokens * (kTileK / 8);
        for (int item = lane; item < kItems; item += 32) {
            const int token = item / (kTileK / 8);
            const int k8    = item - token * (kTileK / 8);
            auto* destination =
                &x_shared[stage][warp][token * kTileK + nvfp4_a16_shared_col_64(token, k8 * 8)];
            if constexpr (!MaskedColumns && kStagedTokens == ActiveTokens) {
                cp_async<16, kActivationCache>(
                    destination, x + static_cast<std::int64_t>(token_begin + token) * kHidden +
                                     group_k0 + warp * kTileK + k8 * 8);
            } else {
                const int source_token = token < live_columns ? token : 0;
                cp_async_zfill<16, kActivationCache>(
                    destination,
                    x + static_cast<std::int64_t>(token_begin + source_token) * kHidden + group_k0 +
                        warp * kTileK + k8 * 8,
                    token < live_columns ? 16 : 0);
            }
        }
    };

    constexpr int kCodeChunks  = kBlockK / 32;
    constexpr int kCodeSwizzle = kCodeChunks < 8 ? kCodeChunks - 1 : 7;
    const auto stage_codes     = [&](int stage, int group_k0) {
#pragma unroll
        for (int ri = 0; ri < Schedule::kRowsPerLoaderWarp; ++ri) {
            const int row    = warp * Schedule::kRowsPerLoaderWarp + ri;
            const int parent = row_policy.weight_row(tile_row0(row / 16), row % 16, operands.rows);
            for (int chunk = lane; chunk < kCodeChunks; chunk += 32) {
                cp_async<16, Schedule::kWeightCache>(
                    &code_shared[stage][row][(chunk ^ (row & kCodeSwizzle)) * 16],
                    weight_codes + static_cast<std::int64_t>(parent) * (kHidden / 2) +
                        group_k0 / 2 + chunk * 16);
            }
            for (int tile = lane; tile < kBlockK / 64; tile += 32) {
                cp_async<4>(&scale_shared[stage][row][tile * 4],
                            weight_scales +
                                nvfp4_scale_byte_offset(parent, group_k0 / 16 + tile * 4, kHidden));
            }
        }
    };

    const int b_row                              = lane & 7;
    const int b_k_offset                         = ((lane >> 3) & 1) << 3;
    const int warp_k0                            = warp * kTileK;
    float accumulators[kRowTiles][kTokenMmas][4] = {};

#pragma unroll
    for (int stage = 0; stage < Schedule::kStages; ++stage) {
        if (stage < kGroups) {
            stage_codes(stage, stage * kBlockK);
            stage_activation(stage, stage * kBlockK);
            cp_commit();
        }
    }
#pragma unroll
    for (int group_index = 0; group_index < kGroups; ++group_index) {
        const int stage = group_index % Schedule::kStages;
        if (group_index + Schedule::kStages - 1 < kGroups)
            cp_wait<Schedule::kStages - 1>();
        else
            cp_wait<0>();
        __syncthreads();
#pragma unroll
        for (int k_step = 0; k_step < kTileK / 16; ++k_step) {
            const int code_col   = k_step * 16 + lid * 2;
            const auto load_pair = [&](int row, int col) {
                const int byte   = (warp_k0 + col) / 2;
                const int offset = ((byte / 16) ^ (row & kCodeSwizzle)) * 16 + (byte & 15);
                return nvfp4_scaled_pair_bf16(code_shared[stage][row][offset],
                                              scale_shared[stage][row][(warp_k0 + col) / 16]);
            };
            unsigned a[kRowTiles][4];
#pragma unroll
            for (int tile = 0; tile < kRowTiles; ++tile) {
                a[tile][0] = load_pair(tile * 16 + gid, code_col);
                a[tile][1] = load_pair(tile * 16 + gid + 8, code_col);
                a[tile][2] = load_pair(tile * 16 + gid, code_col + 8);
                a[tile][3] = load_pair(tile * 16 + gid + 8, code_col + 8);
            }
#pragma unroll
            for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                unsigned b0;
                unsigned b1;
                const int token = token_mma * 8 + b_row;
                const int row   = token < kSharedTokens ? token : token % kSharedTokens;
                ldmatrix_x2(
                    b0, b1,
                    smem_addr(
                        &x_shared[stage][warp][row * kTileK + nvfp4_a16_shared_col_64(
                                                                  row, k_step * 16 + b_k_offset)]));
#pragma unroll
                for (int tile = 0; tile < kRowTiles; ++tile) {
                    auto& accumulator = accumulators[tile][token_mma];
                    mma_bf16(accumulator[0], accumulator[1], accumulator[2], accumulator[3],
                             a[tile][0], a[tile][1], a[tile][2], a[tile][3], b0, b1);
                }
            }
        }

        __syncthreads();
        const int next = group_index + Schedule::kStages;
        if (next < kGroups) {
            stage_codes(stage, next * kBlockK);
            stage_activation(stage, next * kBlockK);
            cp_commit();
        }
    }

    __syncthreads();
    auto* partial = shared.partial;
    const auto partial_at = [&](int split, int tile, int token_mma) {
        return partial + (((split * kRowTiles + tile) * kTokenMmas + token_mma) * 32 + lane) * 4;
    };
    if ((warp & 1) != 0) {
#pragma unroll
        for (int tile = 0; tile < kRowTiles; ++tile) {
#pragma unroll
            for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                const auto& accumulator = accumulators[tile][token_mma];
                store_vec(partial_at(warp, tile, token_mma),
                          make_float4(accumulator[0], accumulator[1], accumulator[2],
                                      accumulator[3]));
            }
        }
    }
    __syncthreads();

    if ((warp & 1) == 0) {
#pragma unroll
        for (int tile = 0; tile < kRowTiles; ++tile) {
#pragma unroll
            for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                auto& accumulator    = accumulators[tile][token_mma];
                const float4 partner = load_vec<float4>(partial_at(warp + 1, tile, token_mma));
                accumulator[0] += partner.x;
                accumulator[1] += partner.y;
                accumulator[2] += partner.z;
                accumulator[3] += partner.w;
                if (warp != 0) {
                    store_vec(partial_at(warp, tile, token_mma),
                              make_float4(accumulator[0], accumulator[1], accumulator[2],
                                          accumulator[3]));
                }
            }
        }
    }
    __syncthreads();

    if (warp == 0) {
        const float top_scale = operands.alpha, bottom_scale = operands.alpha;

#pragma unroll
        for (int tile = 0; tile < kRowTiles; ++tile) {
            const int row0         = tile_row0(tile);
            const auto destination = linear_output_tile<kTileOutputRows>(output, row0);
#pragma unroll
            for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                const auto& accumulator = accumulators[tile][token_mma];
                float4 sum =
                    make_float4(accumulator[0], accumulator[1], accumulator[2], accumulator[3]);
#pragma unroll
                for (int split = 2; split < kWarps; split += 2) {
                    const float4 value = load_vec<float4>(partial_at(split, tile, token_mma));
                    sum.x += value.x;
                    sum.y += value.y;
                    sum.z += value.z;
                    sum.w += value.w;
                }
                const int local_token = token_mma * 8 + 2 * lid;
                const int token0      = token_begin + local_token;
                const int row_a       = row_policy.weight_row(row0, gid, operands.rows);
                const int row_b       = row_policy.weight_row(row0, gid + 8, operands.rows);
                if constexpr (RowPolicy::kPaired) {
                    if (local_token < live_columns)
                        epilogue.apply_pair(destination, row_a, token0, sum.x * top_scale,
                                            sum.z * bottom_scale);
                    if (local_token + 1 < live_columns)
                        epilogue.apply_pair(destination, row_a, token0 + 1, sum.y * top_scale,
                                            sum.w * bottom_scale);
                } else {
                    if (local_token < live_columns) {
                        destination.store(row_a, token0,
                                          epilogue.apply(row_a, token0, sum.x * top_scale));
                        destination.store(row_b, token0,
                                          epilogue.apply(row_b, token0, sum.z * bottom_scale));
                    }
                    if (local_token + 1 < live_columns) {
                        destination.store(row_a, token0 + 1,
                                          epilogue.apply(row_a, token0 + 1, sum.y * top_scale));
                        destination.store(row_b, token0 + 1,
                                          epilogue.apply(row_b, token0 + 1, sum.w * bottom_scale));
                    }
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
