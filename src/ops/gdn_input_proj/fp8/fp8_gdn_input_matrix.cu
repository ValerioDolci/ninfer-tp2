#include "core/weight.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_output.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_simt.cuh"
#include "ops/linear/fp8/fp8_a16_ksplit_mma.cuh"
#include "ops/linear/fp8/fp8_a16_gemm_mma.cuh"

namespace ninfer::ops::detail {
namespace {

template <class Geometry, class Output, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                  cudaStream_t stream) {
    using Schedule =
        Fp8SimtSchedule<8, 2, (ActiveTokens >= 5 && ActiveTokens <= 6) ? 8 : 16, ActiveTokens, 1,
                        ActiveTokens <= 4 ? Fp8SimtActivationAccess::SharedPhase
                                          : Fp8SimtActivationAccess::TokenPacked,
                        Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
    constexpr int kTokenTiles = (ActiveTokens + Schedule::kTokenTile - 1) / Schedule::kTokenTile;
    constexpr int kBlocks     = (Geometry::kOutputRows / Schedule::kRowsPerCta) * kTokenTiles;
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
    fp8_simt_kernel<Geometry, ActiveTokens, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const __nv_bfloat16*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, class Output, int Capacity>
void launch_small_mma(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                      cudaStream_t stream) {
    constexpr int warps = Capacity <= 8 ? 16 : Capacity <= 24 ? 8 : 4;
    using Schedule      = Fp8A16KSplitSchedule<warps, Capacity, warps == 16 ? 1 : 2>;
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
    fp8_a16_ksplit_mma_kernel<Geometry, Capacity, Schedule, Output, true>
        <<<Geometry::kOutputRows / Schedule::kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, class Output, class Schedule>
void launch_gemm(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                 cudaStream_t stream) {
    static_assert(Output::kQkvRows % Schedule::kBlockRows == 0);
    static_assert(Output::kZRows % Schedule::kBlockRows == 0);
    static_assert(Schedule::kSharedBytes <= 48 * 1024);
    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockRows,
                    (x.ne[1] + Schedule::kBlockTokens - 1) / Schedule::kBlockTokens);
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
    fp8_a16_gemm_mma_kernel<Geometry, Schedule, false>
        <<<grid, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, class Output>
void launch_matrix(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                   cudaStream_t stream) {
    static_assert(Geometry::kOutputRows == Output::kRows);
    // SIMT for the latency regime, bounded MMA column capacities, then amortized weight decode.
    const int columns = x.ne[1];
    if (columns == 2) return launch_exact<Geometry, Output, 2>(x, weight, qkv, z, stream);
    if (columns == 3) return launch_exact<Geometry, Output, 3>(x, weight, qkv, z, stream);
    if (columns == 4) return launch_exact<Geometry, Output, 4>(x, weight, qkv, z, stream);
    if (columns <= 8) return launch_small_mma<Geometry, Output, 8>(x, weight, qkv, z, stream);
    if (columns <= 16) return launch_small_mma<Geometry, Output, 16>(x, weight, qkv, z, stream);
    if (columns <= 24) return launch_small_mma<Geometry, Output, 24>(x, weight, qkv, z, stream);
    if (columns <= 32) return launch_small_mma<Geometry, Output, 32>(x, weight, qkv, z, stream);
    if (columns <= 64)
        return launch_gemm<Geometry, Output, Fp8A16GemmSchedule<32, 64, 128, 16, 16, 1, 3>>(
            x, weight, qkv, z, stream);
    if (columns <= 96)
        return launch_gemm<Geometry, Output, Fp8A16GemmSchedule<64, 96, 128, 64, 16, 1, 2>>(
            x, weight, qkv, z, stream);
    return launch_gemm<Geometry, Output, Fp8A16GemmSchedule<64, 128, 64, 32, 16, 2, 2>>(
        x, weight, qkv, z, stream);
}

} // namespace

void fp8_gdn_input_matrix_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 cudaStream_t stream) {
    launch_matrix<Fp8N16384K5120, Fp8GdnInputOutput>(x, weight, qkv, z, stream);
}

void fp8_gdn_input_shard_matrix_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                       Tensor& z, cudaStream_t stream) {
    launch_matrix<Fp8N8192K5120, Fp8GdnInputShardOutput>(x, weight, qkv, z, stream);
}

} // namespace ninfer::ops::detail
