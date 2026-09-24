#include "core/weight.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"

namespace ninfer::ops::detail {
namespace {

using M32N64            = Nvfp4W4a4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using M32N128           = Nvfp4W4a4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using M64N128           = Nvfp4W4a4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using M128N128Pipelined = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using M128N128Resident  = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;

// This projection selects its own route, so the layout the quantizer writes below must be derived
// from the same predicate; the two are read together at the call site for that reason.
constexpr bool w4a4_tma_route(std::int32_t tokens) { return tokens >= 1024; }

template <class Problem, class Schedule>
void launch_gemm(const Weight& weight, Tensor& qkv, Tensor& z, Nvfp4W4a4Workspace workspace,
                 std::int32_t tokens, cudaStream_t stream) {
    using Geometry = typename Problem::Geometry;
    using Output   = typename Problem::Output;
    static_assert((Output::kQkvRows % Schedule::kBlockN) == 0);
    static_assert((Output::kZRows % Schedule::kBlockN) == 0);
    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN,
                    (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const Nvfp4W4a4MaterializedActivation activation{workspace.codes, workspace.scales};
    const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
    nvfp4_w4a4_mma_kernel<Geometry, Schedule><<<grid, Schedule::kThreads, 0, stream>>>(
        activation, static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), tokens, alpha, Nvfp4IdentityEpilogue{},
        Output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)});
    CUDA_CHECK(cudaGetLastError());
}

template <class Problem>
void launch_mma(const Weight& weight, Tensor& qkv, Tensor& z, Nvfp4W4a4Workspace workspace,
                std::int32_t tokens, cudaStream_t stream) {
    if (tokens <= 64) {
        launch_gemm<Problem, M32N64>(weight, qkv, z, workspace, tokens, stream);
    } else if (tokens <= 96) {
        launch_gemm<Problem, M32N128>(weight, qkv, z, workspace, tokens, stream);
    } else if (tokens <= 128) {
        launch_gemm<Problem, M128N128Pipelined>(weight, qkv, z, workspace, tokens, stream);
    } else if (tokens <= 192) {
        launch_gemm<Problem, M64N128>(weight, qkv, z, workspace, tokens, stream);
    } else {
        launch_gemm<Problem, M128N128Resident>(weight, qkv, z, workspace, tokens, stream);
    }
}

} // namespace

void nvfp4_gdn_input_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    launch_nvfp4_w4a4_quantize(
        x, weight, workspace,
        w4a4_tma_route(tokens) ? Nvfp4ScaleLayout::Tiled : Nvfp4ScaleLayout::RowMajor, stream);
    if (w4a4_tma_route(tokens)) {
        const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
        launch_nvfp4_w4a4_tma_gdn(weight.n, workspace.codes, workspace.scales,
                                  static_cast<const std::uint8_t*>(weight.qdata),
                                  static_cast<const std::uint8_t*>(weight.scales),
                                  static_cast<__nv_bfloat16*>(qkv.data),
                                  static_cast<__nv_bfloat16*>(z.data), tokens, alpha, stream);
        return;
    }
    visit_nvfp4_gdn_input_problem(weight.n, [&]<class Problem>() {
        launch_mma<Problem>(weight, qkv, z, workspace, tokens, stream);
    });
}

} // namespace ninfer::ops::detail
