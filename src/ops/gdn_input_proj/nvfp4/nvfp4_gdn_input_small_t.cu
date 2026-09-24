#include "core/weight.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_simt.cuh"

#include <array>
#include <cstddef>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, cudaStream_t);

template <class Problem, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                  cudaStream_t stream) {
    using Geometry = typename Problem::Geometry;
    using Output   = typename Problem::Output;
    using Schedule = Nvfp4SimtSchedule<4, 1, 2, (ActiveTokens >= 17 && ActiveTokens <= 20) ? 8 : 16,
                                       ActiveTokens, 1,
                                       ActiveTokens == 2 ? Nvfp4SimtActivationAccess::SharedPhase
                                                         : Nvfp4SimtActivationAccess::TokenPacked,
                                       Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                       Nvfp4SimtBlockOrder::RowsContiguous, 1>;
    constexpr int kTokenTiles = (ActiveTokens + Schedule::kTokenTile - 1) / Schedule::kTokenTile;
    constexpr int kBlocks     = (Geometry::kOutputRows / Schedule::kRowsPerCta) * kTokenTiles;
    const float inverse       = 1.0F / weight.weight_scale_divisor;
    nvfp4_simt_kernel<Geometry, ActiveTokens, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), inverse, Nvfp4IdentityEpilogue{},
        Output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)});
    CUDA_CHECK(cudaGetLastError());
}

template <class Problem, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<Problem, 2 + static_cast<int>(Offsets)>...};
}

template <class Problem>
constexpr auto kLaunchers = make_launchers<Problem>(std::make_index_sequence<32 - 2 + 1>{});

} // namespace

void nvfp4_gdn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    cudaStream_t stream) {
    visit_nvfp4_gdn_input_problem(weight.n, [&]<class Problem>() {
        kLaunchers<Problem>[x.ne[1] - 2](x, weight, qkv, z, stream);
    });
}

} // namespace ninfer::ops::detail
