#include "core/weight.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_simt.cuh"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_output.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

// Geometry is the gate/up problem: gate rows [0,M) precede their up rows [M,2M), M = N/2.
template <class Geometry, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int kIntermediate = Geometry::kOutputRows / 2;
    using Schedule =
        Fp8SimtSchedule<8, 2, ActiveTokens == 4 ? 8 : 16, ActiveTokens, 1,
                        ActiveTokens <= 3 ? Fp8SimtActivationAccess::SharedPhase
                                          : Fp8SimtActivationAccess::TokenPacked,
                        Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
    static_assert((Schedule::kRowsPerWarp % 2) == 0);
    using Rows                = Fp8SwiGluRows<Schedule::kRowsPerWarp / 2, kIntermediate>;
    constexpr int kTokenTiles = (ActiveTokens + Schedule::kTokenTile - 1) / Schedule::kTokenTile;
    constexpr int kBlocks     = (Geometry::kOutputRows / Schedule::kRowsPerCta) * kTokenTiles;
    fp8_simt_kernel<Geometry, ActiveTokens, Schedule, Fp8SwiGluOutput, Fp8IdentityEpilogue, Rows,
                    true><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const __nv_bfloat16*>(weight.scales),
        Fp8SwiGluOutput{static_cast<__nv_bfloat16*>(out.data), kIntermediate}, {}, Rows{});
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, 2 + static_cast<int>(Offsets)>...};
}

template <class Geometry>
constexpr auto make_launchers() {
    return make_launchers<Geometry>(std::make_index_sequence<4 - 2 + 1>{});
}

constexpr auto kGateUp34816Launchers = make_launchers<Fp8N34816K5120>();
constexpr auto kGateUp17408Launchers = make_launchers<Fp8N17408K5120>();

} // namespace

void fp8_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                      cudaStream_t stream) {
    if (x.ne[1] < 2 || x.ne[1] > 4) {
        throw std::invalid_argument("fp8 linear_swiglu small-T: unsupported T");
    }
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - 2);
    switch (resolve_fp8_geometry(weight.n, weight.k)) {
    case Fp8GeometryId::N34816K5120:
        kGateUp34816Launchers[index](x, weight, out, stream);
        return;
    case Fp8GeometryId::N17408K5120:
        kGateUp17408Launchers[index](x, weight, out, stream);
        return;
    case Fp8GeometryId::N14336K5120:
    case Fp8GeometryId::N16384K5120:
    case Fp8GeometryId::N248320K5120:
    case Fp8GeometryId::N5120K6144:
    case Fp8GeometryId::N5120K17408:
    case Fp8GeometryId::N7168K5120:
    case Fp8GeometryId::N8192K5120:
    case Fp8GeometryId::N124160K5120:
    case Fp8GeometryId::N5120K3072:
    case Fp8GeometryId::N5120K8704:
        break;
    }
    throw std::invalid_argument("fp8 linear_swiglu: unsupported problem");
}

} // namespace ninfer::ops::detail
