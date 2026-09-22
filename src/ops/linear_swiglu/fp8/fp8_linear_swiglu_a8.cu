#include "core/weight.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/launcher/kernel_attr_once.h"
#include "ops/linear/fp8/fp8_a8_mma.cuh"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_a8_schedule.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using Schedule = Fp8A8DefaultSchedule;
static_assert((Schedule::kBlockRows % 2) == 0);

// Geometry is the gate/up problem: gate rows [0,M) precede their up rows [M,2M), M = N/2.
template <class Geometry, bool FullTokens>
void launch_mma(const Weight& weight, Tensor& out, Fp8A8Workspace workspace, std::int32_t tokens,
                cudaStream_t stream) {
    constexpr int kIntermediate = Geometry::kOutputRows / 2;
    using Rows                  = Fp8SwiGluRows<Schedule::kBlockRows / 2, kIntermediate>;
    constexpr int kRowTiles     = Geometry::kOutputRows / Schedule::kBlockRows;
    const int token_tiles       = (tokens + Schedule::kBlockTokens - 1) / Schedule::kBlockTokens;
    const int blocks            = kRowTiles * token_tiles;
    const Rows rows{};
    const Fp8SwiGluOutput output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};

    if constexpr (Schedule::kSharedBytes > 48 * 1024) {
        static FuncAttrPerDevice attribute;
        attribute.ensure(fp8_mma_kernel<Geometry, Schedule, FullTokens, Fp8IdentityEpilogue,
                                        Fp8SwiGluOutput, Rows, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, Schedule::kSharedBytes);
    }
    fp8_mma_kernel<Geometry, Schedule, FullTokens, Fp8IdentityEpilogue, Fp8SwiGluOutput, Rows, true>
        <<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), tokens, Fp8IdentityEpilogue{}, output,
            rows);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void launch_problem(const Weight& weight, Tensor& out, Fp8A8Workspace workspace,
                    std::int32_t tokens, cudaStream_t stream) {
    if ((tokens % Schedule::kBlockTokens) == 0) {
        launch_mma<Geometry, true>(weight, out, workspace, tokens, stream);
    } else {
        launch_mma<Geometry, false>(weight, out, workspace, tokens, stream);
    }
}

} // namespace

void fp8_linear_swiglu_a8_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                 WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope                   = workspace.scope();
    const Fp8A8Workspace scratch = allocate_fp8_a8_workspace(workspace, x.ne[1], weight.k);
    launch_fp8_a8_quantize(x, weight, scratch, stream);
    switch (resolve_fp8_geometry(weight.n, weight.k)) {
    case Fp8GeometryId::N34816K5120:
        launch_problem<Fp8N34816K5120>(weight, out, scratch, x.ne[1], stream);
        return;
    case Fp8GeometryId::N17408K5120:
        launch_problem<Fp8N17408K5120>(weight, out, scratch, x.ne[1], stream);
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
