#include "core/weight.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_output.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Column tiles amortize gate/up decode over the complete speculative block.
using M64N128  = Nvfp4W4a4MmaSchedule<64, 128, 256, 4, 4, 2, 1>;
using M128N128 = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 4, 2, 1>;
using M96N128  = Nvfp4W4a4MmaSchedule<96, 128, 256, 3, 4, 2, 1>;

// Geometry is the gate/up problem: gate rows [0,M) precede their up rows [M,2M), M = N/2.
template <class Geometry>
struct Nvfp4SwiGluRows {
    static constexpr int kIntermediate  = Geometry::kOutputRows / 2;
    static constexpr bool kContiguous   = false;
    static constexpr int kRowsPerBranch = M64N128::kBlockN / 2;

    __device__ __forceinline__ int weight_row(int row_begin, int local_row) const {
        return row_begin + (local_row & (kRowsPerBranch - 1)) +
               (local_row >= kRowsPerBranch ? kIntermediate : 0);
    }
};

union Nvfp4SwiGluBf16Pair {
    unsigned bits;
    __nv_bfloat162 values;
};

template <class Geometry>
struct Nvfp4SwiGluOutput {
    static constexpr int kIntermediate = Geometry::kOutputRows / 2;

    __nv_bfloat16* data;

    __device__ __forceinline__ unsigned combine(unsigned gate_bits, unsigned up_bits) const {
        Nvfp4SwiGluBf16Pair gate{gate_bits};
        Nvfp4SwiGluBf16Pair up{up_bits};
        const float2 gate_values = __bfloat1622float2(gate.values);
        const float2 up_values   = __bfloat1622float2(up.values);
        Nvfp4SwiGluBf16Pair result;
        result.values = __floats2bfloat162_rn(silu(gate_values.x) * up_values.x,
                                              silu(gate_values.y) * up_values.y);
        return result.bits;
    }

    __device__ __forceinline__ void store_pair_vector(std::int32_t row, std::int32_t token,
                                                      uint4 gate, uint4 up) const {
        const uint4 values = make_uint4(combine(gate.x, up.x), combine(gate.y, up.y),
                                        combine(gate.z, up.z), combine(gate.w, up.w));
        store_vec(data + static_cast<std::int64_t>(token) * kIntermediate + row, values);
    }
};

template <class Geometry, class Schedule>
void launch_gemm(const Weight& weight, Tensor& out, Nvfp4W4a4Workspace workspace,
                 std::int32_t tokens, cudaStream_t stream) {
    using Rows              = Nvfp4SwiGluRows<Geometry>;
    using Output            = Nvfp4SwiGluOutput<Geometry>;
    constexpr int kPairRows = Schedule::kBlockN / 2;
    static_assert(kPairRows == Rows::kRowsPerBranch);
    static_assert((Rows::kIntermediate % kPairRows) == 0);
    const dim3 grid(Rows::kIntermediate / kPairRows,
                    (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const Nvfp4W4a4MaterializedActivation activation{workspace.codes, workspace.scales};
    const Rows row_policy{};
    const Output output{static_cast<__nv_bfloat16*>(out.data)};
    const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
    nvfp4_w4a4_mma_kernel<Geometry, Schedule, Nvfp4IdentityEpilogue, Output, Rows, true>
        <<<grid, Schedule::kThreads, 0, stream>>>(
            activation, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), tokens, alpha, Nvfp4IdentityEpilogue{},
            output, row_policy);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, class Schedule>
void launch(const Tensor& x, const Weight& weight, Tensor& out, WorkspaceArena& workspace,
            cudaStream_t stream) {
    auto scope = workspace.scope();
    const Nvfp4W4a4Workspace scratch =
        allocate_nvfp4_w4a4_workspace(workspace, x.ne[1], Geometry::kInputRows);
    launch_nvfp4_w4a4_quantize(x, weight, scratch, Nvfp4ScaleLayout::RowMajor, stream);
    launch_gemm<Geometry, Schedule>(weight, out, scratch, x.ne[1], stream);
}

template <class Geometry>
void launch_problem(const Tensor& x, const Weight& weight, Tensor& out, WorkspaceArena& workspace,
                    cudaStream_t stream) {
    if (x.ne[1] <= M64N128::kBlockM) {
        launch<Geometry, M64N128>(x, weight, out, workspace, stream);
    } else if (x.ne[1] <= M96N128::kBlockM) {
        launch<Geometry, M96N128>(x, weight, out, workspace, stream);
    } else {
        launch<Geometry, M128N128>(x, weight, out, workspace, stream);
    }
}

} // namespace

void nvfp4_linear_swiglu_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                     WorkspaceArena& workspace, cudaStream_t stream) {
    switch (resolve_nvfp4_geometry(weight.n, weight.k)) {
    case Nvfp4GeometryId::N34816K5120:
        launch_problem<Nvfp4N34816K5120>(x, weight, out, workspace, stream);
        return;
    case Nvfp4GeometryId::N17408K5120:
        launch_problem<Nvfp4N17408K5120>(x, weight, out, workspace, stream);
        return;
    case Nvfp4GeometryId::N14336K5120:
    case Nvfp4GeometryId::N16384K5120:
    case Nvfp4GeometryId::N5120K6144:
    case Nvfp4GeometryId::N5120K17408:
    case Nvfp4GeometryId::N5120K8704:
        break;
    }
    throw std::invalid_argument("nvfp4 linear_swiglu: unsupported problem");
}

} // namespace ninfer::ops::detail
