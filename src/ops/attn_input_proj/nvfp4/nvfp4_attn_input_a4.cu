#include "core/weight.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "core/device.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_a4_tma_launch.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using M32N64            = Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using M32N128           = Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using M64N128           = Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using M128N128Pipelined = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using M128N128Resident  = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;

// This projection selects its own route, so the layout the quantizer writes below must be derived
// from the same predicate; the two are read together at the call site for that reason.
constexpr bool uses_tma(std::int32_t tokens) { return tokens >= 512; }

// First width at which the TMA route reads the 256-token scale tiles instead of the 128-token ones.
// The [7168,5120] two-device shard keeps the 128-token schedule through T=1024: its 56 row tiles
// give 224 CTAs at 256 tokens per tile, 3.2 waves on a 70-SM RTX 5070 Ti, and 448 at 128. Measured
// there at the 1024-token prefill chunk: 196.5 -> 183.7 us; from T=1025 the 256-token tile is
// faster again. The [14336,5120] parent keeps 1024 (upstream).
std::int32_t first_tiled256_tokens(std::int32_t rows) {
    return rows == Nvfp4N7168K5120::kOutputRows ? 1025 : 1024;
}

template <class Problem, class Schedule>
void launch_gemm(const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                 Nvfp4A4Workspace workspace, std::int32_t tokens, cudaStream_t stream) {
    using Geometry = typename Problem::Geometry;
    using Output   = typename Problem::Output;
    // Row tiles stay within one section at the parent and at the shard.
    static_assert((Problem::kQueryRows % Schedule::kBlockRows) == 0);
    static_assert((Problem::kKeyRows % Schedule::kBlockRows) == 0);
    launch_nvfp4_a4_mma<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a4_operands(weight, workspace, tokens, Nvfp4ScaleLayout::RowMajor),
        Output{static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
               static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)},
        LinearIdentityEpilogue{}, stream);
}

// The parent and its two-device shard share K and therefore every schedule and token cutoff.
template <class Problem>
void launch_mma(const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                Nvfp4A4Workspace workspace, std::int32_t tokens, cudaStream_t stream) {
    if (tokens <= 64) {
        launch_gemm<Problem, M32N64>(weight, q, gate, k, v, workspace, tokens, stream);
    } else if (tokens <= 96) {
        launch_gemm<Problem, M32N128>(weight, q, gate, k, v, workspace, tokens, stream);
    } else if (tokens <= 128) {
        launch_gemm<Problem, M128N128Pipelined>(weight, q, gate, k, v, workspace, tokens, stream);
    } else if (tokens <= 192) {
        launch_gemm<Problem, M64N128>(weight, q, gate, k, v, workspace, tokens, stream);
    } else if (tokens <= 384) {
        launch_gemm<Problem, M128N128Resident>(weight, q, gate, k, v, workspace, tokens, stream);
    } else if (tokens <= 512) {
        launch_gemm<Problem, M128N128Pipelined>(weight, q, gate, k, v, workspace, tokens, stream);
    } else {
        launch_gemm<Problem, M128N128Resident>(weight, q, gate, k, v, workspace, tokens, stream);
    }
}

} // namespace

void nvfp4_attn_input_a4_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                Tensor& k, Tensor& v, Nvfp4A4Workspace workspace,
                                cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    const auto layout =
        uses_tma(tokens) ? (tokens < first_tiled256_tokens(weight.n) ? Nvfp4ScaleLayout::Tiled128
                                                                      : Nvfp4ScaleLayout::Tiled256)
                         : Nvfp4ScaleLayout::RowMajor;
    // Resolve the parent or shard before quantizing, so an unsupported weight enqueues nothing.
    visit_nvfp4_attn_input_problem(weight.n, [&]<class Problem>() {
        launch_nvfp4_a4_quantize(x, weight, workspace, layout, stream);
        if (uses_tma(tokens)) {
            // Selects the same problem from the operands' rows (`weight.n`).
            launch_nvfp4_a4_tma_attention(
                nvfp4_a4_operands(weight, workspace, tokens, layout),
                static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(gate.data),
                static_cast<__nv_bfloat16*>(k.data), static_cast<__nv_bfloat16*>(v.data), stream);
            return;
        }
        launch_mma<Problem>(weight, q, gate, k, v, workspace, tokens, stream);
    });
}

} // namespace ninfer::ops::detail
