#include "core/weight.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"

#include "core/device.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_output.cuh"
#include "ops/linear/fp8/fp8_a8_mma.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <class Problem, class Schedule, bool FullTokens>
void launch_mma(const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                Fp8A8Workspace workspace, std::int32_t tokens, cudaStream_t stream) {
    using Geometry = typename Problem::Geometry;
    using Output   = typename Problem::Output;
    static_assert((Output::kQueryRows % Schedule::kBlockRows) == 0);
    static_assert((Output::kKeyRows % Schedule::kBlockRows) == 0);
    constexpr int kRowTiles = Geometry::kOutputRows / Schedule::kBlockRows;
    const int token_tiles   = (tokens + Schedule::kBlockTokens - 1) / Schedule::kBlockTokens;
    const int blocks        = kRowTiles * token_tiles;
    const Output output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };

    static_assert(Schedule::kSharedBytes <= 48 * 1024);
    fp8_mma_kernel<Geometry, Schedule, FullTokens>
        <<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), tokens, Fp8IdentityEpilogue{},
            output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Problem, class Schedule>
void run(const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
         Fp8A8Workspace workspace, int tokens, cudaStream_t stream) {
    if (tokens % Schedule::kBlockTokens == 0)
        launch_mma<Problem, Schedule, true>(weight, q, gate, k, v, workspace, tokens, stream);
    else
        launch_mma<Problem, Schedule, false>(weight, q, gate, k, v, workspace, tokens, stream);
}

template <class Problem>
void run_problem(const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                 Fp8A8Workspace workspace, int tokens, cudaStream_t stream) {
    // This Op owns its tile choices; the generic Linear schedules do not describe four-output
    // projection's short-column cost. All variants share the same activation representation.
    using Small32   = Fp8MmaSchedule<32, 64, 128, 1, 2, 3, 2, Cache::cg, Cache::cg,
                                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
    using Small64   = Fp8MmaSchedule<64, 64, 128, 2, 2, 3, 2, Cache::cg, Cache::cg,
                                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
    using ShortTail = Fp8MmaSchedule<32, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
    using Wide128   = Fp8MmaSchedule<64, 64, 128, 2, 2, 2, 3, Cache::cg, Cache::cg,
                                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
    using Tail144   = Fp8MmaSchedule<48, 128, 128, 3, 4, 2, 2, Cache::cg, Cache::cg,
                                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
    using Prefill   = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
    if (tokens <= 32)
        run<Problem, Small32>(weight, q, gate, k, v, workspace, tokens, stream);
    else if (tokens <= 64)
        run<Problem, Small64>(weight, q, gate, k, v, workspace, tokens, stream);
    else if (tokens <= 96)
        run<Problem, ShortTail>(weight, q, gate, k, v, workspace, tokens, stream);
    else if (tokens <= 128)
        run<Problem, Wide128>(weight, q, gate, k, v, workspace, tokens, stream);
    else if (tokens <= 144)
        run<Problem, Tail144>(weight, q, gate, k, v, workspace, tokens, stream);
    else
        run<Problem, Prefill>(weight, q, gate, k, v, workspace, tokens, stream);
}
} // namespace

void fp8_attn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& k, Tensor& v, Fp8A8Workspace workspace, cudaStream_t stream) {
    visit_fp8_attn_input_problem(weight.n, [&]<class Problem>() {
        launch_fp8_a8_quantize(x, weight, workspace, stream);
        run_problem<Problem>(weight, q, gate, k, v, workspace, x.ne[1], stream);
    });
}
} // namespace ninfer::ops::detail
