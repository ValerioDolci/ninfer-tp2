#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_a4_tma_launch.h"
#include "ops/linear/nvfp4/nvfp4_a4_tma.cuh"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh"

namespace ninfer::ops::detail {
namespace {

template <class Problem, class Schedule>
void launch_tma(const Nvfp4A4Operands& p, __nv_bfloat16* query, __nv_bfloat16* gate,
                __nv_bfloat16* key, __nv_bfloat16* value, cudaStream_t stream) {
    using Output = typename Problem::Output;
    // TMA row tiles stay within one section at the parent and at the shard.
    static_assert((Problem::kQueryRows % Schedule::kBlockRows) == 0);
    static_assert((Problem::kKeyRows % Schedule::kBlockRows) == 0);
    launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<Schedule, Problem::Geometry::kInputRows>>(
        p, Output{query, key, gate, value}, LinearIdentityEpilogue{}, stream);
}

} // namespace

void launch_nvfp4_a4_tma_attention(const Nvfp4A4Operands& p, __nv_bfloat16* query,
                                   __nv_bfloat16* gate, __nv_bfloat16* key, __nv_bfloat16* value,
                                   cudaStream_t stream) {
    // `p.rows` selects the [14336,5120] parent or its two-device [7168,5120] shard; the shard keeps
    // the parent's TMA schedules.
    visit_nvfp4_attn_input_problem(p.rows, [&]<class Problem>() {
        if (p.scale_layout == Nvfp4ScaleLayout::Tiled128)
            launch_tma<Problem, Nvfp4A4TmaMmaSchedule<128, 4, 1>>(p, query, gate, key, value,
                                                                  stream);
        else
            launch_tma<Problem, Nvfp4A4TmaMmaSchedule<256, 3, 1>>(p, query, gate, key, value,
                                                                  stream);
    });
}
} // namespace ninfer::ops::detail
