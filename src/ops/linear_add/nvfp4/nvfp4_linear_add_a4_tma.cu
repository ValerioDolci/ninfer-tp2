#include "ops/linear_add/nvfp4/nvfp4_linear_add_a4_tma_launch.h"
#include "ops/linear/nvfp4/nvfp4_a4_tma.cuh"

namespace ninfer::ops::detail {
namespace {
template <int K>
void launch(const Nvfp4A4Operands& p, __nv_bfloat16* y, cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<256, 3, 1>, K>;
    launch_nvfp4_a4_tma_mma<S>(p, LinearBf16Output{y, p.rows},
                               LinearResidualAddEpilogue{{y, p.rows}}, stream);
}
} // namespace

// The two-device input-column halves [5120,3072] and [5120,8704] run the M256 schedule of the
// problem they halve (3072 = 24 K128 tiles = 12 scale tiles, 8704 = 68 K128 tiles = 34 scale
// tiles of 16 groups), as linear() over the same shard does from T=1024, so rank 0's linear_add
// and rank 1's linear() of one row-parallel pair run the same kernel.
void launch_nvfp4_a4_tma_linear_add(const Nvfp4A4Operands& p, __nv_bfloat16* residual,
                                    cudaStream_t stream) {
    if (p.k == 6144)
        launch<6144>(p, residual, stream);
    else if (p.k == 17408)
        launch<17408>(p, residual, stream);
    else if (p.k == 3072)
        launch<3072>(p, residual, stream);
    else if (p.k == 8704)
        launch<8704>(p, residual, stream);
    else
        throw std::invalid_argument("NVFP4 TMA linear_add: unsupported K");
}
} // namespace ninfer::ops::detail
