#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_output.cuh"
#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "ops/linear/fp8/fp8_instances.cuh"

namespace ninfer::ops::detail {
namespace {
using Tma64x128  = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Tma192x128 = Fp8A8TmaMmaSchedule<192, 128, 128, 3, 4, 2, 1>;
using MidBulk = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 2, 1>, 170, 4, 8>;
using Bulk    = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;

// The parent and the two-device shard share K, so the shard runs the parent's schedules and token
// cutoffs (not re-measured at the shard) with its own Q|K|V|Z section output; the row count comes
// from the weight. Parent sections 10240|6144 and shard sections 5120|3072 are whole row tiles of
// every schedule below.
template <class Output>
void launch_a8(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
               Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
    const auto operands = fp8_a8_operands(weight, workspace, x.ne[1]);
    const auto launch   = [&]<class Schedule>() {
        static_assert(3072 % Schedule::kBlockRows == 0 && 5120 % Schedule::kBlockRows == 0);
        using S = Fp8ScheduleInstance<Schedule, 5120>;
        if constexpr (S::kTmaSwizzle)
            launch_fp8_a8_tma_mma<S>(operands, output, LinearIdentityEpilogue{}, stream,
                                       workspace.partials);
        else
            launch_fp8_a8_mma<S>(operands, output, LinearIdentityEpilogue{}, stream);
    };
    if (x.ne[1] <= 32) return launch.template operator()<Fp8A8T32R32K128>();
    if (x.ne[1] <= 64) return launch.template operator()<Fp8A8T64R128K256>();
    if (x.ne[1] <= 128) return launch.template operator()<Tma64x128>();
    if (x.ne[1] <= 192) return launch.template operator()<Tma192x128>();
    // Smaller output tiles leave only two full-K tiles to split near the 512-token anchor.
    if (x.ne[1] > 384 && x.ne[1] <= 512) return launch.template operator()<MidBulk>();
    launch.template operator()<Bulk>();
}

} // namespace

std::size_t fp8_gdn_input_partial_capacity_bytes(std::int32_t max_tokens) {
    return max_tokens > 256 ? Bulk::kPartialBytes : 0;
}

std::size_t fp8_gdn_input_partial_capacity_bytes(std::int32_t rows, std::int32_t max_tokens) {
    if (rows == Fp8N16384K5120::kOutputRows)
        return fp8_gdn_input_partial_capacity_bytes(max_tokens);
    // 32 row tiles: Bulk splits its last wave from T=193 (64 tiles) on; MidBulk needs less.
    return max_tokens > 192 ? Bulk::kPartialBytes : 0;
}

void fp8_gdn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                             Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_a8<Fp8GdnInputOutput>(x, weight, qkv, z, workspace, stream);
}

void fp8_gdn_input_shard_a8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                   Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_a8<Fp8GdnInputShardOutput>(x, weight, qkv, z, workspace, stream);
}

} // namespace ninfer::ops::detail
