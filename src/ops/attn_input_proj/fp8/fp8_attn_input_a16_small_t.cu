#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_output.cuh"
#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "ops/linear/fp8/fp8_instances.cuh"

namespace ninfer::ops::detail {
void fp8_attn_input_a16_small_mma_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                         Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    visit_fp8_attn_input_problem(weight.n, [&]<class Problem>() {
        using Output = typename Problem::Output;
        const Output output{
            static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
            static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
        const auto launch = [&]<class Schedule>() {
            // Row tiles stay within one section at the parent and at the shard.
            static_assert((Problem::kKeyRows % Schedule::kBlockRows) == 0);
            launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Schedule, 5120>>(
                fp8_a16_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
        };
        if (x.ne[1] <= 16) return launch.template operator()<Fp8SlicedInstance<16, 4, 1>>();
        if (x.ne[1] <= 24) return launch.template operator()<Fp8SlicedInstance<32, 4, 2>>();
        launch.template operator()<Fp8SlicedInstance<32, 4, 1>>();
    });
}
} // namespace ninfer::ops::detail
