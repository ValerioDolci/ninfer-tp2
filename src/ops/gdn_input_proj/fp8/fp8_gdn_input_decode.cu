#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "core/weight.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_output.cuh"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/linear/fp8/fp8_a16_gemv.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

// The parent and the two-device shard share K and the schedule; only the section output differs.
template <class Output>
void launch_decode(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                   cudaStream_t stream) {
    using Geometry = Fp8N16384K5120;
    using Schedule = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
    launch_fp8_a16_gemv<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a16_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
}

} // namespace

void fp8_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 cudaStream_t stream) {
    launch_decode<Fp8GdnInputOutput>(x, weight, qkv, z, stream);
}

void fp8_gdn_input_shard_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                       Tensor& z, cudaStream_t stream) {
    launch_decode<Fp8GdnInputShardOutput>(x, weight, qkv, z, stream);
}

} // namespace ninfer::ops::detail
