#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "core/weight.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/linear/fp8/fp8_a16_gemv.cuh"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// The two-device half [17408,5120] shares K with [34816,5120] and inherits its schedule; it was
// not re-measured at the half.
constexpr int kInputRows = Fp8N34816K5120::kInputRows;
using Schedule           = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
static_assert(Schedule::kRowsPerWarp == 2);

// IntermediateRows is M = N/2 of the gate/up problem: gate rows [0,M) precede their up rows
// [M,2M).
template <int IntermediateRows>
void launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    static_assert((IntermediateRows % Schedule::kWarpsPerCta) == 0);
    using Rows = Fp8SwiGluRows<Schedule::kRowsPerWarp / 2, IntermediateRows>;
    if (x.ne[0] != kInputRows || x.ne[1] != 1 || out.ne[0] != IntermediateRows ||
        out.ne[1] != 1 || weight.n != 2 * IntermediateRows || weight.k != kInputRows) {
        throw std::invalid_argument("fp8 linear_swiglu decode: invalid exact problem");
    }
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), IntermediateRows};
    launch_fp8_a16_gemv<Fp8ScheduleInstance<Schedule, kInputRows>>(
        fp8_a16_operands(x, weight), output, Fp8SwiGluEpilogue{}, stream, Rows{});
}

} // namespace

void fp8_linear_swiglu_decode_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                     cudaStream_t stream) {
    visit_fp8_swiglu_intermediate_rows(weight.n, [&]<int IntermediateRows>() {
        launch<IntermediateRows>(x, weight, out, stream);
    });
}

} // namespace ninfer::ops::detail
