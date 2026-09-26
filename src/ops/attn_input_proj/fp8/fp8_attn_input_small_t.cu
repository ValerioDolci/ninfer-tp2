#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "core/weight.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"

#include "core/device.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_output.cuh"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/linear/fp8/fp8_a16_simt.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                        cudaStream_t);

template <class Problem, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                  Tensor& v, cudaStream_t stream) {
    using Geometry = typename Problem::Geometry;
    using Output   = typename Problem::Output;
    using Schedule =
        Fp8A16SimtSchedule<8, 2, 16, ActiveTokens, 1,
                           (ActiveTokens >= 3 && ActiveTokens <= 4)
                               ? Fp8SimtActivationAccess::SharedPhase
                               : Fp8SimtActivationAccess::TokenPacked,
                           Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
    const Output output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };
    launch_fp8_a16_simt<Fp8ScheduleInstance<Schedule, Geometry::kInputRows, ActiveTokens, true>>(
        fp8_a16_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
}

template <class Problem, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<Problem, 2 + static_cast<int>(Offsets)>...};
}

template <class Problem>
constexpr auto kLaunchers =
    make_launchers<Problem>(std::make_index_sequence<kFp8AttnInputLastSimtT - 2 + 1>{});

} // namespace

void fp8_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                   Tensor& k, Tensor& v, cudaStream_t stream) {
    if (x.ne[1] < 2 || x.ne[1] > kFp8AttnInputLastSimtT) {
        throw std::invalid_argument("fp8 attn_input_proj small-T: unsupported T");
    }
    visit_fp8_attn_input_problem(weight.n, [&]<class Problem>() {
        kLaunchers<Problem>[static_cast<std::size_t>(x.ne[1] - 2)](x, weight, q, gate, k, v,
                                                                   stream);
    });
}

} // namespace ninfer::ops::detail
