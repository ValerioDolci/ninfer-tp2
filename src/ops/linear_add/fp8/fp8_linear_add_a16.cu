#include "ops/linear/fp8/fp8_instances.cuh"
#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "core/weight.h"
#include "ops/linear_add/fp8/fp8_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/fp8/fp8_a16_simt.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
// The attention output family: [5120,6144] and its two-device input-column half [5120,3072]. The
// other family is the MLP down projection [5120,17408] and its half [5120,8704].
constexpr bool output_family(int k) { return k == 6144 || k == 3072; }
} // namespace

// The two-device halves [5120,3072] and [5120,8704] inherit the schedules of the problems they
// halve; they were not re-measured at the half K. Every K step below (SIMT 256, sliced-K 128, 256
// or 512, MMA 64 or 128) divides 3072 and 8704.
template <int K>
void launch_matrix(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    const auto operands = fp8_a16_operands(x, weight);
    auto* data          = static_cast<__nv_bfloat16*>(residual.data);
    const LinearBf16Output output{data, weight.n};
    const LinearResidualAddEpilogue epilogue{{data, weight.n}};
    // The short-K residual projection retains its lower-latency SIMT decode capacities.
    if constexpr (output_family(K)) {
        if (x.ne[1] <= 4) {
            const auto tiny = [&]<int Tokens>() {
                using Schedule =
                    Fp8A16SimtSchedule<8, 1, 16, Tokens, 1, Fp8SimtActivationAccess::TokenPacked,
                                       Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous,
                                       1>;
                launch_fp8_a16_simt<Fp8ScheduleInstance<Schedule, K, Tokens, true>>(
                    operands, output, epilogue, stream);
            };
            if (x.ne[1] == 2) return tiny.template operator()<2>();
            if (x.ne[1] == 3) return tiny.template operator()<3>();
            return tiny.template operator()<4>();
        }
    }
    const auto sliced = [&]<int T, int W, int Stages>() {
        launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8SlicedInstance<T, W, Stages>, K>>(
            operands, output, epilogue, stream);
    };
    if constexpr (!output_family(K)) {
        if (x.ne[1] <= 8) return sliced.template operator()<8, 8, 2>();
    }
    if (x.ne[1] <= 16) return sliced.template operator()<16, 8, 2>();
    if constexpr (output_family(K)) {
        if (x.ne[1] <= 32) return sliced.template operator()<16, 4, 2>();
        if (x.ne[1] <= 64) return sliced.template operator()<32, 4, 1>();
    } else {
        if (x.ne[1] <= 32) return sliced.template operator()<32, 8, 1>();
        if (x.ne[1] <= 64) return sliced.template operator()<64, 2, 2>();
    }
    if (x.ne[1] <= 128)
        return launch_fp8_a16_mma<
            Fp8ScheduleInstance<Fp8A16MmaSchedule<64, 64, 128, 32, 16, 2, 2>, K>>(operands, output,
                                                                                  epilogue, stream);
    launch_fp8_a16_mma<Fp8ScheduleInstance<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>, K>>(
        operands, output, epilogue, stream);
}

void fp8_linear_add_matrix_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                  cudaStream_t stream) {
    switch (weight.k) {
    case 6144:
        return launch_matrix<6144>(x, weight, residual, stream);
    case 17408:
        return launch_matrix<17408>(x, weight, residual, stream);
    case 3072:
        return launch_matrix<3072>(x, weight, residual, stream);
    case 8704:
        return launch_matrix<8704>(x, weight, residual, stream);
    default:
        throw std::invalid_argument("fp8 linear_add matrix: unsupported problem");
    }
}
} // namespace ninfer::ops::detail
