#include "core/weight.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_gemv.cuh"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using Schedule = Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
static_assert(Schedule::kRowsPerWarp == 2);

// Geometry is the gate/up problem: gate rows [0,M) precede their up rows [M,2M), M = N/2.
template <class Geometry>
void launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int kIntermediate = Geometry::kOutputRows / 2;
    static_assert((kIntermediate % Schedule::kWarpsPerCta) == 0);
    using Rows = Fp8SwiGluRows<Schedule::kRowsPerWarp / 2, kIntermediate>;
    if (x.ne[0] != Geometry::kInputRows || x.ne[1] != 1 || out.ne[0] != kIntermediate ||
        out.ne[1] != 1) {
        throw std::invalid_argument("fp8 linear_swiglu decode: invalid exact problem");
    }
    constexpr int kBlocks = kIntermediate / Schedule::kWarpsPerCta;
    const Rows rows{};
    const Fp8SwiGluOutput output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    fp8_gemv_kernel<Geometry, Schedule, Fp8SwiGluOutput, Rows, true>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), output, rows);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void fp8_linear_swiglu_decode_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                     cudaStream_t stream) {
    switch (resolve_fp8_geometry(weight.n, weight.k)) {
    case Fp8GeometryId::N34816K5120:
        launch<Fp8N34816K5120>(x, weight, out, stream);
        return;
    case Fp8GeometryId::N17408K5120:
        launch<Fp8N17408K5120>(x, weight, out, stream);
        return;
    case Fp8GeometryId::N14336K5120:
    case Fp8GeometryId::N16384K5120:
    case Fp8GeometryId::N248320K5120:
    case Fp8GeometryId::N5120K6144:
    case Fp8GeometryId::N5120K17408:
    case Fp8GeometryId::N7168K5120:
    case Fp8GeometryId::N8192K5120:
    case Fp8GeometryId::N124160K5120:
    case Fp8GeometryId::N5120K3072:
    case Fp8GeometryId::N5120K8704:
        break;
    }
    throw std::invalid_argument("fp8 linear_swiglu: unsupported problem");
}

} // namespace ninfer::ops::detail
