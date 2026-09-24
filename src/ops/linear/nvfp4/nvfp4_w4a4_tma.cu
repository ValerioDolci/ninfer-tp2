#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"

#include "core/device.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/launcher/kernel_attr_once.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma.cuh"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_epilogue.cuh"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using TmaM256N128 = Nvfp4W4a4TmaSchedule<256, 3, 1>;
// K128 consumes 64 code bytes per row. Prefetch the adjacent half-line for the next K tile.
using TmaM256N128Prefetch128B = Nvfp4W4a4TmaSchedule<256, 3, 1, CU_TENSOR_MAP_L2_PROMOTION_L2_128B>;

template <class Geometry, class Schedule, class Epilogue, class Output>
void launch_tma(const std::uint8_t* activation_codes, const std::uint8_t* activation_scales,
                const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                std::int32_t tokens, float alpha, Epilogue epilogue, Output output,
                cudaStream_t stream) {
    const Nvfp4W4a4TmaDescriptors descriptors =
        make_nvfp4_w4a4_tma_descriptors<Geometry, Schedule::kBlockM>(
            activation_codes, activation_scales, weight_codes, weight_scales, tokens,
            Schedule::kWeightCodePromotion);
    constexpr std::size_t kSharedBytes = sizeof(Nvfp4W4a4TmaSharedStorage<Schedule>);
    static FuncAttrPerDevice attr;
    attr.ensure(nvfp4_w4a4_tma_kernel<Geometry, Schedule, Epilogue, Output>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSharedBytes));

    // The last M tile may be partial; the kernel bounds itself by the real token count.
    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN,
                    (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
    nvfp4_w4a4_tma_kernel<Geometry, Schedule><<<grid, Schedule::kThreads, kSharedBytes, stream>>>(
        descriptors, alpha, epilogue, output, tokens);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, class Schedule = TmaM256N128>
void launch_linear(const std::uint8_t* activation_codes, const std::uint8_t* activation_scales,
                   const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                   __nv_bfloat16* output, std::int32_t tokens, float alpha, cudaStream_t stream) {
    launch_tma<Geometry, Schedule>(activation_codes, activation_scales, weight_codes, weight_scales,
                                   tokens, alpha, Nvfp4IdentityEpilogue{},
                                   Nvfp4ContiguousOutput{output, Geometry::kOutputRows}, stream);
}

} // namespace

void launch_nvfp4_w4a4_tma_linear(Nvfp4GeometryId problem, const std::uint8_t* activation_codes,
                                  const std::uint8_t* activation_scales,
                                  const std::uint8_t* weight_codes,
                                  const std::uint8_t* weight_scales, __nv_bfloat16* output,
                                  std::int32_t tokens, float alpha, cudaStream_t stream) {
    switch (problem) {
    case Nvfp4GeometryId::N14336K5120:
        launch_linear<Nvfp4N14336K5120>(activation_codes, activation_scales, weight_codes,
                                        weight_scales, output, tokens, alpha, stream);
        return;
    case Nvfp4GeometryId::N16384K5120:
        launch_linear<Nvfp4N16384K5120>(activation_codes, activation_scales, weight_codes,
                                        weight_scales, output, tokens, alpha, stream);
        return;
    case Nvfp4GeometryId::N34816K5120:
        launch_linear<Nvfp4N34816K5120, TmaM256N128Prefetch128B>(
            activation_codes, activation_scales, weight_codes, weight_scales, output, tokens, alpha,
            stream);
        return;
    case Nvfp4GeometryId::N5120K6144:
        launch_linear<Nvfp4N5120K6144>(activation_codes, activation_scales, weight_codes,
                                       weight_scales, output, tokens, alpha, stream);
        return;
    case Nvfp4GeometryId::N5120K17408:
        launch_linear<Nvfp4N5120K17408>(activation_codes, activation_scales, weight_codes,
                                        weight_scales, output, tokens, alpha, stream);
        return;
    // The two-device shards keep their parent's TMA schedule.
    case Nvfp4GeometryId::N17408K5120:
        launch_linear<Nvfp4N17408K5120, TmaM256N128Prefetch128B>(
            activation_codes, activation_scales, weight_codes, weight_scales, output, tokens, alpha,
            stream);
        return;
    case Nvfp4GeometryId::N5120K8704:
        launch_linear<Nvfp4N5120K8704>(activation_codes, activation_scales, weight_codes,
                                       weight_scales, output, tokens, alpha, stream);
        return;
    case Nvfp4GeometryId::N5120K3072:
        launch_linear<Nvfp4N5120K3072>(activation_codes, activation_scales, weight_codes,
                                       weight_scales, output, tokens, alpha, stream);
        return;
    }
}

void launch_nvfp4_w4a4_tma_attention(std::int32_t parent_rows,
                                     const std::uint8_t* activation_codes,
                                     const std::uint8_t* activation_scales,
                                     const std::uint8_t* weight_codes,
                                     const std::uint8_t* weight_scales, __nv_bfloat16* query,
                                     __nv_bfloat16* gate, __nv_bfloat16* key, __nv_bfloat16* value,
                                     std::int32_t tokens, float alpha, cudaStream_t stream) {
    // The two-device shard keeps the parent's schedule.
    visit_nvfp4_attn_input_problem(parent_rows, [&]<class Problem>() {
        using Output = typename Problem::Output;
        static_assert((Output::kQueryRows % TmaM256N128::kBlockN) == 0);
        static_assert((Output::kKeyRows % TmaM256N128::kBlockN) == 0);
        launch_tma<typename Problem::Geometry, TmaM256N128>(
            activation_codes, activation_scales, weight_codes, weight_scales, tokens, alpha,
            Nvfp4IdentityEpilogue{}, Output{query, key, gate, value}, stream);
    });
}

void launch_nvfp4_w4a4_tma_gdn(const std::uint8_t* activation_codes,
                               const std::uint8_t* activation_scales,
                               const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                               __nv_bfloat16* qkv, __nv_bfloat16* z, std::int32_t tokens,
                               float alpha, cudaStream_t stream) {
    launch_tma<Nvfp4N16384K5120, TmaM256N128>(activation_codes, activation_scales, weight_codes,
                                              weight_scales, tokens, alpha, Nvfp4IdentityEpilogue{},
                                              Nvfp4GdnInputOutput{qkv, z}, stream);
}

template <class Geometry>
void launch_linear_add(const std::uint8_t* activation_codes, const std::uint8_t* activation_scales,
                       const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                       __nv_bfloat16* residual, std::int32_t tokens, float alpha,
                       cudaStream_t stream) {
    launch_tma<Geometry, TmaM256N128>(
        activation_codes, activation_scales, weight_codes, weight_scales, tokens, alpha,
        Nvfp4AddResidualEpilogue{residual, Geometry::kOutputRows},
        Nvfp4ContiguousOutput{residual, Geometry::kOutputRows}, stream);
}

void launch_nvfp4_w4a4_tma_linear_add(Nvfp4GeometryId problem, const std::uint8_t* activation_codes,
                                      const std::uint8_t* activation_scales,
                                      const std::uint8_t* weight_codes,
                                      const std::uint8_t* weight_scales, __nv_bfloat16* residual,
                                      std::int32_t tokens, float alpha, cudaStream_t stream) {
    switch (problem) {
    case Nvfp4GeometryId::N5120K6144:
        launch_linear_add<Nvfp4N5120K6144>(activation_codes, activation_scales, weight_codes,
                                           weight_scales, residual, tokens, alpha, stream);
        return;
    case Nvfp4GeometryId::N5120K17408:
        launch_linear_add<Nvfp4N5120K17408>(activation_codes, activation_scales, weight_codes,
                                            weight_scales, residual, tokens, alpha, stream);
        return;
    // The two-device input-column halves of [5120,17408] and [5120,6144], with their parents'
    // schedule: rank 0's linear_add then runs the same kernel as rank 1's linear() over the other
    // half.
    case Nvfp4GeometryId::N5120K8704:
        launch_linear_add<Nvfp4N5120K8704>(activation_codes, activation_scales, weight_codes,
                                           weight_scales, residual, tokens, alpha, stream);
        return;
    case Nvfp4GeometryId::N5120K3072:
        launch_linear_add<Nvfp4N5120K3072>(activation_codes, activation_scales, weight_codes,
                                           weight_scales, residual, tokens, alpha, stream);
        return;
    case Nvfp4GeometryId::N14336K5120:
    case Nvfp4GeometryId::N16384K5120:
    case Nvfp4GeometryId::N34816K5120:
    case Nvfp4GeometryId::N17408K5120:
        break;
    }
    throw std::logic_error("nvfp4 W4A4 TMA linear_add has no route for this geometry");
}

} // namespace ninfer::ops::detail
