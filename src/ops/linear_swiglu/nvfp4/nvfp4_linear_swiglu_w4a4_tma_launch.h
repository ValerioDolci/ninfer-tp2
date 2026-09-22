#pragma once

#include "ops/linear/nvfp4/nvfp4_config.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// `problem` is the gate/up geometry: [34816,5120] or its two-device output-row half [17408,5120].
void launch_nvfp4_linear_swiglu_w4a4_tma(Nvfp4GeometryId problem,
                                         const std::uint8_t* activation_codes,
                                         const std::uint8_t* activation_scales,
                                         const std::uint8_t* weight_codes,
                                         const std::uint8_t* weight_scales, __nv_bfloat16* output,
                                         std::int32_t tokens, float alpha, cudaStream_t stream);

} // namespace ninfer::ops::detail
