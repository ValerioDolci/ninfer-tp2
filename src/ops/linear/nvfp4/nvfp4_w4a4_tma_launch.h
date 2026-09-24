#pragma once

#include "ops/linear/nvfp4/nvfp4_config.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void launch_nvfp4_w4a4_tma_linear(Nvfp4GeometryId problem, const std::uint8_t* activation_codes,
                                  const std::uint8_t* activation_scales,
                                  const std::uint8_t* weight_codes,
                                  const std::uint8_t* weight_scales, __nv_bfloat16* output,
                                  std::int32_t tokens, float alpha, cudaStream_t stream);

// `parent_rows` selects the fused [14336,5120] attention parent or its two-device [7168,5120]
// shard (attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh).
void launch_nvfp4_w4a4_tma_attention(std::int32_t parent_rows,
                                     const std::uint8_t* activation_codes,
                                     const std::uint8_t* activation_scales,
                                     const std::uint8_t* weight_codes,
                                     const std::uint8_t* weight_scales, __nv_bfloat16* query,
                                     __nv_bfloat16* gate, __nv_bfloat16* key, __nv_bfloat16* value,
                                     std::int32_t tokens, float alpha, cudaStream_t stream);

// `parent_rows` selects the GDN input [16384,5120] parent or its two-device [8192,5120] shard
// (gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh).
void launch_nvfp4_w4a4_tma_gdn(std::int32_t parent_rows, const std::uint8_t* activation_codes,
                               const std::uint8_t* activation_scales,
                               const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                               __nv_bfloat16* qkv, __nv_bfloat16* z, std::int32_t tokens,
                               float alpha, cudaStream_t stream);

void launch_nvfp4_w4a4_tma_linear_add(Nvfp4GeometryId problem, const std::uint8_t* activation_codes,
                                      const std::uint8_t* activation_scales,
                                      const std::uint8_t* weight_codes,
                                      const std::uint8_t* weight_scales, __nv_bfloat16* residual,
                                      std::int32_t tokens, float alpha, cudaStream_t stream);

} // namespace ninfer::ops::detail
