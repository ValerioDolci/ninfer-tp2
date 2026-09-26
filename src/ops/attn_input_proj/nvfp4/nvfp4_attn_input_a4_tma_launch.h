#pragma once
#include "ops/linear/nvfp4/nvfp4_operands.h"

namespace ninfer::ops::detail {
// `p.rows` selects the fused [14336,5120] parent or its two-device [7168,5120] shard
// (nvfp4_attn_input_output.cuh).
void launch_nvfp4_a4_tma_attention(const Nvfp4A4Operands& p, __nv_bfloat16* query,
                                   __nv_bfloat16* gate, __nv_bfloat16* key, __nv_bfloat16* value,
                                   cudaStream_t stream);
} // namespace ninfer::ops::detail
