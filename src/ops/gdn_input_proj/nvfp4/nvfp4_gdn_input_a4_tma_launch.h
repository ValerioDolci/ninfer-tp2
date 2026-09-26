#pragma once
#include "ops/linear/nvfp4/nvfp4_operands.h"

namespace ninfer::ops::detail {
// `p.rows` selects the GDN input [16384,5120] parent or its two-device [8192,5120] shard
// (nvfp4_gdn_input_output.cuh).
void launch_nvfp4_a4_tma_gdn(const Nvfp4A4Operands& p, __nv_bfloat16* qkv, __nv_bfloat16* z,
                             cudaStream_t stream);
} // namespace ninfer::ops::detail
