#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// The gate/up problems are [34816,5120] and its two-device output-row half [17408,5120]. They share
// the input width, so the workspace does not depend on which one runs; the launchers below take the
// problem from the weight's shape.
[[nodiscard]] std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                                       std::int32_t min_tokens,
                                                                       std::int32_t max_tokens);

void nvfp4_linear_swiglu_decode_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream);
void nvfp4_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
void nvfp4_linear_swiglu_a4_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                   WorkspaceArena& workspace, cudaStream_t stream);

// `workspace` may be null when the resolved route needs none; a quantized-activation route then
// throws.
void nvfp4_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                  LinearPolicy policy, WorkspaceArena* workspace,
                                  cudaStream_t stream);

} // namespace ninfer::ops::detail
