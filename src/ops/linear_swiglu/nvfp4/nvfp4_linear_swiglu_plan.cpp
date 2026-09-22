#include "core/weight.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include "core/layout.h"
#include "ninfer/ops/silu_mul.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma_launch.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4LinearSwiGluRoute {
    DecodeFusedA16,
    SmallTFusedA16,
    FusedW4A4,
    LinearW4A4Post,
    TmaFusedW4A4,
};

constexpr std::int32_t kFusedMaxTokens = 128;

Nvfp4LinearSwiGluRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 linear_swiglu: T must be positive"); }
    if (!valid_linear_policy(policy)) {
        throw std::invalid_argument("nvfp4 linear_swiglu: invalid compute policy");
    }
    if (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8) {
        if (tokens == 1) { return Nvfp4LinearSwiGluRoute::DecodeFusedA16; }
        if (tokens <= 16) { return Nvfp4LinearSwiGluRoute::SmallTFusedA16; }
        throw std::invalid_argument("nvfp4 linear_swiglu A16 is registered only through T=16");
    }
    if (tokens == 1) { return Nvfp4LinearSwiGluRoute::DecodeFusedA16; }
    if (tokens <= 4) { return Nvfp4LinearSwiGluRoute::SmallTFusedA16; }
    if (tokens <= kFusedMaxTokens) { return Nvfp4LinearSwiGluRoute::FusedW4A4; }
    // This route dispatches its own fused kernel rather than a Linear shape's, so it carries its
    // own condition; the call site below forces the matching scale layout.
    if (tokens >= kNvfp4TmaBlockM && (tokens % kNvfp4TmaBlockM) == 0) {
        return Nvfp4LinearSwiGluRoute::TmaFusedW4A4;
    }
    return Nvfp4LinearSwiGluRoute::LinearW4A4Post;
}

struct Nvfp4LinearSwiGluWorkspace {
    Tensor projected;
    DeviceSpan linear;
};

// Every gate/up problem shares the input width; only the projected row count differs.
constexpr std::int32_t kInputRows = Nvfp4N34816K5120::kInputRows;

template <class Allocator>
Nvfp4LinearSwiGluWorkspace
allocate_baseline_workspace(Allocator& allocator, std::int32_t gate_up_rows, std::int32_t tokens) {
    Nvfp4LinearSwiGluWorkspace out;
    out.projected                  = allocator.alloc(DType::BF16, {gate_up_rows, tokens}, 256);
    const std::size_t linear_bytes = linear_workspace_capacity_bytes(
        QType::NVFP4, gate_up_rows, kInputRows, LinearPolicy::AllowA4, tokens, tokens);
    out.linear = allocator.alloc_bytes(linear_bytes, 256);
    return out;
}

template <class Allocator>
Nvfp4W4a4Workspace allocate_fused_workspace(Allocator& allocator, std::int32_t tokens) {
    return allocate_nvfp4_w4a4_workspace(allocator, tokens, kInputRows);
}

std::size_t baseline_workspace_bytes(std::int32_t gate_up_rows, std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_baseline_workspace(layout, gate_up_rows, tokens);
    return layout.peak_bytes(1);
}

std::size_t fused_workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_fused_workspace(layout, tokens);
    return layout.peak_bytes(1);
}

} // namespace

std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(std::int32_t gate_up_rows,
                                                         LinearPolicy policy,
                                                         std::int32_t min_tokens,
                                                         std::int32_t max_tokens) {
    if (gate_up_rows != Nvfp4N34816K5120::kOutputRows &&
        gate_up_rows != Nvfp4N17408K5120::kOutputRows) {
        throw std::invalid_argument("nvfp4 linear_swiglu workspace: unsupported problem");
    }
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 linear_swiglu workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    (void)resolve_route(policy, max_tokens);
    if ((policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8) || max_tokens <= 4) {
        return 0;
    }

    std::size_t maximum = 0;
    if (min_tokens <= kFusedMaxTokens && max_tokens >= 5) {
        maximum = fused_workspace_bytes(std::min(max_tokens, kFusedMaxTokens));
    }
    if (max_tokens >= kNvfp4TmaBlockM) {
        const std::int32_t largest_fused = max_tokens - (max_tokens % kNvfp4TmaBlockM);
        if (largest_fused >= std::max(min_tokens, kNvfp4TmaBlockM)) {
            maximum = std::max(maximum, fused_workspace_bytes(largest_fused));
        }
    }

    std::int32_t last_baseline = max_tokens;
    if (resolve_route(policy, last_baseline) == Nvfp4LinearSwiGluRoute::TmaFusedW4A4) {
        --last_baseline;
    }
    if (last_baseline >= std::max(min_tokens, kFusedMaxTokens + 1)) {
        maximum = std::max(maximum, baseline_workspace_bytes(gate_up_rows, last_baseline));
    }
    return maximum;
}

void nvfp4_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                  LinearPolicy policy, WorkspaceArena* workspace,
                                  cudaStream_t stream) {
    const Nvfp4LinearSwiGluRoute route = resolve_route(policy, x.ne[1]);
    if (route != Nvfp4LinearSwiGluRoute::DecodeFusedA16 &&
        route != Nvfp4LinearSwiGluRoute::SmallTFusedA16 && workspace == nullptr) {
        throw std::invalid_argument("nvfp4 linear_swiglu: A4 route requires caller workspace");
    }
    switch (route) {
    case Nvfp4LinearSwiGluRoute::DecodeFusedA16:
        nvfp4_linear_swiglu_decode_launch(x, weight, out, stream);
        return;
    case Nvfp4LinearSwiGluRoute::SmallTFusedA16:
        nvfp4_linear_swiglu_small_t_launch(x, weight, out, stream);
        return;
    case Nvfp4LinearSwiGluRoute::FusedW4A4:
        nvfp4_linear_swiglu_w4a4_launch(x, weight, out, *workspace, stream);
        return;
    case Nvfp4LinearSwiGluRoute::TmaFusedW4A4: {
        auto scope                       = workspace->scope();
        const Nvfp4W4a4Workspace scratch = allocate_fused_workspace(*workspace, x.ne[1]);
        launch_nvfp4_w4a4_quantize(x, weight, scratch, Nvfp4ScaleLayout::Tiled, stream);
        const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
        launch_nvfp4_linear_swiglu_w4a4_tma(
            resolve_nvfp4_geometry(weight.n, weight.k), scratch.codes, scratch.scales,
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
            x.ne[1], alpha, stream);
        return;
    }
    case Nvfp4LinearSwiGluRoute::LinearW4A4Post:
        break;
    }

    auto scope                         = workspace->scope();
    Nvfp4LinearSwiGluWorkspace scratch = allocate_baseline_workspace(*workspace, weight.n, x.ne[1]);
    WorkspaceArena linear_workspace(scratch.linear);
    linear(x, weight, scratch.projected, LinearPolicy::AllowA4, linear_workspace, stream);
    const std::int32_t intermediate = weight.n / 2;
    silu_mul(scratch.projected.slice(0, 0, intermediate),
             scratch.projected.slice(0, intermediate, intermediate), out, stream);
}

} // namespace ninfer::ops::detail
