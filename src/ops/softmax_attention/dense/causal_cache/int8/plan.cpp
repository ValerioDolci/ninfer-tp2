#include "ops/softmax_attention/dense/causal_cache/int8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/int8/operands.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr int kGroupedPrefillMaxWidth = 256;
} // namespace

Int8KvCausalPlan make_int8_kv_causal_plan(int heads, int width, int batch,
                                          CausalAttentionExecutionEnvelope envelope, int sms) {
    if ((heads != 24 && heads != 16 && heads != 12) || width < 1 || batch < 1 || batch > 8 ||
        sms < 1 ||
        (batch > 1 && width > 16) || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys)
        throw std::invalid_argument("INT8 attention: invalid plan inputs");
    constexpr int grouped_limit = Int8KvCausalPlan::kTokenTile;
    const auto family           = width <= grouped_limit             ? Int8KvFamily::Grouped
                                  : width <= kGroupedPrefillMaxWidth ? Int8KvFamily::ParallelGrouped
                                                                     : Int8KvFamily::Tiled;
    const int tiles =
        family == Int8KvFamily::ParallelGrouped ? (width + grouped_limit - 1) / grouped_limit : 1;
    const int independent_tiles = batch * (heads == 24 ? 4 : 2) * tiles;
    const int wave_ctas         = (sms / independent_tiles) * independent_tiles;
    // Budgets count the launching device's SMs. The 12/2 and 24/4 grouped schedules keep two CTAs
    // per SM (launch bound), so 2 * sms is one full wave; the fixed 5090 count (340 CTAs) left 2.4
    // waves on a 70-SM RTX 5070 Ti. 12/2, one device's half of 24/4 at tp 2, has 24/4's group of
    // six and therefore its CTA shape, wave budget and split granularity. Measured at 12/2 on the
    // 5070 Ti: 1, 1.5, 3 and 4 * sms budgets and key shifts one step either way do not beat this.
    const int budget = heads != 16 || width <= 4 || wave_ctas < sms * 9 / 10 ? 2 * sms : sms;
    CausalKvPartition partition{
        1, std::clamp(budget / independent_tiles, 1, CausalKvPartition::kMaxSplits)};
    // Bound partial traffic by keeping enough KV work in each split.
    partition.key_shift = (width == 1 ? 7 : 8) - (heads == 16 ? 1 : 0);
    partition.capacity  = partition.active(envelope.max_visible_keys);
    return {family, heads, width, batch, envelope, partition};
}

std::size_t int8_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                    CausalAttentionExecutionEnvelope envelope) {
    std::size_t maximum = 0;
    for (int width = min_width; width <= std::min(max_width, kGroupedPrefillMaxWidth); ++width) {
        for (const int sms : causal_attention_device_sm_counts()) {
            const auto plan = make_int8_kv_causal_plan(heads, width, batch, envelope, sms);
            if (plan.family == Int8KvFamily::Tiled) continue;
            const int splits = plan.partition.capacity;
            WorkspaceLayoutBuilder layout;
            (void)allocate_causal_partials(layout, heads, width, splits, batch);
            maximum = std::max(maximum, layout.peak_bytes(1));
        }
    }
    return maximum;
}

} // namespace ninfer::ops::detail
