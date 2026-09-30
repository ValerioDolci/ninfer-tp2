#pragma once

#include <cuda_runtime.h>

#include <vector>

namespace ninfer::ops::detail {

// RTX 5090 target, still used by the FP8, NVFP4 and K8V4 plans. Wave budgets remain owned by
// each dtype plan.
inline constexpr int kCausalAttentionSmCount = 170;

// Streaming multiprocessors of the current device (170 on an RTX 5090, 70 on an RTX 5070 Ti),
// queried once per device. At tp 2 every rank plans on its own device.
int causal_attention_sm_count();

// Distinct SM counts of the visible devices, ascending. Workspace capacity is queried before the
// launching device is known, so it takes the largest plan over these. Without a visible device
// the list is {kCausalAttentionSmCount}.
const std::vector<int>& causal_attention_device_sm_counts();

// Capture reserves partials for the largest live row. Producer and merge use
// the same live count; a wider capture never changes a row's work partition.
struct CausalKvPartition {
    static constexpr int kMaxSplits = 256;
    int capacity                    = 1;
    int target                      = 1;
    int key_shift                   = 6; // log2 of the minimum KV keys per split

    __host__ __device__ int active(int visible) const {
        const int count = (visible + (1 << key_shift) - 1) >> key_shift;
        return count < target ? count : target;
    }
};

} // namespace ninfer::ops::detail
