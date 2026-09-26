#pragma once
#include "ops/linear/common/output.cuh"

namespace ninfer::ops::detail {
using Fp8GdnInputOutput = LinearBf16SegmentedOutput<10240, 6144>;

// Two-device shard: each rank's contiguous [8192,5120] parent holds its 8 of 16 key heads and 24
// of 48 value heads in the same Q|K|V|Z order, so qkv takes 1024 + 1024 + 3072 rows and z 3072.
using Fp8GdnInputShardOutput = LinearBf16SegmentedOutput<5120, 3072>;
} // namespace ninfer::ops::detail
