#pragma once
#include "ops/linear/common/output.cuh"
#include "ops/linear/nvfp4/nvfp4_geometry.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
using Nvfp4GdnInputOutput = LinearBf16SegmentedOutput<10240, 6144>;

// Two-device shard: each rank's contiguous [8192,5120] parent holds its 8 of 16 key heads and 24
// of 48 value heads in the same Q|K|V|Z order, so qkv takes 1024 + 1024 + 3072 rows and z 3072.
// Both sections are whole 128-row tiles, like the parent's.
using Nvfp4GdnInputShardOutput = LinearBf16SegmentedOutput<5120, 3072>;

// Calls `body.template operator()<Output>()` with the section output of the registered parent with
// `parent_rows` (`weight.n`) rows: the whole [16384,5120] parent or one device's [8192,5120]
// shard. Both share K, so the shard runs the parent's instances, routes and workspace.
template <class Body>
void visit_nvfp4_gdn_input_output(std::int32_t parent_rows, Body&& body) {
    if (parent_rows == Nvfp4N16384K5120::kOutputRows) {
        body.template operator()<Nvfp4GdnInputOutput>();
        return;
    }
    if (parent_rows == Nvfp4N8192K5120::kOutputRows) {
        body.template operator()<Nvfp4GdnInputShardOutput>();
        return;
    }
    throw std::invalid_argument("nvfp4 gdn_input_proj: unsupported parent rows");
}
} // namespace ninfer::ops::detail
