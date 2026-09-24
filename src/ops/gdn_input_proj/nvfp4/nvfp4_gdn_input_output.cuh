#pragma once

#include "ops/common/memory.cuh"
#include "ops/linear/nvfp4/nvfp4_geometry.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

// Routes parent rows [0,Q+K+V) to qkv and the trailing Z rows to z. KeyRows is shared by the Q and
// K sections and ValueRows by the V and Z sections.
template <std::int32_t KeyRows, std::int32_t ValueRows>
struct Nvfp4GdnInputSectionOutput {
    static constexpr std::int32_t kQkvRows = 2 * KeyRows + ValueRows;
    static constexpr std::int32_t kZRows   = ValueRows;
    static constexpr std::int32_t kRows    = kQkvRows + kZRows;

    // The W4A4 MMA and TMA tiles store 128-row blocks that must not straddle qkv and z.
    static_assert((kQkvRows % 128) == 0);
    static_assert((kZRows % 128) == 0);

    __nv_bfloat16* qkv;
    __nv_bfloat16* z;

    __device__ __forceinline__ __nv_bfloat16* destination(std::int32_t parent_row,
                                                          std::int32_t token) const {
        if (parent_row < kQkvRows) {
            return qkv + static_cast<std::int64_t>(token) * kQkvRows + parent_row;
        }
        return z + static_cast<std::int64_t>(token) * kZRows + parent_row - kQkvRows;
    }

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        *destination(parent_row, token) = __float2bfloat16_rn(value);
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        store_vec(destination(parent_row, token), values);
    }
};

// A registered GDN input parent: its NVFP4 problem geometry and its section output.
template <class GeometryType, std::int32_t KeyRows, std::int32_t ValueRows>
struct Nvfp4GdnInputProblem {
    using Geometry = GeometryType;
    using Output   = Nvfp4GdnInputSectionOutput<KeyRows, ValueRows>;

    static_assert(Geometry::kOutputRows == Output::kRows);
};

// The whole [16384,5120] parent, and one device's [8192,5120] shard under two-device tensor
// parallelism: a standalone weight holding that device's 8 of 16 key heads and 24 of 48 value
// heads in the same Q|K|V|Z order.
using Nvfp4GdnInputParent = Nvfp4GdnInputProblem<Nvfp4N16384K5120, 2048, 6144>;
using Nvfp4GdnInputShard  = Nvfp4GdnInputProblem<Nvfp4N8192K5120, 1024, 3072>;

// Calls `body.template operator()<Problem>()` for the registered problem with `parent_rows` rows.
template <class Body>
void visit_nvfp4_gdn_input_problem(std::int32_t parent_rows, Body&& body) {
    if (parent_rows == Nvfp4GdnInputParent::Geometry::kOutputRows) {
        body.template operator()<Nvfp4GdnInputParent>();
        return;
    }
    if (parent_rows == Nvfp4GdnInputShard::Geometry::kOutputRows) {
        body.template operator()<Nvfp4GdnInputShard>();
        return;
    }
    throw std::invalid_argument("nvfp4 gdn_input_proj: unsupported parent rows");
}

} // namespace ninfer::ops::detail
