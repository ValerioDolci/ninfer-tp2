#pragma once

#include "ops/common/memory.cuh"
#include "ops/linear/nvfp4/nvfp4_geometry.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

// Stores one fused query|key|gate|value parent row into its independent section output. Gate has
// the query's row count and value has the key's.
template <std::int32_t QueryRows, std::int32_t KeyRows>
struct Nvfp4AttentionInputSections {
    static constexpr std::int32_t kQueryRows  = QueryRows;
    static constexpr std::int32_t kKeyRows    = KeyRows;
    static constexpr std::int32_t kGateRows   = QueryRows;
    static constexpr std::int32_t kKeyBegin   = kQueryRows;
    static constexpr std::int32_t kGateBegin  = kKeyBegin + kKeyRows;
    static constexpr std::int32_t kValueBegin = kGateBegin + kGateRows;
    static constexpr std::int32_t kParentRows = kValueBegin + kKeyRows;

    // The W4A4 MMA and TMA tiles store 128-row blocks that must not straddle a section.
    static_assert((kQueryRows % 128) == 0);
    static_assert((kKeyRows % 128) == 0);

    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;

    __device__ __forceinline__ __nv_bfloat16* destination(std::int32_t parent_row,
                                                          std::int32_t token) const {
        if (parent_row < kKeyBegin) {
            return query + static_cast<std::int64_t>(token) * kQueryRows + parent_row;
        }
        if (parent_row < kGateBegin) {
            return key + static_cast<std::int64_t>(token) * kKeyRows + parent_row - kKeyBegin;
        }
        if (parent_row < kValueBegin) {
            return gate + static_cast<std::int64_t>(token) * kGateRows + parent_row - kGateBegin;
        }
        return value + static_cast<std::int64_t>(token) * kKeyRows + parent_row - kValueBegin;
    }

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float result) const {
        *destination(parent_row, token) = __float2bfloat16_rn(result);
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        store_vec(destination(parent_row, token), values);
    }
};

// A registered fused parent: its NVFP4 problem geometry and its section output.
template <class GeometryType, std::int32_t QueryRows, std::int32_t KeyRows>
struct Nvfp4AttnInputProblem {
    using Geometry = GeometryType;
    using Output   = Nvfp4AttentionInputSections<QueryRows, KeyRows>;

    static_assert(Geometry::kOutputRows == Output::kParentRows);
};

// The whole [14336,5120] parent, and one device's [7168,5120] shard under two-device tensor
// parallelism. The shard is a standalone weight whose sections are that device's head-local
// halves of the parent's sections (12 of 24 query/gate heads, 2 of 4 key/value heads), in the
// same query|key|gate|value order.
using Nvfp4AttnInputParent = Nvfp4AttnInputProblem<Nvfp4N14336K5120, 6144, 1024>;
using Nvfp4AttnInputShard  = Nvfp4AttnInputProblem<Nvfp4N7168K5120, 3072, 512>;

// Calls `body.template operator()<Problem>()` for the registered problem with `parent_rows` rows.
template <class Body>
void visit_nvfp4_attn_input_problem(std::int32_t parent_rows, Body&& body) {
    if (parent_rows == Nvfp4AttnInputParent::Geometry::kOutputRows) {
        body.template operator()<Nvfp4AttnInputParent>();
        return;
    }
    if (parent_rows == Nvfp4AttnInputShard::Geometry::kOutputRows) {
        body.template operator()<Nvfp4AttnInputShard>();
        return;
    }
    throw std::invalid_argument("nvfp4 attn_input_proj: unsupported parent rows");
}

} // namespace ninfer::ops::detail
