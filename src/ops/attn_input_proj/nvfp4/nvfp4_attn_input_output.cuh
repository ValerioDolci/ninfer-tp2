#pragma once
#include "ops/linear/common/output.cuh"
#include "ops/linear/nvfp4/nvfp4_geometry.h"

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail {
using Nvfp4AttentionInputOutput = LinearBf16SegmentedOutput<6144, 1024, 6144, 1024>;

// A registered fused parent: its NVFP4 geometry and its query|key|gate|value section output. Gate
// has the query's row count and value has the key's.
template <class GeometryType, std::int32_t QueryRows, std::int32_t KeyRows>
struct Nvfp4AttnInputProblem {
    using Geometry = GeometryType;
    using Output   = LinearBf16SegmentedOutput<QueryRows, KeyRows, QueryRows, KeyRows>;

    static constexpr std::int32_t kQueryRows = QueryRows;
    static constexpr std::int32_t kKeyRows   = KeyRows;

    static_assert(Geometry::kOutputRows == 2 * (QueryRows + KeyRows));
    // Every row tile of the routes below (at most the 128-row A4 MMA and TMA tiles) divides each
    // section, so a tile binds to one section and vector stores never straddle two.
    static_assert((QueryRows % 128) == 0 && (KeyRows % 128) == 0);
};

// The whole [14336,5120] parent, and one device's [7168,5120] shard under two-device tensor
// parallelism. The shard is a standalone weight whose sections are that device's head-local
// halves of the parent's sections (12 of 24 query/gate heads, 2 of 4 key/value heads), in the
// same query|key|gate|value order. Both share K, so they share every schedule and token cutoff.
using Nvfp4AttnInputParent = Nvfp4AttnInputProblem<Nvfp4N14336K5120, 6144, 1024>;
using Nvfp4AttnInputShard  = Nvfp4AttnInputProblem<Nvfp4N7168K5120, 3072, 512>;

static_assert(std::is_same_v<Nvfp4AttnInputParent::Output, Nvfp4AttentionInputOutput>);
static_assert(Nvfp4AttnInputShard::Geometry::kInputRows ==
              Nvfp4AttnInputParent::Geometry::kInputRows);

// Calls `body.template operator()<Problem>()` for the registered problem with `parent_rows` rows
// (`weight.n`).
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
