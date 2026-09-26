#pragma once

#include "ops/linear/common/output.cuh"
#include "ops/linear/fp8/fp8_geometry.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kFp8AttnInputQueryRows = 6144;
inline constexpr std::int32_t kFp8AttnInputKeyRows   = 1024;
inline constexpr std::int32_t kFp8AttnInputGateRows  = 6144;
inline constexpr std::int32_t kFp8AttnInputKeyBegin  = kFp8AttnInputQueryRows;
inline constexpr std::int32_t kFp8AttnInputGateBegin = kFp8AttnInputKeyBegin + kFp8AttnInputKeyRows;
inline constexpr std::int32_t kFp8AttnInputValueBegin =
    kFp8AttnInputGateBegin + kFp8AttnInputGateRows;

static_assert((kFp8AttnInputQueryRows % 8) == 0);
static_assert((kFp8AttnInputKeyRows % 8) == 0);
static_assert((kFp8AttnInputGateRows % 8) == 0);

using Fp8AttentionInputOutput = LinearBf16SegmentedOutput<6144, 1024, 6144, 1024>;

// A registered fused parent: its FP8 geometry and its query|key|gate|value section output. Gate
// has the query's row count and value has the key's. Every section is a multiple of 8 rows, so
// vector stores never straddle two sections; a row tile that divides every section binds to one
// section (LinearBf16SegmentedOutput::bind_tile).
template <class GeometryType, std::int32_t QueryRows, std::int32_t KeyRows>
struct Fp8AttnInputProblem {
    using Geometry = GeometryType;
    using Output   = LinearBf16SegmentedOutput<QueryRows, KeyRows, QueryRows, KeyRows>;

    static constexpr std::int32_t kQueryRows = QueryRows;
    static constexpr std::int32_t kKeyRows   = KeyRows;

    static_assert(Geometry::kOutputRows == 2 * (QueryRows + KeyRows));
    static_assert((QueryRows % 8) == 0 && (KeyRows % 8) == 0);
};

// The whole [14336,5120] parent, and one device's [7168,5120] shard under two-device tensor
// parallelism. The shard is a standalone weight whose sections are that device's head-local
// halves of the parent's sections (12 of 24 query/gate heads, 2 of 4 key/value heads), in the
// same query|key|gate|value order. Both share K, so they share every schedule and token cutoff.
using Fp8AttnInputParent =
    Fp8AttnInputProblem<Fp8N14336K5120, kFp8AttnInputQueryRows, kFp8AttnInputKeyRows>;
using Fp8AttnInputShard = Fp8AttnInputProblem<Fp8N7168K5120, 3072, 512>;

static_assert(std::is_same_v<Fp8AttnInputParent::Output, Fp8AttentionInputOutput>);
static_assert(Fp8AttnInputShard::Geometry::kInputRows == Fp8AttnInputParent::Geometry::kInputRows);

// Calls `body.template operator()<Problem>()` for the registered problem with `parent_rows` rows
// (`weight.n`).
template <class Body>
void visit_fp8_attn_input_problem(std::int32_t parent_rows, Body&& body) {
    if (parent_rows == Fp8AttnInputParent::Geometry::kOutputRows) {
        body.template operator()<Fp8AttnInputParent>();
        return;
    }
    if (parent_rows == Fp8AttnInputShard::Geometry::kOutputRows) {
        body.template operator()<Fp8AttnInputShard>();
        return;
    }
    throw std::invalid_argument("fp8 attn_input_proj: unsupported parent rows");
}

} // namespace ninfer::ops::detail
