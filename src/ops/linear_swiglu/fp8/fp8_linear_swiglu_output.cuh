#pragma once
#include "ops/common/math.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/fp8/fp8_geometry.h"
#include "ops/linear_swiglu/token_major_mma_epilogue.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
template <int RowsPerBranch, int IntermediateRows>
struct Fp8SwiGluRows {
    static_assert(RowsPerBranch > 0 && (RowsPerBranch & (RowsPerBranch - 1)) == 0);
    static constexpr bool kPaired = true;

    __device__ __forceinline__ int weight_row(int begin, int row, int) const {
        return begin + row % RowsPerBranch + (row >= RowsPerBranch ? IntermediateRows : 0);
    }
};

struct Fp8SwiGluEpilogue {
    template <class Output>
    __device__ __forceinline__ void apply_pair(Output output, int row, int token, float gate,
                                               float up) const {
        output.store(row, token, silu(gate) * up);
    }
};

// Fp8SwiGluRows names the intermediate width M = N/2 at compile time: gate rows [0,M) precede
// their up rows [M,2M). The FP8 gate/up problems are the single-device [34816,5120] and its
// two-device output-row half [17408,5120]; this calls `body.template operator()<M>()` with the
// width of the one `gate_up_rows` names.
template <class Body>
void visit_fp8_swiglu_intermediate_rows(std::int32_t gate_up_rows, Body&& body) {
    if (gate_up_rows == Fp8N34816K5120::kOutputRows) {
        return body.template operator()<Fp8N34816K5120::kOutputRows / 2>();
    }
    if (gate_up_rows == Fp8N17408K5120::kOutputRows) {
        return body.template operator()<Fp8N17408K5120::kOutputRows / 2>();
    }
    throw std::invalid_argument("fp8 linear_swiglu: unsupported gate/up problem");
}
} // namespace ninfer::ops::detail
