#pragma once
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
template <std::int32_t OutputRows, std::int32_t InputRows>
struct Nvfp4Geometry {
    static_assert(OutputRows > 0 && InputRows > 0);
    static_assert((OutputRows % 128) == 0);
    static_assert((InputRows % 64) == 0);

    static constexpr std::int32_t kOutputRows       = OutputRows;
    static constexpr std::int32_t kInputRows        = InputRows;
    static constexpr std::int32_t kGroupsPerRow     = InputRows / 16;
    static constexpr std::int32_t kScaleTilesPerRow = InputRows / 64;
    static constexpr std::int32_t kCodeBytesPerRow  = InputRows / 2;
};

template <std::int32_t InputRows>
struct Nvfp4ActivationGeometry {
    static_assert(InputRows > 0);
    static_assert((InputRows % 64) == 0);

    static constexpr std::int32_t kInputRows       = InputRows;
    static constexpr std::int32_t kGroupsPerRow    = InputRows / 16;
    static constexpr std::int32_t kCodeBytesPerRow = InputRows / 2;
};

using Nvfp4N14336K5120 = Nvfp4Geometry<14336, 5120>;
using Nvfp4N16384K5120 = Nvfp4Geometry<16384, 5120>;
using Nvfp4N34816K5120 = Nvfp4Geometry<34816, 5120>;
using Nvfp4N5120K6144  = Nvfp4Geometry<5120, 6144>;
using Nvfp4N5120K17408 = Nvfp4Geometry<5120, 17408>;
using Nvfp4N17408K5120 = Nvfp4Geometry<17408, 5120>;
using Nvfp4N5120K8704  = Nvfp4Geometry<5120, 8704>;
using Nvfp4N5120K3072  = Nvfp4Geometry<5120, 3072>;
// Two-device head-local shards of the fused attention [14336,5120] and GDN [16384,5120] input
// parents. They are problems of those projections' own kernels only, not Linear registrations.
using Nvfp4N7168K5120 = Nvfp4Geometry<7168, 5120>;
using Nvfp4N8192K5120 = Nvfp4Geometry<8192, 5120>;

// First token count at which nvfp4 linear_add over the attention output ([5120,6144] and its
// two-device half [5120,3072]) and over the MLP down projection ([5120,17408] and [5120,8704])
// takes the A4 route (nvfp4_linear_add_plan.cpp: 17 and 8 since upstream fc3993d8, 7 and 8
// before). linear() over each half uses the same crossover, so rank 0's linear_add and rank 1's
// linear() in one row-parallel pair always take the same route.
inline constexpr std::int32_t kNvfp4OutputFamilyFirstA4Tokens = 17;
inline constexpr std::int32_t kNvfp4DownFamilyFirstA4Tokens   = 8;
// First token count at which fused NVFP4 linear_swiglu over [34816,5120] and its two-device half
// [17408,5120] leaves the A16 small-T route for A4 (nvfp4_linear_swiglu_plan.cpp).
inline constexpr std::int32_t kNvfp4GateUpFirstA4Tokens = 5;

// EXPERIMENT KNOB (branch tp2/a4-error-probe, not a product setting). NINFER_TP2_A4_HALF lowers
// the A16->A4 floor of the three two-device halves above, one by one, so that one binary runs the
// per-op A4 ablation at tp 2:
//   swiglu  linear_swiglu over [17408,5120]            (kNvfp4GateUpFirstA4Tokens)
//   down    linear_add/linear over [5120,8704]         (kNvfp4DownFamilyFirstA4Tokens)
//   out     linear_add/linear over [5120,3072]         (kNvfp4OutputFamilyFirstA4Tokens)
// e.g. NINFER_TP2_A4_HALF=swiglu (A4 from T=3) or NINFER_TP2_A4_HALF=swiglu=3,down=4. A floor may
// only be lowered, to 2 at least. Unset or empty, every half keeps its constant, so routes,
// workspaces and results are those of the constants; the parents never read it. Both ranks of a
// row-parallel pair read the same floor. The environment is read once, at the first decision.
enum class Nvfp4HalfFloor : std::uint8_t { GateUp, Down, Output };

struct Nvfp4HalfFloorOverride {
    std::int32_t gate_up = 0; // 0: the constant
    std::int32_t down    = 0;
    std::int32_t output  = 0;
};

// Parses a NINFER_TP2_A4_HALF value; throws std::invalid_argument on an unknown name or a floor
// outside [2, constant].
Nvfp4HalfFloorOverride parse_nvfp4_half_floor_override(const char* spec);

// The floor of `which`: the override if NINFER_TP2_A4_HALF names it, else its constant.
std::int32_t nvfp4_half_first_a4_tokens(Nvfp4HalfFloor which);

using Nvfp4Activation3072Geometry  = Nvfp4ActivationGeometry<3072>;
using Nvfp4Activation5120Geometry  = Nvfp4ActivationGeometry<5120>;
using Nvfp4Activation6144Geometry  = Nvfp4ActivationGeometry<6144>;
using Nvfp4Activation8704Geometry  = Nvfp4ActivationGeometry<8704>;
using Nvfp4Activation17408Geometry = Nvfp4ActivationGeometry<17408>;

enum class Nvfp4GeometryId : std::uint8_t {
    N14336K5120,
    N16384K5120,
    N34816K5120,
    N5120K6144,
    N5120K17408,
    // Two-device shards: the output-row half of the MLP gate/up projection and the input-column
    // halves of the MLP down projection and of the attention and GDN output projections.
    N17408K5120,
    N5120K8704,
    N5120K3072,
};

inline Nvfp4GeometryId resolve_nvfp4_geometry(std::int32_t output_rows, std::int32_t input_rows) {
    if (output_rows == Nvfp4N14336K5120::kOutputRows &&
        input_rows == Nvfp4N14336K5120::kInputRows) {
        return Nvfp4GeometryId::N14336K5120;
    }
    if (output_rows == Nvfp4N16384K5120::kOutputRows &&
        input_rows == Nvfp4N16384K5120::kInputRows) {
        return Nvfp4GeometryId::N16384K5120;
    }
    if (output_rows == Nvfp4N34816K5120::kOutputRows &&
        input_rows == Nvfp4N34816K5120::kInputRows) {
        return Nvfp4GeometryId::N34816K5120;
    }
    if (output_rows == Nvfp4N5120K6144::kOutputRows && input_rows == Nvfp4N5120K6144::kInputRows) {
        return Nvfp4GeometryId::N5120K6144;
    }
    if (output_rows == Nvfp4N5120K17408::kOutputRows &&
        input_rows == Nvfp4N5120K17408::kInputRows) {
        return Nvfp4GeometryId::N5120K17408;
    }
    if (output_rows == Nvfp4N17408K5120::kOutputRows &&
        input_rows == Nvfp4N17408K5120::kInputRows) {
        return Nvfp4GeometryId::N17408K5120;
    }
    if (output_rows == Nvfp4N5120K8704::kOutputRows && input_rows == Nvfp4N5120K8704::kInputRows) {
        return Nvfp4GeometryId::N5120K8704;
    }
    if (output_rows == Nvfp4N5120K3072::kOutputRows && input_rows == Nvfp4N5120K3072::kInputRows) {
        return Nvfp4GeometryId::N5120K3072;
    }
    throw std::invalid_argument("unsupported NVFP4 problem");
}

} // namespace ninfer::ops::detail
