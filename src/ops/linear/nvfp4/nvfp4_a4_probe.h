#pragma once

// EXPERIMENT PROBE (branch tp2/a4-error-probe; not a product feature).
//
// With NINFER_A4_PROBE=/path/out.jsonl the tp 2 split wrappers of the NVFP4 projections whose
// A16->A4 floor is under study run, before issuing their own work and on the same input and
// stream, BOTH activation routes into private buffers and append one JSON line per call:
//   swiglu  linear_swiglu_column_parallel over [17408,5120]
//   down    linear_add_row_parallel over [5120,8704]   (rank 0 linear_add, rank 1 linear)
//   out     linear_add_row_parallel over [5120,3072]   (attention and GDN output)
// The line carries the relative RMS and maximum absolute difference A4 - A16 per rank and for the
// pair (the concatenation for swiglu, the sum of the partials for the row-parallel pairs), the
// difference relative to the residual stream it is added to, and statistics of the input and of
// its NVFP4 quantization (quantized on the device by the product's own quantizer). For down it
// also carries swiglu's A4 error carried through the A16 down projection of the same layer. The
// attention and GDN input projections, A4 from T=4 upstream, log input statistics only.
//
//   NINFER_A4_PROBE_T=4        probed widths, "T" or "T0-T1" (default 4, at most 64)
//   NINFER_A4_PROBE_MAX=N      stop after N records (default: no limit)
//
// Only calls whose policy allows A4 are probed. The probe synchronizes both ranks' streams, so it
// needs eager execution (--no-cuda-graph); it skips a call issued under stream capture. Unset,
// enabled() is false and each wrapper makes one extra test of a cached flag: no kernel, buffer or
// synchronization is added, and results are those of the code without the probe.

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail::a4probe {

enum class Op : std::uint8_t { Swiglu, Down, Out, AttnIn, GdnIn };

bool enabled();

// Probes run BEFORE the wrapper issues its ranks; they return whether a record is pending, in
// which case the wrapper calls the matching finish_* after issuing its ranks (and, for the
// row-parallel pair, before its all-reduce), which compares the product's output with both routes.
bool begin_swiglu_impl(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                       LinearPolicy policy, const ExecutionContext& ec);
void finish_swiglu(const std::array<Tensor, 2>& out, const ExecutionContext& ec);
bool begin_residual_impl(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                         const std::array<Tensor, 2>& residual, LinearPolicy policy,
                         const ExecutionContext& ec);
void finish_residual(const std::array<Tensor, 2>& partial, const ExecutionContext& ec);
void input_only_impl(Op op, const Tensor& x, const Weight& w, LinearPolicy policy,
                     const ExecutionContext& ec);

inline bool begin_swiglu(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                         LinearPolicy policy, const ExecutionContext& ec) {
    return enabled() && begin_swiglu_impl(x, w, policy, ec);
}
inline bool begin_residual(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                           const std::array<Tensor, 2>& residual, LinearPolicy policy,
                           const ExecutionContext& ec) {
    return enabled() && begin_residual_impl(x, w, residual, policy, ec);
}
inline void input_only(Op op, const Tensor& x, const Weight& w, LinearPolicy policy,
                       const ExecutionContext& ec) {
    if (enabled()) { input_only_impl(op, x, w, policy, ec); }
}

// ---- Host helpers, exposed for tests ----------------------------------------------------------

float decode_e2m1(std::uint8_t nibble) noexcept; // low 4 bits: sign, 2 exponent, 1 mantissa
float decode_e4m3(std::uint8_t byte) noexcept;   // E4M3FN, finite codes
float bf16_to_float(std::uint16_t bits) noexcept;

// Dequantizes an NVFP4 A4 activation plane in the row-major layout the quantizer writes below
// T=1024: codes [T][K/2] (element 2j in the low nibble of byte j), scales [T][K/16]. x = code *
// scale / input_scale_divisor.
void dequantize_a4(const std::uint8_t* codes, const std::uint8_t* scales, std::int32_t k,
                   std::int32_t tokens, float input_scale_divisor, float* out);

struct DiffStats {
    double err2     = 0.0; // sum (test - ref)^2
    double ref2     = 0.0; // sum ref^2
    double max_abs  = 0.0; // max |test - ref|
    double ref_amax = 0.0; // max |ref|
    std::size_t count = 0;
    [[nodiscard]] double rel() const noexcept;
    [[nodiscard]] double ref_rms() const noexcept;
    void add(const DiffStats& other) noexcept;
};
DiffStats diff_stats(const float* test, const float* ref, std::size_t count) noexcept;

struct InputStats {
    double amax = 0.0, rms = 0.0;
    double block_amax_median = 0.0, block_amax_p99 = 0.0; // over the 16-value blocks along K
    double calibrated_amax = 0.0;                          // 6 * 448 / input_scale_divisor
    double saturated_blocks = 0.0; // fraction whose scale clips at 448 (block amax > calibrated)
    double subnormal_blocks = 0.0; // fraction with an E4M3-subnormal nonzero scale (< 2^-6)
    double zero_blocks      = 0.0; // fraction of nonzero blocks whose scale rounds to 0
    double quant_rel        = 0.0; // ||dequant - x|| / ||x||
    double quant_zeroed     = 0.0; // fraction of nonzero inputs that dequantize to 0
};
// x: [T][K] floats (the BF16 input); scales: [T][K/16] as above; dequantized: [T][K].
InputStats input_stats(const float* x, const float* dequantized, const std::uint8_t* scales,
                       std::int32_t k, std::int32_t tokens, float input_scale_divisor);

} // namespace ninfer::ops::detail::a4probe
