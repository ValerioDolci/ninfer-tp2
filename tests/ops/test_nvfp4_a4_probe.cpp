// Host helpers of the NINFER_A4_PROBE experiment and the NINFER_TP2_A4_HALF parser (CPU), and the
// probe's reading of the product's A4 activation quantizer against an independent host model of
// the format (GPU, skipped without a device).

#include "core/arena.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include "ops/linear/nvfp4/nvfp4_a4_probe.h"
#include "ops/linear/nvfp4/nvfp4_geometry.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
namespace probe = ninfer::ops::detail::a4probe;
namespace detail = ninfer::ops::detail;

namespace {

int failures = 0;

void expect(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

bool throws(const char* spec) {
    try {
        (void)detail::parse_nvfp4_half_floor_override(spec);
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

void check_parser() {
    auto o = detail::parse_nvfp4_half_floor_override("");
    expect(o.gate_up == 0 && o.down == 0 && o.output == 0, "empty spec overrides nothing");
    o = detail::parse_nvfp4_half_floor_override("swiglu");
    expect(o.gate_up == 3 && o.down == 0 && o.output == 0, "bare name means T=3");
    o = detail::parse_nvfp4_half_floor_override(" swiglu=4 , down,out=17");
    expect(o.gate_up == 4 && o.down == 3 && o.output == 17, "explicit floors and spaces");
    expect(throws("mlp"), "unknown name rejected");
    expect(throws("swiglu=6"), "a floor above the constant is rejected");
    expect(throws("down=1"), "a floor under 2 is rejected");
    expect(throws("out=x"), "a non-number is rejected");
    // The test process does not set NINFER_TP2_A4_HALF: every half keeps its constant.
    expect(detail::nvfp4_half_first_a4_tokens(detail::Nvfp4HalfFloor::GateUp) ==
               detail::kNvfp4GateUpFirstA4Tokens,
           "gate/up floor defaults to its constant");
    expect(detail::nvfp4_half_first_a4_tokens(detail::Nvfp4HalfFloor::Down) ==
               detail::kNvfp4DownFamilyFirstA4Tokens,
           "down floor defaults to its constant");
    expect(detail::nvfp4_half_first_a4_tokens(detail::Nvfp4HalfFloor::Output) ==
               detail::kNvfp4OutputFamilyFirstA4Tokens,
           "output floor defaults to its constant");
}

void check_decoders() {
    const float e2m1[16] = {0, 0.5F, 1, 1.5F, 2, 3, 4, 6, -0.0F, -0.5F, -1, -1.5F, -2, -3, -4, -6};
    for (int i = 0; i < 16; ++i)
        expect(probe::decode_e2m1(static_cast<std::uint8_t>(i)) == e2m1[i], "E2M1 code " +
                                                                               std::to_string(i));
    expect(probe::decode_e4m3(0x38) == 1.0F, "E4M3 1.0");
    expect(probe::decode_e4m3(0x7E) == 448.0F, "E4M3 max 448");
    expect(probe::decode_e4m3(0x01) == std::ldexp(1.0F, -9), "E4M3 smallest subnormal");
    expect(probe::decode_e4m3(0x08) == std::ldexp(1.0F, -6), "E4M3 smallest normal");
    expect(probe::decode_e4m3(0xB8) == -1.0F, "E4M3 sign");
    expect(probe::bf16_to_float(0x3F80) == 1.0F, "BF16 1.0");
}

void check_stats() {
    const std::vector<float> ref{1, -2, 3, -4};
    const std::vector<float> same = ref;
    const std::vector<float> twice{2, -4, 6, -8};
    expect(probe::diff_stats(same.data(), ref.data(), 4).rel() == 0.0, "identical: rel 0");
    const auto d = probe::diff_stats(twice.data(), ref.data(), 4);
    expect(std::abs(d.rel() - 1.0) < 1e-12 && d.max_abs == 4.0 && d.ref_amax == 4.0,
           "doubled: rel 1, max_abs 4");

    // One token, one block of 16: codes {6, 1, 0.5, 0, ...}, scale 2.0 (0x40), divisor 4.
    std::vector<std::uint8_t> codes(8, 0), scales{0x40};
    codes[0] = 0x7 | (0x2 << 4); // elements 0 (6) and 1 (1.0)
    codes[1] = 0x1;              // element 2 (0.5)
    std::vector<float> dequantized(16);
    probe::dequantize_a4(codes.data(), scales.data(), 16, 1, 4.0F, dequantized.data());
    expect(dequantized[0] == 3.0F && dequantized[1] == 0.5F && dequantized[2] == 0.25F &&
               dequantized[3] == 0.0F,
           "dequantize: nibble order and code * scale / divisor");
    std::vector<float> x(16, 0.0F);
    x[0] = 3.0F;
    x[1] = 0.5F;
    x[2] = 0.25F;
    x[3] = 0.01F; // lost: quantizes to 0
    const auto in = probe::input_stats(x.data(), dequantized.data(), scales.data(), 16, 1, 4.0F);
    expect(in.amax == 3.0 && std::abs(in.calibrated_amax - 672.0) < 1e-9, "input amax, cal");
    expect(std::abs(in.quant_zeroed - 0.25) < 1e-12, "one of four nonzero inputs zeroed");
    expect(in.saturated_blocks == 0.0 && in.zero_blocks == 0.0 && in.subnormal_blocks == 0.0,
           "no saturated, zero or subnormal scale");
}

// Independent host model of quantize_nvfp4_k16 (nvfp4_codec.cuh): per 16 values along K,
// scale = E4M3_rn_satfinite(d * amax / 6), code = E2M1_rn_satfinite(x * d / scale),
// value = code * scale / d.
float round_to_grid(float v, int min_exponent, int mantissa_bits, float max) {
    if (!(v > 0.0F)) return 0.0F;
    if (v >= max) return max;
    int e = 0;
    (void)std::frexp(v, &e);
    const float quantum = std::ldexp(1.0F, std::max(e - 1, min_exponent) - mantissa_bits);
    return std::min(std::nearbyint(v / quantum) * quantum, max);
}

void model_block(const float* x, float d, float* out) {
    float amax = 0.0F;
    for (int i = 0; i < 16; ++i) amax = std::max(amax, std::abs(x[i]));
    const float scale = round_to_grid((d * amax) / 6.0F, -6, 3, 448.0F);
    for (int i = 0; i < 16; ++i) {
        if (scale == 0.0F) {
            out[i] = 0.0F;
            continue;
        }
        const float code = round_to_grid(std::abs((x[i] * d) / scale), 0, 1, 6.0F);
        out[i]           = static_cast<float>(std::copysign(code, x[i]) * (double(scale) / d));
    }
}

void check_device_quantizer() {
    constexpr std::int32_t k = 8704, tokens = 4;
    constexpr float calibrated = 71.0F;
    const float divisor        = 6.0F * 448.0F / calibrated;
    std::mt19937 rng(20260928U);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> x(static_cast<std::size_t>(k) * tokens);
    for (std::size_t i = 0; i < x.size(); ++i) {
        float v = normal(rng) * 0.3F;
        if (i % 97 == 0) v *= 200.0F;       // outliers, some past the calibrated amax
        if (i % 16 == 5 && i % 3 == 0) v = 1e-7F; // tiny values
        if (i >= 32 && i < 48) v *= 1e-6F;  // a block whose scale underflows
        x[i] = v;
    }
    round_to_bf16(x);
    DeviceBuffer device_x = to_device_bf16(x);
    DeviceArena arena(detail::nvfp4_a4_workspace_capacity_bytes(tokens, k) + 4096);
    const detail::Nvfp4A4Workspace q = detail::allocate_nvfp4_a4_workspace(arena, tokens, k);
    Weight w;
    w.qtype               = QType::NVFP4;
    w.n                   = 5120;
    w.k                   = k;
    w.input_scale_divisor = divisor;
    const Tensor tx(device_x.p, DType::BF16, {k, tokens});
    detail::launch_nvfp4_a4_quantize(tx, w, q, detail::Nvfp4ScaleLayout::RowMajor, nullptr);
    cuda_synchronize();
    const auto codes  = from_device<std::uint8_t>(q.codes, static_cast<std::size_t>(k / 2) * tokens);
    const auto scales = from_device<std::uint8_t>(q.scales, static_cast<std::size_t>(k / 16) * tokens);
    std::vector<float> got(x.size()), want(x.size());
    probe::dequantize_a4(codes.data(), scales.data(), k, tokens, divisor, got.data());
    std::fesetround(FE_TONEAREST);
    for (std::size_t b = 0; b < x.size() / 16; ++b) model_block(&x[b * 16], divisor, &want[b * 16]);
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < x.size(); ++i) mismatches += got[i] != want[i];
    const auto in = probe::input_stats(x.data(), got.data(), scales.data(), k, tokens, divisor);
    std::cout << "device quantizer vs host model: " << mismatches << " of " << x.size()
              << " values differ; qrel " << in.quant_rel << ", saturated blocks "
              << in.saturated_blocks << ", zero-scale blocks " << in.zero_blocks
              << ", zeroed inputs " << in.quant_zeroed << '\n';
    expect(mismatches == 0, "device quantizer matches the host model of the format");
    expect(in.saturated_blocks > 0.0 && in.zero_blocks > 0.0, "fixture reaches both scale edges");
}

} // namespace

int main() {
    try {
        check_parser();
        check_decoders();
        check_stats();
        if (cuda_unavailable()) {
            std::cout << "no usable CUDA device: device quantizer check skipped\n";
        } else {
            check_device_quantizer();
        }
    } catch (const std::exception& error) {
        std::cerr << "nvfp4 A4 probe test: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " nvfp4 A4 probe helpers\n";
    return failures == 0 ? 0 : 1;
}
