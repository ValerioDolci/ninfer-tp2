#include "ops/linear/nvfp4/nvfp4_a4_probe.h"

#include "core/arena.h"
#include "core/device_scope.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include "ops/linear/nvfp4/nvfp4_geometry.h"
#include "ops/linear/nvfp4/nvfp4_layout.h"
#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdarg>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ninfer::ops::detail::a4probe {
namespace {

constexpr std::int32_t kMaxProbeTokens = 64;
constexpr std::int32_t kSwigluRows     = Nvfp4N17408K5120::kOutputRows / 2; // 8704 per rank
constexpr std::int32_t kHidden         = Nvfp4N5120K8704::kOutputRows;      // 5120
constexpr std::size_t kSwigluBytes =
    static_cast<std::size_t>(kSwigluRows) * kMaxProbeTokens * sizeof(std::uint16_t);
constexpr std::size_t kRegionBytes = 32U << 20;

const char* op_name(Op op) {
    switch (op) {
    case Op::Swiglu:
        return "swiglu";
    case Op::Down:
        return "down";
    case Op::Out:
        return "out";
    case Op::AttnIn:
        return "attn_in";
    case Op::GdnIn:
        return "gdn_in";
    }
    return "?";
}

struct Config {
    bool on = false;
    std::string path;
    std::int32_t t_min        = 4;
    std::int32_t t_max        = 4;
    long long max_records     = -1;
};

std::int32_t parse_width(const std::string& text) {
    char* end         = nullptr;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if (text.empty() || end == nullptr || *end != '\0' || parsed < 1 || parsed > kMaxProbeTokens) {
        throw std::invalid_argument("NINFER_A4_PROBE_T: widths must be integers in [1, 64]");
    }
    return static_cast<std::int32_t>(parsed);
}

Config read_config() {
    Config c;
    const char* path = std::getenv("NINFER_A4_PROBE");
    if (path == nullptr || *path == '\0') { return c; }
    c.on   = true;
    c.path = path;
    if (const char* widths = std::getenv("NINFER_A4_PROBE_T"); widths != nullptr && *widths) {
        const std::string text(widths);
        const std::size_t dash = text.find('-');
        c.t_min = parse_width(text.substr(0, dash));
        c.t_max = dash == std::string::npos ? c.t_min : parse_width(text.substr(dash + 1));
        if (c.t_max < c.t_min) { throw std::invalid_argument("NINFER_A4_PROBE_T: empty range"); }
    }
    if (const char* limit = std::getenv("NINFER_A4_PROBE_MAX"); limit != nullptr && *limit) {
        c.max_records = std::atoll(limit);
    }
    std::fprintf(stderr,
                 "ninfer: NINFER_A4_PROBE=%s (experiment): A4 vs A16 on T=%d..%d, needs "
                 "--no-cuda-graph\n",
                 c.path.c_str(), c.t_min, c.t_max);
    return c;
}

const Config& config() {
    static const Config c = read_config();
    return c;
}

struct RankBuffers {
    std::uint8_t* base = nullptr;
    std::uint16_t* s16 = nullptr; // this rank's swiglu outputs, kept for the down probe
    std::uint16_t* s4  = nullptr;
    DeviceSpan scratch{};         // carved per call
};

struct PendingSwiglu {
    bool valid          = false;
    int layer           = -1;
    std::int32_t tokens = 0;
    std::array<std::vector<std::uint16_t>, 2> a16, a4; // host copies of s16/s4
};

struct PendingRecord {
    bool active = false;
    std::string head; // JSON members before "prod"
    std::string tail; // JSON members after "prod"
    std::array<std::vector<std::uint16_t>, 2> a16, a4;
    std::array<bool, 2> compare{false, false};
};

struct State {
    std::mutex mutex;
    std::FILE* file       = nullptr;
    long long records     = 0;
    long long sequence    = 0;
    bool warned_capture   = false;
    std::array<RankBuffers, 2> ranks{};
    std::array<std::unordered_map<const void*, int>, 5> layers;
    PendingSwiglu swiglu;
    PendingRecord pending;
};

State& state() {
    static State s;
    return s;
}

int layer_index(State& s, Op op, const void* key) {
    auto& map         = s.layers[static_cast<std::size_t>(op)];
    return map.emplace(key, static_cast<int>(map.size())).first->second;
}

bool budget_left(const State& s) {
    const long long limit = config().max_records;
    return limit < 0 || s.records < limit;
}

bool capturing(const ExecutionContext& ec, State& s) {
    for (int rank = 0; rank < 2; ++rank) {
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(ec.dev[rank]->stream, &status));
        if (status != cudaStreamCaptureStatusNone) {
            if (!s.warned_capture) {
                std::fprintf(stderr, "ninfer: NINFER_A4_PROBE skips calls under stream capture; "
                                     "run with --no-cuda-graph\n");
                s.warned_capture = true;
            }
            return true;
        }
    }
    return false;
}

RankBuffers& buffers(State& s, int rank) {
    RankBuffers& b = s.ranks[static_cast<std::size_t>(rank)];
    if (b.base == nullptr) {
        void* base = nullptr;
        CUDA_CHECK(cudaMalloc(&base, kRegionBytes)); // current device = this rank's
        b.base    = static_cast<std::uint8_t*>(base);
        b.s16     = reinterpret_cast<std::uint16_t*>(b.base);
        b.s4      = reinterpret_cast<std::uint16_t*>(b.base + kSwigluBytes);
        b.scratch = {b.base + 2 * kSwigluBytes, kRegionBytes - 2 * kSwigluBytes};
    }
    return b;
}

template <class T>
void to_host(std::vector<T>& host, const void* device, std::size_t count, cudaStream_t stream) {
    host.resize(count);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost,
                               stream));
}

std::vector<float> widen(const std::vector<std::uint16_t>& bits) {
    std::vector<float> out(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) out[i] = bf16_to_float(bits[i]);
    return out;
}

std::uint16_t* carve_bf16(DeviceArena& arena, std::size_t elements) {
    return static_cast<std::uint16_t*>(arena.alloc_bytes(elements * sizeof(std::uint16_t)).data);
}

// Quantizes `x` [K,T] with the product's quantizer into `q` (row-major) and brings back the
// input, its codes and scales.
struct HostInput {
    std::vector<std::uint16_t> x;
    std::vector<std::uint8_t> codes, scales;
};
void quantize_to_host(const Tensor& x, const Weight& w, Nvfp4A4Workspace q, HostInput& host,
                      cudaStream_t stream) {
    const std::int32_t k      = x.ne[0];
    const std::int32_t tokens = x.ne[1];
    launch_nvfp4_a4_quantize(x, w, q, Nvfp4ScaleLayout::RowMajor, stream);
    to_host(host.x, x.data, static_cast<std::size_t>(k) * tokens, stream);
    to_host(host.codes, q.codes, static_cast<std::size_t>(k / 2) * tokens, stream);
    to_host(host.scales, q.scales, static_cast<std::size_t>(k / 16) * tokens, stream);
}

InputStats host_input_stats(const HostInput& host, std::int32_t k, std::int32_t tokens,
                            float divisor) {
    const std::vector<float> x = widen(host.x);
    std::vector<float> dequantized(x.size());
    dequantize_a4(host.codes.data(), host.scales.data(), k, tokens, divisor, dequantized.data());
    return input_stats(x.data(), dequantized.data(), host.scales.data(), k, tokens, divisor);
}

void append(std::string& out, const char* format, ...) __attribute__((format(printf, 2, 3)));
void append(std::string& out, const char* format, ...) {
    char buffer[1024];
    va_list args;
    va_start(args, format);
    const int n = std::vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    if (n > 0) {
        out.append(buffer, static_cast<std::size_t>(
                               std::min(n, static_cast<int>(sizeof(buffer)) - 1)));
    }
}

std::string input_json(const InputStats& in) {
    std::string s;
    append(s,
           "{\"amax\":%.6g,\"rms\":%.6g,\"blk_med\":%.6g,\"blk_p99\":%.6g,\"cal\":%.6g,"
           "\"sat\":%.6g,\"sub\":%.6g,\"zero\":%.6g,\"qrel\":%.6g,\"qzero\":%.6g}",
           in.amax, in.rms, in.block_amax_median, in.block_amax_p99, in.calibrated_amax,
           in.saturated_blocks, in.subnormal_blocks, in.zero_blocks, in.quant_rel,
           in.quant_zeroed);
    return s;
}

std::string diff_json(const DiffStats& d) {
    std::string s;
    append(s, "{\"rel\":%.6g,\"max_abs\":%.6g,\"ref_rms\":%.6g,\"ref_amax\":%.6g}", d.rel(),
           d.max_abs, d.ref_rms(), d.ref_amax);
    return s;
}

// Relative error of the worst token column; `rows` values per column.
double worst_column(const std::vector<float>& test, const std::vector<float>& ref,
                    std::int32_t rows, std::int32_t tokens) {
    double worst = 0.0;
    for (std::int32_t t = 0; t < tokens; ++t) {
        const DiffStats d = diff_stats(test.data() + static_cast<std::size_t>(t) * rows,
                                       ref.data() + static_cast<std::size_t>(t) * rows,
                                       static_cast<std::size_t>(rows));
        worst = std::max(worst, d.rel());
    }
    return worst;
}

std::string record_head(State& s, Op op, int layer, std::int32_t tokens, std::int32_t n,
                        std::int32_t k) {
    const double now =
        std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
            .count();
    std::string head;
    append(head, "{\"seq\":%lld,\"ts\":%.3f,\"op\":\"%s\",\"layer\":%d,\"T\":%d,\"n\":%d,\"k\":%d",
           s.sequence++, now, op_name(op), layer, tokens, n, k);
    return head;
}

void write_line(State& s, const std::string& line) {
    if (s.file == nullptr) {
        s.file = std::fopen(config().path.c_str(), "a");
        if (s.file == nullptr) {
            throw std::runtime_error("NINFER_A4_PROBE: cannot open " + config().path);
        }
    }
    std::fputs(line.c_str(), s.file);
    std::fputc('\n', s.file);
    std::fflush(s.file);
    ++s.records;
}

const char* route_of(const std::vector<std::uint16_t>& product, const std::vector<std::uint16_t>& a16,
                     const std::vector<std::uint16_t>& a4) {
    const bool is_a4  = product == a4;
    const bool is_a16 = product == a16;
    if (is_a4 && is_a16) return "both";
    if (is_a4) return "a4";
    if (is_a16) return "a16";
    return "other";
}

} // namespace

bool enabled() { return config().on; }

// ---- swiglu ---------------------------------------------------------------------------------

bool begin_swiglu_impl(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                       LinearPolicy policy, const ExecutionContext& ec) {
    if (w[0].qtype != QType::NVFP4 || w[0].n != Nvfp4N17408K5120::kOutputRows ||
        w[0].k != Nvfp4N17408K5120::kInputRows) {
        return false;
    }
    State& s = state();
    const std::lock_guard<std::mutex> lock(s.mutex);
    const int layer           = layer_index(s, Op::Swiglu, w[0].qdata);
    const std::int32_t tokens = x[0].ne[1];
    s.swiglu.valid            = false;
    // The A16 reference route of the fused SwiGLU is registered through T=16 only.
    if (!allows_a4(policy) || tokens < config().t_min || tokens > config().t_max ||
        tokens > 16 || !budget_left(s) || capturing(ec, s)) {
        return false;
    }
    const std::int32_t k = w[0].k;
    std::array<std::vector<std::uint16_t>, 2> a16, a4;
    HostInput input;
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        const ScopedCurrentDevice device(ec.dev[slot]->device);
        const cudaStream_t stream = ec.dev[slot]->stream;
        RankBuffers& b            = buffers(s, rank);
        DeviceArena arena(b.scratch);
        Tensor out16(b.s16, DType::BF16, {kSwigluRows, tokens});
        Tensor out4(b.s4, DType::BF16, {kSwigluRows, tokens});
        nvfp4_linear_swiglu_dispatch(x[slot], w[slot], out16, LinearPolicy::A16Only, nullptr,
                                     stream);
        {
            auto scope = arena.scope();
            nvfp4_linear_swiglu_a4_launch(x[slot], w[slot], out4, arena, stream);
        }
        if (rank == 0) {
            quantize_to_host(x[0], w[0], allocate_nvfp4_a4_workspace(arena, tokens, k), input,
                             stream);
        }
        const std::size_t count = static_cast<std::size_t>(kSwigluRows) * tokens;
        to_host(a16[slot], b.s16, count, stream);
        to_host(a4[slot], b.s4, count, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    DiffStats pair;
    std::array<DiffStats, 2> per_rank;
    double worst = 0.0;
    for (std::size_t r = 0; r < 2; ++r) {
        const std::vector<float> ref = widen(a16[r]), test = widen(a4[r]);
        per_rank[r] = diff_stats(test.data(), ref.data(), ref.size());
        pair.add(per_rank[r]);
        worst = std::max(worst, worst_column(test, ref, kSwigluRows, tokens));
    }
    const InputStats in = host_input_stats(input, k, tokens, w[0].input_scale_divisor);

    PendingRecord& p = s.pending;
    p.active         = true;
    p.head           = record_head(s, Op::Swiglu, layer, tokens, w[0].n, k);
    p.tail.clear();
    append(p.tail, ",\"pair\":%s", diff_json(pair).c_str());
    p.tail.pop_back(); // reopen the pair object for the worst column
    append(p.tail, ",\"rel_tok_max\":%.6g}", worst);
    append(p.tail, ",\"rank\":[%s,%s]", diff_json(per_rank[0]).c_str(),
           diff_json(per_rank[1]).c_str());
    append(p.tail, ",\"in\":[%s]}", input_json(in).c_str());
    p.a16     = a16;
    p.a4      = a4;
    p.compare = {true, true};

    s.swiglu.valid  = true;
    s.swiglu.layer  = layer;
    s.swiglu.tokens = tokens;
    s.swiglu.a16    = std::move(a16);
    s.swiglu.a4     = std::move(a4);
    return true;
}

namespace {
void finish(State& s, const std::array<Tensor, 2>& out, const ExecutionContext& ec) {
    PendingRecord& p = s.pending;
    if (!p.active) return;
    std::array<const char*, 2> route{"?", "?"};
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        if (!p.compare[slot]) continue;
        const ScopedCurrentDevice device(ec.dev[slot]->device);
        const cudaStream_t stream = ec.dev[slot]->stream;
        std::vector<std::uint16_t> product;
        to_host(product, out[slot].data, p.a16[slot].size(), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        route[slot] = route_of(product, p.a16[slot], p.a4[slot]);
    }
    std::string line = p.head;
    append(line, ",\"prod\":[\"%s\",\"%s\"]", route[0], route[1]);
    line += p.tail;
    write_line(s, line);
    p = PendingRecord{};
}
} // namespace

void finish_swiglu(const std::array<Tensor, 2>& out, const ExecutionContext& ec) {
    State& s = state();
    const std::lock_guard<std::mutex> lock(s.mutex);
    finish(s, out, ec);
}

// ---- row-parallel pairs (down, out) -----------------------------------------------------------

bool begin_residual_impl(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                         const std::array<Tensor, 2>& residual, LinearPolicy policy,
                         const ExecutionContext& ec) {
    if (w[0].qtype != QType::NVFP4 || w[0].n != kHidden ||
        (w[0].k != Nvfp4N5120K8704::kInputRows && w[0].k != Nvfp4N5120K3072::kInputRows)) {
        return false;
    }
    const Op op              = w[0].k == Nvfp4N5120K8704::kInputRows ? Op::Down : Op::Out;
    const Nvfp4LinearShape& shape =
        op == Op::Down ? kNvfp4N5120K8704 : kNvfp4N5120K3072;
    State& s = state();
    const std::lock_guard<std::mutex> lock(s.mutex);
    const int layer           = layer_index(s, op, w[0].qdata);
    const std::int32_t tokens = x[0].ne[1];
    const bool propagate      = op == Op::Down && s.swiglu.valid && s.swiglu.layer == layer &&
                           s.swiglu.tokens == tokens;
    if (op == Op::Down) s.swiglu.valid = false;
    if (!allows_a4(policy) || tokens < config().t_min || tokens > config().t_max ||
        !budget_left(s) || capturing(ec, s)) {
        return false;
    }
    const std::int32_t k    = w[0].k;
    const std::size_t count = static_cast<std::size_t>(kHidden) * tokens;
    std::array<std::vector<std::uint16_t>, 2> a16, a4, p16, p4;
    std::array<HostInput, 2> input;
    std::vector<std::uint16_t> resid;
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        const ScopedCurrentDevice device(ec.dev[slot]->device);
        const cudaStream_t stream = ec.dev[slot]->stream;
        RankBuffers& b            = buffers(s, rank);
        DeviceArena arena(b.scratch);
        Tensor out16(carve_bf16(arena, count), DType::BF16, {kHidden, tokens});
        Tensor out4(carve_bf16(arena, count), DType::BF16, {kHidden, tokens});
        Tensor prop16(carve_bf16(arena, count), DType::BF16, {kHidden, tokens});
        Tensor prop4(carve_bf16(arena, count), DType::BF16, {kHidden, tokens});
        const Nvfp4A4Workspace route_ws = allocate_nvfp4_a4_workspace(arena, tokens, k);
        const Nvfp4A4Workspace quant_ws = allocate_nvfp4_a4_workspace(arena, tokens, k);
        Tensor s16(b.s16, DType::BF16, {k, tokens});
        Tensor s4(b.s4, DType::BF16, {k, tokens});
        if (rank == 0) {
            // linear_add with a zero residual leaves exactly the rounded partial.
            for (Tensor* t : {&out16, &out4, &prop16, &prop4}) {
                CUDA_CHECK(cudaMemsetAsync(t->data, 0, t->bytes(), stream));
            }
            nvfp4_linear_add_a16_launch(x[slot], w[slot], out16, stream);
            nvfp4_linear_add_a4_launch(x[slot], w[slot], out4, route_ws, stream);
            if (propagate) {
                nvfp4_linear_add_a16_launch(s16, w[slot], prop16, stream);
                nvfp4_linear_add_a16_launch(s4, w[slot], prop4, stream);
            }
            to_host(resid, residual[0].data, count, stream);
        } else {
            shape.a16(x[slot], w[slot], out16, stream);
            shape.a4(x[slot], w[slot], out4, route_ws, stream);
            if (propagate) {
                shape.a16(s16, w[slot], prop16, stream);
                shape.a16(s4, w[slot], prop4, stream);
            }
        }
        quantize_to_host(x[slot], w[slot], quant_ws, input[slot], stream);
        to_host(a16[slot], out16.data, count, stream);
        to_host(a4[slot], out4.data, count, stream);
        if (propagate) {
            to_host(p16[slot], prop16.data, count, stream);
            to_host(p4[slot], prop4.data, count, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // The pair's output is the sum of the two partials (what the all-reduce adds to the residual).
    const auto summed = [&](const std::array<std::vector<std::uint16_t>, 2>& parts) {
        std::vector<float> out(count);
        for (std::size_t i = 0; i < count; ++i)
            out[i] = bf16_to_float(parts[0][i]) + bf16_to_float(parts[1][i]);
        return out;
    };
    const std::vector<float> ref = summed(a16), test = summed(a4);
    const DiffStats pair         = diff_stats(test.data(), ref.data(), count);
    const double worst           = worst_column(test, ref, kHidden, tokens);
    std::array<DiffStats, 2> per_rank;
    for (std::size_t r = 0; r < 2; ++r) {
        const std::vector<float> r16 = widen(a16[r]), r4 = widen(a4[r]);
        per_rank[r] = diff_stats(r4.data(), r16.data(), count);
    }
    const std::vector<float> h = widen(resid);
    std::vector<float> zero(count, 0.0F);
    const DiffStats resid_stats = diff_stats(h.data(), zero.data(), count); // err2 = ||h||^2
    const double h_norm         = std::sqrt(resid_stats.err2);
    double worst_resid          = 0.0;
    for (std::int32_t t = 0; t < tokens; ++t) {
        const std::size_t o = static_cast<std::size_t>(t) * kHidden;
        const DiffStats d   = diff_stats(test.data() + o, ref.data() + o, kHidden);
        const DiffStats ht  = diff_stats(h.data() + o, zero.data() + o, kHidden);
        if (ht.err2 > 0) worst_resid = std::max(worst_resid, std::sqrt(d.err2 / ht.err2));
    }

    PendingRecord& p = s.pending;
    p.active         = true;
    p.head           = record_head(s, op, layer, tokens, w[0].n, k);
    p.tail.clear();
    append(p.tail, ",\"pair\":%s", diff_json(pair).c_str());
    p.tail.pop_back();
    append(p.tail, ",\"rel_tok_max\":%.6g}", worst);
    append(p.tail, ",\"rank\":[%s,%s]", diff_json(per_rank[0]).c_str(),
           diff_json(per_rank[1]).c_str());
    append(p.tail, ",\"resid_rms\":%.6g,\"rel_resid\":%.6g,\"rel_resid_tok_max\":%.6g",
           std::sqrt(resid_stats.err2 / static_cast<double>(count)),
           h_norm > 0 ? std::sqrt(pair.err2) / h_norm : 0.0, worst_resid);
    const InputStats in0 = host_input_stats(input[0], k, tokens, w[0].input_scale_divisor);
    const InputStats in1 = host_input_stats(input[1], k, tokens, w[1].input_scale_divisor);
    append(p.tail, ",\"in\":[%s,%s]", input_json(in0).c_str(), input_json(in1).c_str());
    if (op == Op::Down) {
        std::array<const char*, 2> source{"?", "?"};
        if (propagate) {
            for (std::size_t r = 0; r < 2; ++r)
                source[r] = route_of(input[r].x, s.swiglu.a16[r], s.swiglu.a4[r]);
        }
        append(p.tail, ",\"x_src\":[\"%s\",\"%s\"]", source[0], source[1]);
        if (propagate) {
            const std::vector<float> q16 = summed(p16), q4 = summed(p4);
            const DiffStats prop         = diff_stats(q4.data(), q16.data(), count);
            append(p.tail, ",\"prop\":{\"rel\":%.6g,\"max_abs\":%.6g,\"rel_resid\":%.6g}",
                   prop.rel(), prop.max_abs, h_norm > 0 ? std::sqrt(prop.err2) / h_norm : 0.0);
        }
    }
    p.tail += "}";
    // Rank 0's product adds its partial into the residual; rank 1's is the bare partial.
    p.a16[1]  = std::move(a16[1]);
    p.a4[1]   = std::move(a4[1]);
    p.compare = {false, true};
    return true;
}

void finish_residual(const std::array<Tensor, 2>& partial, const ExecutionContext& ec) {
    State& s = state();
    const std::lock_guard<std::mutex> lock(s.mutex);
    finish(s, partial, ec);
}

// ---- inputs of the projections that are A4 already ----------------------------------------------

void input_only_impl(Op op, const Tensor& x, const Weight& w, LinearPolicy policy,
                     const ExecutionContext& ec) {
    if (w.qtype != QType::NVFP4 || x.dtype != DType::BF16 || !x.is_contiguous()) return;
    State& s = state();
    const std::lock_guard<std::mutex> lock(s.mutex);
    const int layer           = layer_index(s, op, w.qdata);
    const std::int32_t tokens = x.ne[1] * x.ne[2] * x.ne[3];
    if (!allows_a4(policy) || tokens < config().t_min || tokens > config().t_max ||
        !budget_left(s) || capturing(ec, s)) {
        return;
    }
    const std::int32_t k = x.ne[0];
    const ScopedCurrentDevice device(ec.dev[0]->device);
    const cudaStream_t stream = ec.dev[0]->stream;
    RankBuffers& b            = buffers(s, 0);
    DeviceArena arena(b.scratch);
    HostInput input;
    const Tensor flat(x.data, DType::BF16, {k, tokens});
    quantize_to_host(flat, w, allocate_nvfp4_a4_workspace(arena, tokens, k), input, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const InputStats in = host_input_stats(input, k, tokens, w.input_scale_divisor);
    std::string line    = record_head(s, op, layer, tokens, w.n, k);
    append(line, ",\"in\":[%s]}", input_json(in).c_str());
    write_line(s, line);
}

// ---- host helpers -------------------------------------------------------------------------------

float decode_e2m1(std::uint8_t nibble) noexcept {
    static constexpr float kMagnitude[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    const float magnitude                = kMagnitude[nibble & 7U];
    return (nibble & 8U) != 0 ? -magnitude : magnitude;
}

float decode_e4m3(std::uint8_t byte) noexcept {
    const int exponent = (byte >> 3) & 0xF;
    const int mantissa = byte & 7;
    const float magnitude =
        exponent == 0 ? std::ldexp(static_cast<float>(mantissa), -9)
                      : std::ldexp(static_cast<float>(8 + mantissa), exponent - 10);
    return (byte & 0x80U) != 0 ? -magnitude : magnitude;
}

float bf16_to_float(std::uint16_t bits) noexcept {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
    float out;
    std::memcpy(&out, &wide, sizeof(out));
    return out;
}

void dequantize_a4(const std::uint8_t* codes, const std::uint8_t* scales, std::int32_t k,
                   std::int32_t tokens, float input_scale_divisor, float* out) {
    const std::int32_t groups = k / 16;
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t g = 0; g < groups; ++g) {
            const double scale = decode_e4m3(scales[static_cast<std::size_t>(t) * groups + g]) /
                                 static_cast<double>(input_scale_divisor);
            const std::uint8_t* block =
                codes + static_cast<std::size_t>(t) * (k / 2) + static_cast<std::size_t>(g) * 8;
            float* dst = out + static_cast<std::size_t>(t) * k + static_cast<std::size_t>(g) * 16;
            for (int e = 0; e < 16; ++e) {
                const std::uint8_t byte   = block[e / 2];
                const std::uint8_t nibble = (e & 1) != 0 ? (byte >> 4) : (byte & 0xFU);
                dst[e] = static_cast<float>(decode_e2m1(nibble) * scale);
            }
        }
    }
}

double DiffStats::rel() const noexcept { return ref2 > 0 ? std::sqrt(err2 / ref2) : 0.0; }

double DiffStats::ref_rms() const noexcept {
    return count != 0 ? std::sqrt(ref2 / static_cast<double>(count)) : 0.0;
}

void DiffStats::add(const DiffStats& other) noexcept {
    err2 += other.err2;
    ref2 += other.ref2;
    max_abs  = std::max(max_abs, other.max_abs);
    ref_amax = std::max(ref_amax, other.ref_amax);
    count += other.count;
}

DiffStats diff_stats(const float* test, const float* ref, std::size_t count) noexcept {
    DiffStats d;
    d.count = count;
    for (std::size_t i = 0; i < count; ++i) {
        const double r = ref[i];
        const double e = static_cast<double>(test[i]) - r;
        d.err2 += e * e;
        d.ref2 += r * r;
        d.max_abs  = std::max(d.max_abs, std::abs(e));
        d.ref_amax = std::max(d.ref_amax, std::abs(r));
    }
    return d;
}

InputStats input_stats(const float* x, const float* dequantized, const std::uint8_t* scales,
                       std::int32_t k, std::int32_t tokens, float input_scale_divisor) {
    InputStats in;
    const std::size_t count  = static_cast<std::size_t>(k) * tokens;
    const std::size_t blocks = count / 16;
    if (count == 0) return in;
    std::vector<double> block_amax(blocks);
    double sum2 = 0.0, err2 = 0.0;
    std::size_t nonzero = 0, zeroed = 0, saturated = 0, subnormal = 0, zero_scale = 0;
    for (std::size_t b = 0; b < blocks; ++b) {
        double amax = 0.0;
        for (std::size_t i = b * 16; i < b * 16 + 16; ++i) {
            const double v = x[i];
            const double e = static_cast<double>(dequantized[i]) - v;
            amax           = std::max(amax, std::abs(v));
            sum2 += v * v;
            err2 += e * e;
            if (v != 0.0) {
                ++nonzero;
                if (dequantized[i] == 0.0F) ++zeroed;
            }
        }
        block_amax[b]            = amax;
        in.amax                  = std::max(in.amax, amax);
        const std::uint8_t scale = scales[b]; // [T][K/16] is block order
        if (amax * input_scale_divisor / 6.0 > 448.0) ++saturated;
        if (scale == 0 && amax > 0.0) ++zero_scale;
        if (scale != 0 && ((scale >> 3) & 0xF) == 0) ++subnormal;
    }
    in.rms             = std::sqrt(sum2 / static_cast<double>(count));
    in.calibrated_amax = 6.0 * 448.0 / input_scale_divisor;
    std::vector<double> sorted = block_amax;
    const auto nth             = [&](double q) {
        const std::size_t index =
            std::min(blocks - 1, static_cast<std::size_t>(q * static_cast<double>(blocks - 1)));
        std::nth_element(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(index),
                         sorted.end());
        return sorted[index];
    };
    in.block_amax_median = nth(0.5);
    in.block_amax_p99    = nth(0.99);
    in.saturated_blocks  = static_cast<double>(saturated) / static_cast<double>(blocks);
    in.subnormal_blocks  = static_cast<double>(subnormal) / static_cast<double>(blocks);
    in.zero_blocks       = static_cast<double>(zero_scale) / static_cast<double>(blocks);
    in.quant_rel         = sum2 > 0 ? std::sqrt(err2 / sum2) : 0.0;
    in.quant_zeroed      = nonzero ? static_cast<double>(zeroed) / static_cast<double>(nonzero) : 0.0;
    return in;
}

} // namespace ninfer::ops::detail::a4probe
