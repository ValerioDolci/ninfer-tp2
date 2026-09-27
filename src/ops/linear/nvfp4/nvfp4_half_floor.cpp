#include "ops/linear/nvfp4/nvfp4_geometry.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::ops::detail {
namespace {

std::int32_t constant_floor(Nvfp4HalfFloor which) {
    switch (which) {
    case Nvfp4HalfFloor::GateUp:
        return kNvfp4GateUpFirstA4Tokens;
    case Nvfp4HalfFloor::Down:
        return kNvfp4DownFamilyFirstA4Tokens;
    case Nvfp4HalfFloor::Output:
        return kNvfp4OutputFamilyFirstA4Tokens;
    }
    throw std::logic_error("unreachable NVFP4 half floor");
}

std::string_view trim(std::string_view text) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) text.remove_prefix(1);
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t')) text.remove_suffix(1);
    return text;
}

Nvfp4HalfFloorOverride read_environment() {
    const char* spec = std::getenv("NINFER_TP2_A4_HALF");
    const Nvfp4HalfFloorOverride parsed =
        parse_nvfp4_half_floor_override(spec == nullptr ? "" : spec);
    if (parsed.gate_up != 0 || parsed.down != 0 || parsed.output != 0) {
        const auto shown = [](std::int32_t value, std::int32_t constant) {
            return value != 0 ? value : constant;
        };
        std::fprintf(stderr,
                     "ninfer: NINFER_TP2_A4_HALF=%s (experiment): A4 floor of the tp 2 halves "
                     "swiglu %d, down %d, out %d\n",
                     spec, shown(parsed.gate_up, kNvfp4GateUpFirstA4Tokens),
                     shown(parsed.down, kNvfp4DownFamilyFirstA4Tokens),
                     shown(parsed.output, kNvfp4OutputFamilyFirstA4Tokens));
    }
    return parsed;
}

} // namespace

Nvfp4HalfFloorOverride parse_nvfp4_half_floor_override(const char* spec) {
    Nvfp4HalfFloorOverride result;
    std::string_view rest = spec == nullptr ? std::string_view{} : std::string_view{spec};
    while (!rest.empty()) {
        const std::size_t comma = rest.find(',');
        const std::string_view item =
            trim(comma == std::string_view::npos ? rest : rest.substr(0, comma));
        rest = comma == std::string_view::npos ? std::string_view{} : rest.substr(comma + 1);
        if (item.empty()) { continue; }
        const std::size_t equals    = item.find('=');
        const std::string_view name = trim(item.substr(0, equals));
        Nvfp4HalfFloor which{};
        std::int32_t* slot = nullptr;
        if (name == "swiglu") {
            which = Nvfp4HalfFloor::GateUp;
            slot  = &result.gate_up;
        } else if (name == "down") {
            which = Nvfp4HalfFloor::Down;
            slot  = &result.down;
        } else if (name == "out") {
            which = Nvfp4HalfFloor::Output;
            slot  = &result.output;
        } else {
            throw std::invalid_argument("NINFER_TP2_A4_HALF: unknown projection '" +
                                        std::string(name) + "' (swiglu, down, out)");
        }
        std::int32_t floor = 3;
        if (equals != std::string_view::npos) {
            const std::string digits(trim(item.substr(equals + 1)));
            char* end         = nullptr;
            const long parsed = std::strtol(digits.c_str(), &end, 10);
            if (digits.empty() || end == nullptr || *end != '\0' || parsed < 2 ||
                parsed > constant_floor(which)) {
                throw std::invalid_argument("NINFER_TP2_A4_HALF: floor of '" + std::string(name) +
                                            "' must be an integer in [2, " +
                                            std::to_string(constant_floor(which)) + "]");
            }
            floor = static_cast<std::int32_t>(parsed);
        }
        *slot = floor;
    }
    return result;
}

std::int32_t nvfp4_half_first_a4_tokens(Nvfp4HalfFloor which) {
    static const Nvfp4HalfFloorOverride environment = read_environment();
    std::int32_t value = 0;
    switch (which) {
    case Nvfp4HalfFloor::GateUp:
        value = environment.gate_up;
        break;
    case Nvfp4HalfFloor::Down:
        value = environment.down;
        break;
    case Nvfp4HalfFloor::Output:
        value = environment.output;
        break;
    }
    return value != 0 ? value : constant_floor(which);
}

} // namespace ninfer::ops::detail
