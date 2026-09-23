#pragma once

#include "models/qwen3_5/program/internal.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "models/qwen3_5/program/vision_control.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/vision_prefill.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

using detail::VisionWorkspacePlan;
using detail::VisionPrefillPlan;
using detail::VisionUseSpan;

struct VisionItemView {
    std::span<const std::uint16_t> patches;
    const qwen3_5::VisionItemControl* control = nullptr;
};

class VisionContext {
public:
    VisionContext(DeviceContext& device, const execution::Parameters& parameters);

    [[nodiscard]] static std::size_t workspace_bytes(const VisionConfig& config,
                                                     const VisionParameters& parameters,
                                                     std::size_t patches, std::size_t merged_tokens,
                                                     const VisionWorkspacePlan& plan);
    [[nodiscard]] static VisionWorkspacePlan plan_workspace(const VisionConfig& config,
                                                            const VisionParameters& parameters,
                                                            std::uint32_t max_merged_tokens,
                                                            std::size_t general_capacity_bytes);
    // The workspace of a tensor-parallel rank that receives `encode`'s item output without
    // encoding: the same general prefix and handoff extent, with the handoff right after the
    // prefix and no encode region.
    [[nodiscard]] static VisionWorkspacePlan plan_receiver(const VisionWorkspacePlan& encode);

    [[nodiscard]] const VisionConfig& config() const noexcept { return config_; }

    [[nodiscard]] static Tensor bind_output(DeviceSpan backing, const VisionWorkspacePlan& plan,
                                            std::size_t merged_tokens);
    void encode(const VisionItemView& item, Tensor& output, DeviceSpan backing,
                const VisionWorkspacePlan& plan) const;

private:
    DeviceContext& ctx_;
    const VisionConfig& config_;
    const VisionParameters& parameters_;
};

struct VisionChunk {
    std::int32_t length                       = 0;
    const qwen3_5::VisionItemControl* control = nullptr;
    // The active item's merged embeddings [H,V] in rank 0's handoff, and at tp 2 the same values
    // in rank 1's handoff; empty otherwise.
    Tensor embeddings;
    Tensor peer_embeddings;
};

// One tensor-parallel rank's view of a Vision prefill: its device, Parameters (which hold the
// tower only on the encoding rank), workspace and that workspace's Vision plan.
struct VisionRank {
    DeviceContext* device                   = nullptr;
    const execution::Parameters* parameters = nullptr;
    DeviceSpan workspace;
    const VisionWorkspacePlan* workspace_plan = nullptr;
};

// Encodes each suffix item once. At tp 1 the item is encoded in the handoff of `device`'s
// workspace. At tp 2 (`peer` names rank 1) the rank `vision_rank` encodes it in its own handoff,
// then the other rank's stream waits for the encode and copies the merged embeddings into its own
// handoff (the driver stages the copy through host memory without peer access), and the encoding
// rank's stream waits for that copy before it may overwrite its handoff. Each rank's text prefill
// reads its own copy in stream order.
class VisionPrefillSession {
public:
    VisionPrefillSession(DeviceContext& device, const execution::Parameters& parameters,
                         DeviceSpan workspace, const VisionWorkspacePlan& workspace_plan,
                         qwen3_5::PreparedPromptData& prompt, const VisionPrefillPlan& plan,
                         std::size_t& handoff_peak_bytes);
    VisionPrefillSession(const VisionRank& rank0, const VisionRank& peer, int vision_rank,
                         qwen3_5::PreparedPromptData& prompt, const VisionPrefillPlan& plan,
                         std::size_t& handoff_peak_bytes);

    [[nodiscard]] VisionChunk prepare_chunk(std::uint32_t begin, std::uint32_t nominal_length);
    void release_encoded_media_payloads() noexcept;
    void retire_handoff() noexcept;
    [[nodiscard]] double elapsed_seconds() const;

    [[nodiscard]] std::size_t active_handoff_bytes() const noexcept {
        return active_handoff_bytes_;
    }

private:
    VisionPrefillSession(const std::array<VisionRank, 2>& ranks, bool tensor_parallel,
                         int vision_rank, qwen3_5::PreparedPromptData& prompt,
                         const VisionPrefillPlan& plan, std::size_t& handoff_peak_bytes);

    [[nodiscard]] Tensor bind_rank_output(int rank, std::size_t merged_tokens) const;
    void encode_on_ranks(const VisionItemView& item, Tensor& rank0_output, Tensor& peer_output);

    // ranks_[0] is rank 0 (the only rank at tp 1); ranks_[1] is rank 1 at tp 2.
    std::array<VisionRank, 2> ranks_;
    bool tensor_parallel_ = false;
    int vision_rank_      = 0;
    qwen3_5::PreparedPromptData& prompt_;
    const VisionPrefillPlan& plan_;
    std::size_t& handoff_peak_bytes_;
    VisionContext context_;
    // tp 2: `encoded_` is recorded on the encoding rank's stream after an item's encode and
    // `copied_` on the other rank's stream after its copy of the item output.
    std::optional<CudaCompletionEvent> encoded_;
    std::optional<CudaCompletionEvent> copied_;
    std::size_t next_use_ = 0;
    std::optional<std::uint32_t> active_item_;
    std::size_t active_handoff_bytes_ = 0;
    std::vector<std::uint32_t> encoded_payloads_pending_release_;
    std::vector<CudaEventTimer> timers_;
};

} // namespace ninfer::models::qwen3_5::execution
