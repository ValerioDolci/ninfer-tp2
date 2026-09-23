#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/vision.h"

#include "core/device.h"
#include "core/device_scope.h"
#include "core/layout.h"
#include "core/nvtx.h"
#include "models/qwen3_5/program/vision_control.h"
#include "ninfer/ops/add_bias.h"
#include "ninfer/ops/gelu.h"
#include "ninfer/ops/layer_norm.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/vision_pos_embed.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5::execution {
namespace {

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(std::string("Vision ") + label + " overflows size_t");
    }
    return a * b;
}

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        throw std::overflow_error(std::string("Vision ") + label + " overflows size_t");
    }
    return a + b;
}

std::size_t align_up(std::size_t value, std::size_t alignment, const char* label) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument(std::string("Vision ") + label +
                                    " alignment must be a power of two");
    }
    return checked_add(value, alignment - 1, label) & ~(alignment - 1);
}

constexpr std::size_t kWorkspaceAlignment = 256;

struct VisionWorkspaceLayout {
    TensorRegion position_ids;
    TensorRegion pos_indices;
    TensorRegion pos_weights;
    TensorRegion x;
    TensorRegion patch_bf16;
    TensorRegion attended;
    TensorRegion qkv;
    TensorRegion attention_norm;
    TensorRegion projected;
    TensorRegion mlp_down;
    TensorRegion mlp_up;
    TensorRegion mlp_norm;
    TensorRegion normalized;
    TensorRegion merger_hidden;
    LayoutRegion patch_scratch, qkv_scratch, projection_scratch;
    LayoutRegion up_scratch, down_scratch, merger_first_scratch, merger_second_scratch;
    std::size_t bytes = 0;
};

TensorRegion alias_tensor(const TensorRegion& storage, DType dtype,
                          std::initializer_list<std::int32_t> shape, const char* label) {
    Tensor tensor(nullptr, dtype, shape);
    if (tensor.bytes() > storage.region.bytes) {
        throw std::logic_error(std::string("Vision ") + label +
                               " does not fit its aliased storage");
    }
    TensorRegion out;
    out.region = LayoutRegion{storage.region.offset, tensor.bytes(), storage.region.alignment};
    out.dtype  = dtype;
    std::copy(shape.begin(), shape.end(), out.shape.begin());
    return out;
}

VisionWorkspaceLayout build_workspace_layout(const VisionConfig& config,
                                             const VisionParameters& parameters,
                                             std::size_t patches64, std::size_t tokens64,
                                             std::size_t handoff_offset_bytes) {
    if (patches64 == 0 || tokens64 == 0 ||
        patches64 != checked_mul(tokens64,
                                 dimension(std::uint64_t(config.spatial_merge_size) *
                                           config.spatial_merge_size),
                                 "patch/token relation")) {
        throw std::invalid_argument(
            "Vision workspace requires the configured positive patch/token ratio");
    }
    if (patches64 > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        tokens64 > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("Vision request dimensions exceed int32");
    }
    const auto patches = static_cast<std::int32_t>(patches64);
    const auto tokens  = static_cast<std::int32_t>(tokens64);

    const auto capacity = [](const LinearParameters& p, std::int32_t first, std::int32_t last) {
        return ops::linear_workspace_capacity_bytes(p.weight.qtype, p.weight.n, p.weight.k,
                                                    p.policy, first, last);
    };
    const auto minimum_patches =
        dimension(std::uint64_t(config.spatial_merge_size) * config.spatial_merge_size);
    std::size_t qkv_bytes = 0, projection_bytes = 0, up_bytes = 0, down_bytes = 0;
    for (const auto& layer : parameters.layers) {
        qkv_bytes = std::max(qkv_bytes, capacity(layer.qkv, minimum_patches, patches));
        projection_bytes =
            std::max(projection_bytes, capacity(layer.output, minimum_patches, patches));
        up_bytes   = std::max(up_bytes, capacity(layer.fc1, minimum_patches, patches));
        down_bytes = std::max(down_bytes, capacity(layer.fc2, minimum_patches, patches));
    }
    LayoutBuilder builder;
    const auto scratch = [&](std::size_t bytes, const char* label) {
        auto scope = builder.scope();
        // A borrowed WorkspaceArena needs backing even when this Op uses no scratch.
        return builder.add(std::max(std::size_t{1}, bytes), kWorkspaceAlignment, label);
    };
    VisionWorkspaceLayout out;
    const auto add = [&](DType dtype, std::initializer_list<std::int32_t> shape,
                         const char* label) {
        return builder.add_tensor(dtype, shape, kWorkspaceAlignment, label);
    };
    out.x = add(DType::BF16, {dimension(config.hidden_size), patches}, "vision residual");
    {
        auto position_lifetime = builder.scope();
        out.position_ids       = add(DType::I32, {patches, 2}, "vision position ids");
        {
            auto patch_scope = builder.scope();
            out.patch_bf16 =
                add(DType::BF16, {dimension(config.patch_width()), patches}, "vision BF16 patches");
            out.patch_scratch =
                scratch(capacity(parameters.patch_embedding, minimum_patches, patches),
                        "patch projection scratch");
        }
        {
            auto position_scope = builder.scope();
            out.pos_indices     = add(DType::I32, {4, patches}, "vision position indices");
            out.pos_weights     = add(DType::FP32, {4, patches}, "vision position weights");
        }
        {
            auto attention_scope = builder.scope();
            out.qkv = add(DType::BF16, {3 * dimension(config.hidden_size), patches}, "vision QKV");
            out.attention_norm     = add(DType::BF16, {dimension(config.hidden_size), patches},
                                         "vision attention norm/attended");
            out.attended           = out.attention_norm;
            out.qkv_scratch        = scratch(qkv_bytes, "QKV scratch");
            out.projection_scratch = scratch(projection_bytes, "attention output scratch");
            out.projected =
                alias_tensor(out.qkv, DType::BF16, {dimension(config.hidden_size), patches},
                             "attention projection output");
        }
        {
            auto mlp_scope = builder.scope();
            out.mlp_up =
                add(DType::BF16, {dimension(config.intermediate_size), patches}, "vision MLP up");
            out.mlp_norm =
                add(DType::BF16, {dimension(config.hidden_size), patches}, "vision MLP norm/down");
            out.mlp_down     = out.mlp_norm;
            out.up_scratch   = scratch(up_bytes, "MLP up scratch");
            out.down_scratch = scratch(down_bytes, "MLP down scratch");
        }
    }
    {
        auto merger_scope = builder.scope();
        out.normalized =
            add(DType::BF16, {dimension(config.hidden_size), patches}, "vision merger norm");
        out.merger_first_scratch =
            scratch(capacity(parameters.merger_fc1, 1, tokens), "merger first scratch");
        out.merger_hidden = alias_tensor(
            out.x, DType::BF16, {dimension(config.merger_width()), tokens}, "merger hidden");
    }
    out.bytes                = builder.finish(1, "vision workspace");
    const auto final_scratch = std::max(std::size_t{1}, capacity(parameters.merger_fc2, 1, tokens));
    {
        LayoutBuilder finish;
        (void)finish.add(handoff_offset_bytes, kWorkspaceAlignment, "handoff start");
        (void)finish.add(
            checked_mul(checked_mul(parameters.merger_fc2.weight.n, tokens64, "output elements"), 2,
                        "output bytes"),
            1, "handoff");
        out.merger_second_scratch =
            finish.add(final_scratch, kWorkspaceAlignment, "merger second scratch");
        out.bytes = std::max(out.bytes, finish.finish(1, "merger call"));
    }
    return out;
}

std::size_t output_handoff_bytes(std::int32_t output_hidden, std::size_t merged_tokens) {
    if (merged_tokens == 0 ||
        merged_tokens > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("Vision output handoff extent must fit positive int32");
    }
    LayoutBuilder layout;
    (void)layout.add_tensor(DType::BF16, {output_hidden, static_cast<std::int32_t>(merged_tokens)},
                            kWorkspaceAlignment, "Vision item output handoff");
    return layout.finish(kWorkspaceAlignment, "Vision item output handoff layout");
}

std::size_t merger_hidden_bytes(const VisionConfig& config, std::size_t merged_tokens) {
    if (merged_tokens == 0 ||
        merged_tokens > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("Vision merger hidden extent must fit positive int32");
    }
    Tensor tensor(nullptr, DType::BF16,
                  {dimension(config.merger_width()), static_cast<std::int32_t>(merged_tokens)});
    return tensor.bytes();
}

void copy_host(const void* src, Tensor& dst, cudaStream_t stream) {
    if (dst.bytes() == 0) { return; }
    CUDA_CHECK(cudaMemcpyAsync(dst.data, src, dst.bytes(), cudaMemcpyHostToDevice, stream));
}

} // namespace

VisionContext::VisionContext(DeviceContext& ctx, const Parameters& parameters)
    : ctx_(ctx), config_(parameters.model.config().vision.value()),
      parameters_(parameters.vision.value()) {}

std::size_t VisionContext::workspace_bytes(const VisionConfig& config,
                                           const VisionParameters& parameters, std::size_t patches,
                                           std::size_t merged_tokens,
                                           const VisionWorkspacePlan& plan) {
    return build_workspace_layout(config, parameters, patches, merged_tokens,
                                  plan.handoff_offset_bytes)
        .bytes;
}

VisionWorkspacePlan VisionContext::plan_workspace(const VisionConfig& config,
                                                  const VisionParameters& parameters,
                                                  std::uint32_t max_merged_tokens,
                                                  std::size_t general_capacity_bytes) {
    if (max_merged_tokens == 0 || general_capacity_bytes == 0) {
        throw std::invalid_argument("Vision workspace extents must be positive");
    }
    VisionWorkspacePlan out;
    out.output_hidden          = parameters.merger_fc2.weight.n;
    out.max_merged_tokens      = max_merged_tokens;
    out.general_capacity_bytes = general_capacity_bytes;
    out.handoff_offset_bytes =
        align_up(std::max(general_capacity_bytes, merger_hidden_bytes(config, max_merged_tokens)),
                 kWorkspaceAlignment, "handoff offset");
    out.handoff_capacity_bytes = output_handoff_bytes(out.output_hidden, max_merged_tokens);
    out.encode_peak_bytes =
        build_workspace_layout(
            config, parameters,
            checked_mul(max_merged_tokens,
                        std::uint64_t(config.spatial_merge_size) * config.spatial_merge_size,
                        "capacity patch count"),
            max_merged_tokens, out.handoff_offset_bytes)
            .bytes;
    out.capacity_bytes = std::max(
        out.encode_peak_bytes,
        checked_add(out.handoff_offset_bytes, out.handoff_capacity_bytes, "workspace capacity"));
    return out;
}

VisionWorkspacePlan VisionContext::plan_receiver(const VisionWorkspacePlan& encode) {
    if (encode.general_capacity_bytes == 0 || encode.handoff_capacity_bytes == 0) {
        throw std::invalid_argument("Vision receiver needs a complete encode workspace plan");
    }
    VisionWorkspacePlan out = encode;
    out.encode_peak_bytes   = 0;
    out.handoff_offset_bytes =
        align_up(encode.general_capacity_bytes, kWorkspaceAlignment, "receiver handoff offset");
    out.capacity_bytes = checked_add(out.handoff_offset_bytes, out.handoff_capacity_bytes,
                                     "receiver workspace capacity");
    return out;
}

Tensor VisionContext::bind_output(DeviceSpan backing, const VisionWorkspacePlan& plan,
                                  std::size_t merged_tokens) {
    if (backing.data == nullptr || backing.bytes < plan.capacity_bytes || merged_tokens == 0 ||
        merged_tokens > plan.max_merged_tokens) {
        throw std::invalid_argument("Vision output binding exceeds its workspace plan");
    }
    const std::size_t bytes = output_handoff_bytes(plan.output_hidden, merged_tokens);
    if (bytes > plan.handoff_capacity_bytes) {
        throw std::logic_error("Vision output binding exceeds its handoff region");
    }
    TensorRegion region;
    region.region = LayoutRegion{plan.handoff_offset_bytes, bytes, kWorkspaceAlignment};
    region.dtype  = DType::BF16;
    region.shape  = {plan.output_hidden, static_cast<std::int32_t>(merged_tokens), 1, 1};
    return region.bind(backing);
}

void VisionContext::encode(const VisionItemView& item, Tensor& output, DeviceSpan backing,
                           const VisionWorkspacePlan& plan) const {
    if (item.control == nullptr) { throw std::invalid_argument("Vision item control is null"); }
    const qwen3_5::VisionItemControl& control = *item.control;
    const auto patches64                      = control.patch_count;
    const auto tokens64                       = control.merged_count;
    nvtx::ScopedRange encode_range(nvtx::Name::VisionEncode, nvtx::Category::Vision,
                                   static_cast<std::uint64_t>(patches64));
    if (item.patches.size() !=
        checked_mul(patches64, dimension(config_.patch_width()), "patch elements")) {
        throw std::invalid_argument("Vision processor patch buffer has invalid shape");
    }
    if (output.dtype != DType::BF16 || output.ne[0] != parameters_.merger_fc2.weight.n ||
        output.ne[1] != static_cast<std::int32_t>(tokens64) || output.ne[2] != 1 ||
        output.ne[3] != 1 || !output.is_contiguous() || output.data == nullptr) {
        throw std::invalid_argument("Vision output must be contiguous BF16 [H,V]");
    }
    const Tensor planned_output = bind_output(backing, plan, tokens64);
    if (output.data != planned_output.data || output.bytes() != planned_output.bytes()) {
        throw std::invalid_argument("Vision output does not name the planned handoff region");
    }
    const VisionWorkspaceLayout layout = build_workspace_layout(
        config_, parameters_, patches64, tokens64, plan.handoff_offset_bytes);
    if (layout.bytes > plan.encode_peak_bytes || backing.bytes < plan.capacity_bytes) {
        throw std::invalid_argument("Vision workspace capacity is too small for request");
    }
    const auto patches  = static_cast<std::int32_t>(patches64);
    const auto tokens   = static_cast<std::int32_t>(tokens64);
    cudaStream_t stream = ctx_.stream;

    const auto project = [&](const Tensor& x, const LinearParameters& p, Tensor& out,
                             const LayoutRegion& region) {
        WorkspaceArena scratch(region.bind(backing));
        ops::linear(x, p.weight, out, p.policy, scratch, stream);
    };
    Tensor position_ids = layout.position_ids.bind(backing);
    Tensor x            = layout.x.bind(backing);
    Tensor patch_bf16   = layout.patch_bf16.bind(backing);
    Tensor pos_indices  = layout.pos_indices.bind(backing);
    Tensor pos_weights  = layout.pos_weights.bind(backing);
    {
        nvtx::ScopedRange patch_range(nvtx::Name::VisionPatchEmbedding, nvtx::Category::Vision,
                                      static_cast<std::uint64_t>(patches64));
        copy_host(control.position_ids.data(), position_ids, stream);
        copy_host(item.patches.data(), patch_bf16, stream);
        project(patch_bf16, parameters_.patch_embedding, x, layout.patch_scratch);
        ops::add_bias(parameters_.patch_embedding_bias, x, stream);
        // The artifact records the source table shape [rows,hidden], while Tensor's
        // contiguous matrix convention is [inner,columns]. The payload is already
        // row-major, so this is a zero-copy [hidden,rows] view, not a transpose.
        copy_host(control.position_table_indices.data(), pos_indices, stream);
        copy_host(control.position_table_weights.data(), pos_weights, stream);
        Tensor position_table = parameters_.position_embedding.reshape(
            {dimension(config_.hidden_size), dimension(config_.num_position_embeddings)});
        ops::vision_pos_embed_add(position_table, pos_indices, pos_weights, x, stream);
    }
    for (std::size_t layer = 0; layer < parameters_.layers.size(); ++layer) {
        nvtx::ScopedRange layer_range(nvtx::Name::VisionLayer, nvtx::Category::Vision,
                                      static_cast<std::uint64_t>(layer));
        const auto& block = parameters_.layers[layer];
        {
            nvtx::ScopedRange attention_range(nvtx::Name::VisionAttention,
                                              nvtx::Category::Attention,
                                              static_cast<std::uint64_t>(layer));
            Tensor attended = layout.attended.bind(backing);
            {
                Tensor qkv = layout.qkv.bind(backing);
                {
                    Tensor h = layout.attention_norm.bind(backing);
                    ops::layer_norm(x, block.norm1.weight, block.norm1.bias, 1.0e-6F, h, stream);
                    project(h, block.qkv, qkv, layout.qkv_scratch);
                }
                ops::add_bias(block.qkv_bias, qkv, stream);
                const std::int32_t plane      = dimension(config_.hidden_size);
                const std::size_t plane_bytes = static_cast<std::size_t>(plane) * 2;
                Tensor q(qkv.data, DType::BF16,
                         {dimension(config_.hidden_size / config_.num_heads),
                          dimension(config_.num_heads), patches});
                Tensor k(static_cast<unsigned char*>(qkv.data) + plane_bytes, DType::BF16,
                         {dimension(config_.hidden_size / config_.num_heads),
                          dimension(config_.num_heads), patches});
                Tensor v(static_cast<unsigned char*>(qkv.data) + 2 * plane_bytes, DType::BF16,
                         {dimension(config_.hidden_size / config_.num_heads),
                          dimension(config_.num_heads), patches});
                q.nb[2] = qkv.nb[1];
                k.nb[2] = qkv.nb[1];
                v.nb[2] = qkv.nb[1];
                ops::rope(position_ids, dimension(config_.hidden_size / config_.num_heads),
                          10'000.0F, q, k, stream);
                Tensor attended_heads =
                    attended.view({dimension(config_.hidden_size / config_.num_heads),
                                   dimension(config_.num_heads), patches});
                ops::packed_softmax_attention(
                    q, k, v,
                    {dimension(config_.hidden_size / config_.num_heads),
                     dimension(config_.num_heads), dimension(config_.num_heads)},
                    static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.hidden_size /
                                                                           config_.num_heads))),
                    control.segment_length, attended_heads, stream);
            }
            Tensor projected = layout.projected.bind(backing);
            project(attended, block.output, projected, layout.projection_scratch);
            ops::add_bias(block.output_bias, projected, stream);
            ops::residual_add(projected, x, stream);
        }
        {
            nvtx::ScopedRange mlp_range(nvtx::Name::VisionMlp, nvtx::Category::PostMixer,
                                        static_cast<std::uint64_t>(layer));
            Tensor down = layout.mlp_down.bind(backing);
            Tensor up   = layout.mlp_up.bind(backing);
            {
                Tensor h = layout.mlp_norm.bind(backing);
                ops::layer_norm(x, block.norm2.weight, block.norm2.bias, 1.0e-6F, h, stream);
                project(h, block.fc1, up, layout.up_scratch);
            }
            ops::add_bias(block.fc1_bias, up, stream);
            ops::gelu(up, ops::GeluMode::Tanh, stream);
            project(up, block.fc2, down, layout.down_scratch);
            ops::add_bias(block.fc2_bias, down, stream);
            ops::residual_add(down, x, stream);
        }
    }

    {
        nvtx::ScopedRange merge_range(nvtx::Name::VisionMerge, nvtx::Category::Vision,
                                      static_cast<std::uint64_t>(tokens64));
        Tensor normalized = layout.normalized.bind(backing);
        ops::layer_norm(x, parameters_.merger_norm.weight, parameters_.merger_norm.bias, 1.0e-6F,
                        normalized, stream);
        Tensor merged = normalized.view({dimension(config_.merger_width()), tokens});
        Tensor hidden = layout.merger_hidden.bind(backing);
        project(merged, parameters_.merger_fc1, hidden, layout.merger_first_scratch);
        ops::add_bias(parameters_.merger_fc1_bias, hidden, stream);
        ops::gelu(hidden, ops::GeluMode::Exact, stream);
        project(hidden, parameters_.merger_fc2, output, layout.merger_second_scratch);
        ops::add_bias(parameters_.merger_fc2_bias, output, stream);
    }
}

namespace {

// The rank that runs the encoder, after checking both ranks' bindings.
const VisionRank& encoding_rank(const std::array<VisionRank, 2>& ranks, bool tensor_parallel,
                                int vision_rank) {
    const int count = tensor_parallel ? 2 : 1;
    if (vision_rank < 0 || vision_rank >= count) {
        throw std::invalid_argument("Vision prefill names no encoding rank");
    }
    for (int rank = 0; rank < count; ++rank) {
        const VisionRank& binding = ranks[static_cast<std::size_t>(rank)];
        if (binding.device == nullptr || binding.parameters == nullptr ||
            binding.workspace_plan == nullptr) {
            throw std::invalid_argument("Vision prefill rank binding is incomplete");
        }
    }
    const VisionRank& encoder = ranks[static_cast<std::size_t>(vision_rank)];
    if (!encoder.parameters->vision) {
        throw std::invalid_argument("Vision prefill encoding rank holds no Vision tower");
    }
    return encoder;
}

} // namespace

VisionPrefillSession::VisionPrefillSession(
    DeviceContext& device, const execution::Parameters& parameters, DeviceSpan workspace,
    const VisionWorkspacePlan& workspace_plan, qwen3_5::PreparedPromptData& prompt,
    const VisionPrefillPlan& plan, std::size_t& handoff_peak_bytes)
    : VisionPrefillSession(
          std::array<VisionRank, 2>{VisionRank{&device, &parameters, workspace, &workspace_plan},
                                    VisionRank{}},
          false, 0, prompt, plan, handoff_peak_bytes) {}

VisionPrefillSession::VisionPrefillSession(const VisionRank& rank0, const VisionRank& peer,
                                           int vision_rank, qwen3_5::PreparedPromptData& prompt,
                                           const VisionPrefillPlan& plan,
                                           std::size_t& handoff_peak_bytes)
    : VisionPrefillSession(std::array<VisionRank, 2>{rank0, peer}, true, vision_rank, prompt, plan,
                           handoff_peak_bytes) {}

VisionPrefillSession::VisionPrefillSession(const std::array<VisionRank, 2>& ranks,
                                           bool tensor_parallel, int vision_rank,
                                           qwen3_5::PreparedPromptData& prompt,
                                           const VisionPrefillPlan& plan,
                                           std::size_t& handoff_peak_bytes)
    : ranks_(ranks), tensor_parallel_(tensor_parallel), vision_rank_(vision_rank), prompt_(prompt),
      plan_(plan), handoff_peak_bytes_(handoff_peak_bytes),
      context_(*encoding_rank(ranks, tensor_parallel, vision_rank).device,
               *encoding_rank(ranks, tensor_parallel, vision_rank).parameters) {
    if (plan_.control == nullptr || plan_.control->items.empty() || plan_.uses.empty()) {
        throw std::invalid_argument("Vision prefill plan has no suffix item spans");
    }
    const int rank_count = tensor_parallel_ ? 2 : 1;
    for (int rank = 0; rank < rank_count; ++rank) {
        const VisionRank& binding = ranks_[static_cast<std::size_t>(rank)];
        if (binding.workspace.data == nullptr ||
            binding.workspace.bytes < binding.workspace_plan->capacity_bytes ||
            plan_.max_merged_count == 0 ||
            plan_.max_merged_count > binding.workspace_plan->max_merged_tokens) {
            throw std::invalid_argument("Vision prefill workspace plan is invalid");
        }
    }
    if (tensor_parallel_) {
        // The copy moves one item output from the encoding rank's handoff into the other's.
        const VisionWorkspacePlan& first  = *ranks_[0].workspace_plan;
        const VisionWorkspacePlan& second = *ranks_[1].workspace_plan;
        if (ranks_[0].device == ranks_[1].device || first.output_hidden != second.output_hidden ||
            first.max_merged_tokens != second.max_merged_tokens ||
            first.handoff_capacity_bytes != second.handoff_capacity_bytes) {
            throw std::invalid_argument("tensor-parallel Vision ranks disagree on the item output");
        }
        const VisionRank& encoder  = ranks_[static_cast<std::size_t>(vision_rank_)];
        const VisionRank& receiver = ranks_[static_cast<std::size_t>(1 - vision_rank_)];
        // Each event belongs to the device of the stream that records it; creating one makes its
        // device current, which the scope restores.
        const ScopedCurrentDevice restore;
        encoded_.emplace(*encoder.device);
        copied_.emplace(*receiver.device);
    }
    std::uint32_t previous_end = 0;
    std::optional<std::uint32_t> previous_item;
    for (const VisionUseSpan& use : plan_.uses) {
        if (use.begin >= use.end || use.begin < previous_end ||
            use.end > prompt_.token_ids.size()) {
            throw std::invalid_argument("Vision suffix item spans are invalid or unordered");
        }
        if (use.control_index >= plan_.control->items.size() ||
            use.prepared_item_index >= prompt_.vision_items.size() ||
            use.prepared_item_index >= prompt_.media_payloads.size() ||
            plan_.control->prepared_item_begin + use.control_index != use.prepared_item_index ||
            (previous_item && use.prepared_item_index <= *previous_item)) {
            throw std::invalid_argument("Vision suffix item indices are invalid or unordered");
        }
        const qwen3_5::VisionItemControl& control = plan_.control->items[use.control_index];
        const qwen3_5::VisionItem& source         = prompt_.vision_items[use.prepared_item_index];
        if (control.scatter_indices.empty() ||
            use.end != static_cast<std::uint32_t>(control.scatter_indices.back()) + 1U ||
            (use.begin != static_cast<std::uint32_t>(control.scatter_indices.front()) &&
             use.begin + 1U != static_cast<std::uint32_t>(control.scatter_indices.front())) ||
            source.modality != control.modality || source.grid.temporal != control.grid.temporal ||
            source.grid.height != control.grid.height || source.grid.width != control.grid.width ||
            source.patch_begin != control.patch_begin ||
            source.patch_count != control.patch_count) {
            throw std::invalid_argument("Vision suffix plan does not describe the prepared item");
        }
        if (control.merged_count > plan_.max_merged_count) {
            throw std::invalid_argument("Vision suffix item exceeds its request workspace extent");
        }
        const std::size_t patch_elements = checked_mul(
            control.patch_count, static_cast<std::size_t>(context_.config().patch_width()),
            "item patch elements");
        const auto& payload = prompt_.media_payloads[use.prepared_item_index];
        bool handoff_fits   = true;
        for (int rank = 0; rank < rank_count; ++rank) {
            const Tensor output = bind_rank_output(rank, control.merged_count);
            handoff_fits =
                handoff_fits &&
                output.bytes() <=
                    ranks_[static_cast<std::size_t>(rank)].workspace_plan->handoff_capacity_bytes;
        }
        if (!handoff_fits || !payload || payload->patch_elements != patch_elements) {
            throw std::invalid_argument("Vision suffix item storage has an invalid shape");
        }
        previous_end  = use.end;
        previous_item = use.prepared_item_index;
    }
    if (plan_.max_merged_count != 0 &&
        std::none_of(plan_.control->items.begin(), plan_.control->items.end(),
                     [&](const qwen3_5::VisionItemControl& item) {
                         return item.merged_count == plan_.max_merged_count;
                     })) {
        throw std::invalid_argument("Vision request workspace extent has no matching suffix item");
    }
    encoded_payloads_pending_release_.reserve(plan_.uses.size());
    timers_.reserve(plan_.uses.size());
}

Tensor VisionPrefillSession::bind_rank_output(int rank, std::size_t merged_tokens) const {
    const VisionRank& binding = ranks_[static_cast<std::size_t>(rank)];
    return VisionContext::bind_output(binding.workspace, *binding.workspace_plan, merged_tokens);
}

void VisionPrefillSession::encode_on_ranks(const VisionItemView& item, Tensor& rank0_output,
                                           Tensor& peer_output) {
    const VisionRank& encoder  = ranks_[static_cast<std::size_t>(vision_rank_)];
    const VisionRank& receiver = ranks_[static_cast<std::size_t>(1 - vision_rank_)];
    Tensor& encoded            = vision_rank_ == 0 ? rank0_output : peer_output;
    const Tensor& received     = vision_rank_ == 0 ? peer_output : rank0_output;
    // Kernel launches, events and timers follow the current device.
    const ScopedCurrentDevice restore(encoder.device->device);
    timers_.emplace_back(*encoder.device);
    timers_.back().start();
    context_.encode(item, encoded, encoder.workspace, *encoder.workspace_plan);
    timers_.back().record_stop();
    encoded_->record(encoder.device->stream);

    ScopedCurrentDevice::select(receiver.device->device);
    encoded_->wait(receiver.device->stream);
    CUDA_CHECK(cudaMemcpyAsync(received.data, encoded.data, encoded.bytes(),
                               cudaMemcpyDeviceToDevice, receiver.device->stream));
    copied_->record(receiver.device->stream);

    // The next item's encode overwrites the encoding rank's handoff only after the copy read it.
    ScopedCurrentDevice::select(encoder.device->device);
    copied_->wait(encoder.device->stream);
}

VisionChunk VisionPrefillSession::prepare_chunk(std::uint32_t begin, std::uint32_t nominal_length) {
    if (nominal_length == 0 || begin >= prompt_.token_ids.size()) {
        throw std::invalid_argument("Vision chunk range is empty or outside the prompt");
    }
    const std::uint64_t nominal_end64 =
        static_cast<std::uint64_t>(begin) + static_cast<std::uint64_t>(nominal_length);
    std::uint32_t end = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(nominal_end64, prompt_.token_ids.size()));

    while (next_use_ < plan_.uses.size() && plan_.uses[next_use_].end <= begin) { ++next_use_; }
    const VisionUseSpan* active = nullptr;
    if (next_use_ < plan_.uses.size() && plan_.uses[next_use_].begin < end) {
        active = &plan_.uses[next_use_];
        if (next_use_ + 1U < plan_.uses.size()) {
            end = std::min(end, plan_.uses[next_use_ + 1U].begin);
        }
    }
    if (end <= begin) { throw std::logic_error("Vision chunk cap made no forward progress"); }
    if (active == nullptr) {
        return VisionChunk{static_cast<std::int32_t>(end - begin), nullptr, {}, {}};
    }
    const qwen3_5::VisionItemControl& control = plan_.control->items[active->control_index];
    Tensor output                             = bind_rank_output(0, control.merged_count);
    Tensor peer_output = tensor_parallel_ ? bind_rank_output(1, control.merged_count) : Tensor{};

    if (!active_item_ || *active_item_ != active->prepared_item_index) {
        const auto& payload = prompt_.media_payloads[active->prepared_item_index];
        const VisionItemView item{payload->span(), &control};
        if (tensor_parallel_) {
            encode_on_ranks(item, output, peer_output);
        } else {
            timers_.emplace_back(*ranks_[0].device);
            timers_.back().start();
            context_.encode(item, output, ranks_[0].workspace, *ranks_[0].workspace_plan);
            timers_.back().record_stop();
        }
        active_item_          = active->prepared_item_index;
        active_handoff_bytes_ = output.bytes();
        handoff_peak_bytes_   = std::max(handoff_peak_bytes_, active_handoff_bytes_);
        encoded_payloads_pending_release_.push_back(active->prepared_item_index);
    }
    return VisionChunk{static_cast<std::int32_t>(end - begin), &control, output, peer_output};
}

void VisionPrefillSession::release_encoded_media_payloads() noexcept {
    for (const std::uint32_t item_index : encoded_payloads_pending_release_) {
        if (item_index >= prompt_.media_payloads.size()) { std::terminate(); }
        prompt_.media_payloads[item_index].reset();
    }
    encoded_payloads_pending_release_.clear();
}

void VisionPrefillSession::retire_handoff() noexcept {
    active_item_.reset();
    active_handoff_bytes_ = 0;
}

double VisionPrefillSession::elapsed_seconds() const {
    double milliseconds = 0.0;
    for (const CudaEventTimer& timer : timers_) { milliseconds += timer.elapsed_ms(); }
    return milliseconds / 1000.0;
}

} // namespace ninfer::models::qwen3_5::execution
