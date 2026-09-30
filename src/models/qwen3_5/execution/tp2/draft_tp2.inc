// Translation-unit fragment of src/models/qwen3_5/execution/draft.cpp, included once at its end
// (after its closing namespace): the split DFlash2 drafter at tensor-parallel width 2
// (execution/tp2/draft_split.h), sharing draft.cpp's private helpers (the context append, the
// dynamic convolution's prepare, the feature sinks). Not a header. Our file; see
// docs/maintainer/upstream-merge.md.
//
// Rank r holds its 16 of the 32 query heads and 4 of the 8 KV heads of every drafter layer, half of
// the MLP rows and the matching input columns of the attention output and MLP down projections;
// the norms, the dynamic convolution kernels, the feature projection and the token embedding are
// replicated. The residual is replicated and bit-identical on both ranks: the two row-parallel
// projections of a layer write BF16 partials that one allreduce_sum adds (the same combine on both
// ranks), and both ranks then apply the convolution finish to the identical sum.

#include "core/device_scope.h"
#include "models/qwen3_5/execution/tp2/workspace_split.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/tp2/attn_input_proj.h"
#include "ninfer/ops/tp2/dynamic_grouped_conv.h"
#include "ninfer/ops/tp2/rmsnorm_rope.h"
#include "ninfer/ops/tp2/sliding_window_attention.h"

namespace ninfer::models::qwen3_5::execution {
namespace {

constexpr std::int32_t kSplitQueryHeads = 16;
constexpr std::int32_t kSplitKvHeads    = 4;

// Rank r's operands of the split round.
struct DraftRank {
    ExecutionCore* execution;
    DFlashPersistentState* dflash;
    qwen3_5::DFlashDecodeState* frame;
};

std::array<DraftRank, 2> draft_ranks(DFlashBatchContext& state) {
    DFlashPeerDrafter& peer = *state.execution.tp->dflash;
    return {DraftRank{&state.execution, &state.dflash, &state.frame},
            DraftRank{&peer.execution, &peer.dflash, state.peer_frame}};
}

template <class Body>
void each_rank(const ExecutionContext& ec, Body&& body) {
    const ScopedCurrentDevice restore;
    for (int rank = 0; rank < 2; ++rank) {
        ScopedCurrentDevice::select(ec.dev[static_cast<std::size_t>(rank)]->device);
        body(static_cast<std::size_t>(rank));
    }
}

void propose_dflash2_batch_tp2(DFlashBatchContext& state, int batch, int k,
                               DFlashEnvelopes envelopes) {
    const TpExecution& tp     = *state.execution.tp;
    const ExecutionContext& ec = *tp.execution;
    const auto& target        = state.execution.parameters.model.config().text;
    const auto& config        = *state.execution.parameters.model.config().draft;
    if (config.rms_norm_eps != 1e-6F || config.rope_theta != 1e7F ||
        config.attention.head_dim != 128 || config.attention.num_attention_heads != 32 ||
        config.attention.num_key_value_heads != 8) {
        throw std::invalid_argument("DFlash2 tp2: the split drafter requires the D128 32/8 profile");
    }
    const auto ranks            = draft_ranks(state);
    const int width             = k + 1;
    const int columns           = width * batch;
    const int mask_columns      = k * batch;
    const std::int32_t hidden   = dimension(target.hidden_size);
    const std::int32_t q_rows   = kSplitQueryHeads * 128;
    const std::int32_t kv_rows  = kSplitKvHeads * 128;
    std::array<Tensor, 2> residual, staging, positions, valid, slots;
    each_rank(ec, [&](std::size_t r) {
        const DraftRank& d = ranks[r];
        WorkspaceArena& work = d.execution->work;
        const cudaStream_t stream = d.execution->device.stream;
        work.reset();
        Tensor anchors   = d.frame->anchors.slice(0, 0, batch);
        Tensor frontiers = d.frame->execution_frontiers.slice(0, 0, batch);
        valid[r]         = d.frame->proposal_valid_columns.slice(0, 0, batch);
        slots[r]         = d.frame->state_destination_slots.slice(0, 0, batch);
        Tensor ids       = d.frame->proposal_ids.slice(1, 0, batch);
        positions[r]     = d.frame->proposal_positions.slice(1, 0, batch);
        ops::prepare_masked_block(anchors, frontiers, valid[r], dimension(config.mask_token_id), ids,
                                  positions[r], stream);
        residual[r] = work.alloc(DType::BF16, {hidden, width, batch});
        staging[r]  = work.alloc(DType::BF16, {hidden, width, batch});
        Tensor flat = residual[r].view({hidden, columns});
        ops::embedding(ids.view({columns}), d.execution->parameters.text.token_embedding, flat,
                       stream);
    });
    const auto layer_of = [&](std::size_t r, std::size_t layer) -> const DraftBlockParameters& {
        return ranks[r].execution->parameters.draft->layers[layer];
    };
    const std::size_t layer_count = state.execution.parameters.draft->layers.size();
    for (std::size_t layer = 0; layer < layer_count; ++layer) {
        nvtx::ScopedRange layer_range(nvtx::Name::DFlashLayer, nvtx::Category::DFlash, layer);
        {
            auto scope0 = ranks[0].execution->work.scope();
            auto scope1 = ranks[1].execution->work.scope();
            std::array<workspace::DFlash2BranchRoots, 2> branch;
            std::array<Tensor, 2> partial;
            each_rank(ec, [&](std::size_t r) {
                ExecutionCore& core       = *ranks[r].execution;
                WorkspaceArena& work      = core.work;
                const cudaStream_t stream = core.device.stream;
                const auto& p             = layer_of(r, layer);
                branch[r] = workspace::dflash2_branch(work, target, config, width, batch);
                prepare_dynamic_branch(core, residual[r], p.input_norm, config.rms_norm_eps,
                                       *p.attention_conv, branch[r]);
                Tensor query = work.alloc(DType::BF16, {128, kSplitQueryHeads, width, batch});
                Tensor key   = work.alloc(DType::BF16, {128, kSplitKvHeads, width, batch});
                Tensor value = work.alloc(DType::BF16, {128, kSplitKvHeads, width, batch});
                Tensor query_flat = query.view({q_rows, columns});
                Tensor key_flat   = key.view({kv_rows, columns});
                Tensor value_flat = value.view({kv_rows, columns});
                ops::attn_input_proj_head_block(branch[r].prepared.view({hidden, columns}),
                                                p.query_key_value.weight, query_flat, key_flat,
                                                value_flat, stream);
                ops::rmsnorm_rope_head_block(positions[r], p.query_norm, p.key_norm, query, key,
                                             stream);
                Tensor attention = work.alloc(DType::BF16, {128, kSplitQueryHeads, width, batch});
                ops::sliding_window_attention_head_block(
                    query, key, value, positions[r], valid[r], slots[r],
                    {128, kSplitQueryHeads, kSplitKvHeads},
                    dimension(config.sliding_window.value_or(0)),
                    static_cast<float>(1.0 / std::sqrt(double(config.attention.head_dim))),
                    ranks[r].dflash->local_layer(static_cast<std::uint32_t>(layer)),
                    envelopes.local, work, attention, stream);
                partial[r]       = work.alloc(DType::BF16, {hidden, width, batch});
                Tensor projected = partial[r].view({hidden, columns});
                project(attention.view({q_rows, columns}), p.output, projected, work, stream);
            });
            ops::allreduce_sum(partial, staging, ec, *tp.events);
            each_rank(ec, [&](std::size_t r) {
                Tensor target_residual = residual[r];
                ops::dynamic_grouped_conv_finish_add(partial[r],
                                                     layer_of(r, layer).attention_conv->base_kernel,
                                                     branch[r].finish_delta, target_residual,
                                                     ranks[r].execution->device.stream);
            });
        }
        {
            auto scope0 = ranks[0].execution->work.scope();
            auto scope1 = ranks[1].execution->work.scope();
            std::array<workspace::DFlash2BranchRoots, 2> branch;
            std::array<Tensor, 2> partial;
            each_rank(ec, [&](std::size_t r) {
                ExecutionCore& core       = *ranks[r].execution;
                WorkspaceArena& work      = core.work;
                const cudaStream_t stream = core.device.stream;
                const auto& p             = layer_of(r, layer);
                branch[r] = workspace::dflash2_branch(work, target, config, width, batch);
                prepare_dynamic_branch(core, residual[r], p.post_attention_norm,
                                       config.rms_norm_eps, *p.mlp_conv, branch[r]);
                const std::int32_t gate_up = p.mlp.gate_up.weight.n;
                const std::int32_t half    = gate_up / 2;
                Tensor gu = work.alloc(DType::BF16, {gate_up, columns});
                project(branch[r].prepared.view({hidden, columns}), p.mlp.gate_up, gu, work,
                        stream);
                Tensor activation = work.alloc(DType::BF16, {half, columns});
                ops::silu_mul(gu.slice(0, 0, half), gu.slice(0, half, half), activation, stream);
                partial[r]       = work.alloc(DType::BF16, {hidden, width, batch});
                Tensor projected = partial[r].view({hidden, columns});
                project(activation, p.mlp.down, projected, work, stream);
            });
            ops::allreduce_sum(partial, staging, ec, *tp.events);
            each_rank(ec, [&](std::size_t r) {
                Tensor target_residual = residual[r];
                ops::dynamic_grouped_conv_finish_add(partial[r],
                                                     layer_of(r, layer).mlp_conv->base_kernel,
                                                     branch[r].finish_delta, target_residual,
                                                     ranks[r].execution->device.stream);
            });
        }
    }
    std::array<Tensor, 2> final_hidden;
    each_rank(ec, [&](std::size_t r) {
        ExecutionCore& core = *ranks[r].execution;
        final_hidden[r]     = core.work.alloc(DType::BF16, {hidden, mask_columns});
        ops::rmsnorm_pack_tail(residual[r], core.parameters.draft->final_norm, final_hidden[r],
                               core.device.stream);
    });

    // Candidates: both ranks rank their head blocks, rank 0 merges; the selector is rank 0's.
    const ScopedCurrentDevice rank0(ec.dev[0]->device);
    auto& work                     = state.execution.work;
    const cudaStream_t stream      = state.execution.device.stream;
    const DraftParameters& weights = *state.execution.parameters.draft;
    qwen3_5::DFlashDecodeState& frame = state.frame;
    Tensor anchors    = frame.anchors.slice(0, 0, batch);
    Tensor frontiers  = frame.execution_frontiers.slice(0, 0, batch);
    Tensor candidates = frame.candidate_ids.slice(2, 0, batch);
    Tensor ids_flat   = candidates.view({dimension(config.dflash2->selector_top_k), mask_columns});
    Tensor scores =
        work.alloc(DType::FP32, {dimension(config.dflash2->selector_top_k), mask_columns});
    const ProposalParameters& head0 = *state.execution.parameters.proposal;
    if (head0.split()) {
        dflash2_candidates_split(final_hidden, {&head0, &*ranks[1].execution->parameters.proposal},
                                 false, ids_flat, scores,
                                 {&work, &ranks[1].execution->work}, ec, *tp.events);
    } else if (head0.token_ids) {
        ops::linear_topk(final_hidden[0], head0.head.weight, *head0.token_ids, ids_flat, scores,
                         work, stream);
    } else {
        ops::linear_topk(final_hidden[0], head0.head.weight,
                         dimension(state.execution.parameters.model.resources().public_token_count),
                         ids_flat, scores, work, stream);
    }
    Tensor projected =
        work.alloc(DType::BF16, {dimension(config.dflash2->selector_rank), mask_columns});
    project(final_hidden[0], weights.selector->hidden_projection, projected, work, stream);
    Tensor drafts     = frame.draft_tokens.slice(1, 0, batch);
    Tensor proposal_q = frame.proposal_q.slice(2, 0, batch);
    ops::candidate_selector_path(
        candidates, scores.view({dimension(config.dflash2->selector_top_k), k, batch}),
        projected.view({dimension(config.dflash2->selector_rank), k, batch}), anchors,
        weights.selector->predecessor_codebook, weights.selector->successor_codebook, frontiers,
        frame.sampling, drafts, proposal_q, work, stream);
    work.reset();
    ranks[1].execution->work.reset();
}

} // namespace

bool dflash2_drafter_split(const Parameters& parameters) {
    if (!parameters.draft || parameters.draft->layers.empty()) { return false; }
    const auto& draft = parameters.model.config().draft;
    return draft && parameters.draft->layers.front().context_key.weight.n <
                        static_cast<std::int32_t>(draft->attention.key_width());
}

void dflash2_draft_round_tp2(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                             DFlashEnvelopes envelopes) {
    const TpExecution* tp = state.execution.tp;
    if (tp == nullptr || tp->dflash == nullptr || state.peer_frame == nullptr ||
        !state.execution.parameters.model.config().draft->dflash2) {
        throw std::logic_error("DFlash2 tp2 round requires the split drafter on both ranks");
    }
    const std::int32_t width = static_cast<std::int32_t>(k) + 1;
    const auto ranks         = draft_ranks(state);
    each_rank(*tp->execution, [&](std::size_t r) {
        const DraftRank& d        = ranks[r];
        ExecutionCore& core       = *d.execution;
        const cudaStream_t stream = core.device.stream;
        core.work.reset();
        Tensor active_lanes   = d.frame->active_lanes.slice(0, 0, batch_size);
        Tensor context_starts = d.frame->context_frontiers.slice(0, 0, batch_size);
        Tensor frontiers      = d.frame->execution_frontiers.slice(0, 0, batch_size);
        Tensor positions      = d.frame->append_positions.slice(1, 0, batch_size);
        Tensor counts         = d.frame->append_counts.slice(0, 0, batch_size);
        Tensor destinations   = d.frame->state_destination_slots.slice(0, 0, batch_size);
        Tensor dflash_rows    = d.frame->dflash_kv_table_rows.slice(0, 0, batch_size);
        Tensor compact        = core.work.alloc(
            DType::BF16,
            {dimension(core.parameters.draft->feature_projection.weight.k), width, batch_size});
        ops::prepare_ragged_prefix(d.dflash->pending_features, active_lanes, context_starts,
                                   frontiers, compact, positions, counts, stream);
        DFlashAppendContext append{core, *d.dflash};
        append_context_impl(append, compact, positions, counts, destinations, dflash_rows,
                            envelopes.append);
    });
    nvtx::ScopedRange proposal_range(nvtx::Name::DFlashProposal, nvtx::Category::DFlash,
                                     static_cast<std::uint64_t>(k + 1U) * batch_size);
    propose_dflash2_batch_tp2(state, batch_size, static_cast<int>(k), envelopes);
}

void dflash_append_context_peer(DFlashPeerDrafter& peer, const qwen3_5::DFlashDecodeIngress& host,
                                std::int32_t batch, std::uint32_t draft_window,
                                ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    ExecutionCore& core = peer.execution;
    if (!core.io.dflash_decode) {
        throw std::logic_error("DFlash2 tp2: rank 1 has no DFlash decode frame");
    }
    const ScopedCurrentDevice rank1(core.device.device);
    qwen3_5::DFlashDecodeState& frame = *core.io.dflash_decode;
    const cudaStream_t stream         = core.device.stream;
    CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &host, sizeof(qwen3_5::DFlashDecodeIngress),
                               cudaMemcpyHostToDevice, stream));
    core.work.reset();
    Tensor features = core.work.alloc(
        DType::BF16, {dimension(core.parameters.draft->feature_projection.weight.k),
                      static_cast<std::int32_t>(draft_window + 1U), batch});
    Tensor positions = frame.append_positions.slice(1, 0, batch);
    Tensor counts    = frame.append_counts.slice(0, 0, batch);
    ops::prepare_ragged_prefix(peer.dflash.pending_features, frame.active_lanes.slice(0, 0, batch),
                               frame.context_frontiers.slice(0, 0, batch),
                               frame.execution_frontiers.slice(0, 0, batch), features, positions,
                               counts, stream);
    DFlashAppendContext append{core, peer.dflash};
    append_context_impl(append, features, positions, counts,
                        frame.state_destination_slots.slice(0, 0, batch),
                        frame.dflash_kv_table_rows.slice(0, 0, batch), envelope);
}

DFlashFeatureSink dflash_peer_batch_sink(DFlashPeerDrafter& peer, const Tensor& lanes,
                                         const Tensor& valid_columns, std::int32_t width,
                                         std::int32_t batch) {
    const auto& config = *peer.execution.parameters.model.config().draft;
    return DFlashFeatureSink{
        .batch_features      = &peer.dflash.pending_features,
        .batch_lanes         = &lanes,
        .batch_valid_columns = &valid_columns,
        .batch_width         = width,
        .batch_size          = batch,
        .layers              = std::span<const std::uint32_t>(config.target_layer_ids),
    };
}

DFlashFeatureSink dflash_peer_prefill_sink(DFlashPeerDrafter& peer,
                                           qwen3_5::DFlashPrefillIngress* host_ingress) {
    const auto& config = *peer.execution.parameters.model.config().draft;
    return DFlashFeatureSink{
        .features        = &peer.dflash.prefill_features,
        .positions       = &peer.dflash.prefill_positions,
        .layers          = std::span<const std::uint32_t>(config.target_layer_ids),
        .consume_prefill =
            [&peer, host_ingress](const Tensor& features, const Tensor& positions, bool) {
                ExecutionCore& core = peer.execution;
                if (!core.io.dflash_prefill || host_ingress == nullptr) {
                    throw std::logic_error("DFlash2 tp2: rank 1 prefill controls are unavailable");
                }
                auto& frame = *core.io.dflash_prefill;
                // Rank 0's consumer filled the host record with this chunk's bindings.
                CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, host_ingress,
                                           sizeof(qwen3_5::DFlashPrefillIngress),
                                           cudaMemcpyHostToDevice, core.device.stream));
                const auto exact = static_cast<std::uint32_t>(features.ne[1]);
                DFlashAppendContext append{core, peer.dflash};
                append_context_impl(append, features, positions, frame.append_count,
                                    frame.state_destination_slot, frame.full_kv_table_row,
                                    {exact, exact});
            },
    };
}

} // namespace ninfer::models::qwen3_5::execution
