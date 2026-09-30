#pragma once

// Two-device (tp 2) forms of the DFlash2 drafter (execution/draft.cpp). Our file.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "models/qwen3_5/execution/parameters.h"
#include "ninfer/ops/allreduce.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::models::qwen3_5::execution {

// Vocabulary-split DFlash2 candidate ranking. `head[r]` is rank r's row block of the optimized
// proposal head with its block of the row -> token map (load/sharding.h: DFlash2 splits both by
// rows). Rank r ranks `hidden[r]` [H,M] over its block (ops::linear_topk_split) and packs its top
// sixteen (ops::topk_split_pack); one allreduce_sum of the packed candidates gives both ranks both
// lists, and rank 0 writes the complete top sixteen, in linear_topk's order and encoding, to `ids`
// I32 [16,M] and `scores` FP32 [16,M]. Each block row equals the whole head's row bit for bit and
// the keys carry global token ids, so the result equals ops::linear_topk over the whole head.
//
// With `broadcast` only rank 0 holds the hidden (the drafter runs on rank 0 alone): rank 1's
// `hidden[1]` is overwritten with rank 0's through one more allreduce_sum against zeros, whose
// FP32 combine of x and +0 returns x (a -0 becomes +0, which no product or sum can tell apart);
// rank 0's `hidden[0]` is rewritten in place with the same values. Without it both ranks already
// hold the same hidden. Rank r's scratch comes from `workspace[r]`, which must hold
// dflash2_candidates_split_workspace_bytes(M, broadcast).
void dflash2_candidates_split(const std::array<Tensor, 2>& hidden,
                              const std::array<const ProposalParameters*, 2>& head, bool broadcast,
                              Tensor& ids, Tensor& scores,
                              const std::array<WorkspaceArena*, 2>& workspace,
                              const ExecutionContext& execution, const ops::PeerEvents& events);

// One rank's scratch for dflash2_candidates_split over `columns` hidden columns of `hidden` rows.
[[nodiscard]] std::size_t dflash2_candidates_split_workspace_bytes(const ProposalParameters& head,
                                                                   std::int32_t hidden,
                                                                   std::int32_t columns,
                                                                   bool broadcast);

// Whether the DFlash2 drafter ranks its candidates through the vocabulary-split head: rank 1 holds
// its block of the optimized head.
[[nodiscard]] bool dflash2_split_candidates(const Parameters& rank0, const Parameters* rank1);

} // namespace ninfer::models::qwen3_5::execution
