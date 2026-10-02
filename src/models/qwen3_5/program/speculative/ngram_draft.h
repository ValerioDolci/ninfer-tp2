#pragma once
// Prompt-lookup drafting for the MTP decode round (experiment, 03/10/2026).
//
// When the last n tokens of a sequence's ledger already occurred earlier (prompt or generated
// text), the tokens that followed that earlier occurrence become the round's drafts in place of the
// MTP head's proposals. Verification is unchanged, so greedy output stays identical; only the
// acceptance rate (and thus tokens per round) changes. Enabled with NINFER_NGRAM=<n> (n in [1,8];
// unset or 0 = off). Host-only: one hash map per sequence, O(1) per committed token.
#include "ninfer/types.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <unordered_map>

namespace ninfer::models::qwen3_5::detail {

inline int ngram_draft_order() {
    static const int order = [] {
        const char* value = std::getenv("NINFER_NGRAM");
        const int n       = value ? std::atoi(value) : 0;
        return n < 1 ? 0 : std::min(n, 8);
    }();
    return order;
}

class NgramIndex {
public:
    void reset() {
        map_.clear();
        indexed_ = 0;
        order_   = 0;
    }

    // Drafts for the ledger's current tail: up to k tokens that followed the most recent earlier
    // occurrence of the last n tokens. Returns the number written to `out` (0 = no match).
    std::uint32_t propose(std::span<const TokenId> ledger, int n, std::uint32_t k,
                          std::span<TokenId> out) {
        const std::size_t size = ledger.size();
        if (n <= 0 || size < static_cast<std::size_t>(n) + 1 || k == 0) return 0;
        sync(ledger, n, size - 1);
        const auto it = map_.find(hash(ledger.subspan(size - n, n)));
        if (it == map_.end()) return 0;
        const std::size_t p = it->second; // position right after the earlier occurrence
        if (p >= size || p < static_cast<std::size_t>(n) ||
            !std::equal(ledger.begin() + static_cast<std::ptrdiff_t>(p - n),
                        ledger.begin() + static_cast<std::ptrdiff_t>(p),
                        ledger.begin() + static_cast<std::ptrdiff_t>(size - n))) {
            return 0;
        }
        const std::uint32_t count = static_cast<std::uint32_t>(
            std::min<std::size_t>({k, size - p, out.size()}));
        std::copy_n(ledger.begin() + static_cast<std::ptrdiff_t>(p), count, out.begin());
        return count;
    }

private:
    // Index every n-gram whose end (exclusive) lies in (indexed_, upto]; rebuild after a rewind.
    void sync(std::span<const TokenId> ledger, int n, std::size_t upto) {
        if (order_ != n || indexed_ > upto) {
            map_.clear();
            indexed_ = 0;
            order_   = n;
        }
        std::size_t end = std::max<std::size_t>(indexed_ + 1, static_cast<std::size_t>(n));
        for (; end <= upto; ++end) map_[hash(ledger.subspan(end - n, n))] = end;
        indexed_ = std::max(indexed_, upto);
    }

    static std::uint64_t hash(std::span<const TokenId> tokens) {
        std::uint64_t h = 1469598103934665603ULL;
        for (const TokenId t : tokens) {
            h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(t)) + 1U;
            h *= 1099511628211ULL;
        }
        return h;
    }

    std::unordered_map<std::uint64_t, std::size_t> map_;
    std::size_t indexed_ = 0;
    int order_           = 0;
};

} // namespace ninfer::models::qwen3_5::detail
