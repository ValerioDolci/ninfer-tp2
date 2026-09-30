// The [7168,5120] two-device shard's copy of upstream's nvfp4/nvfp4_attn_input_a4.cu: the same
// source, compiled with the shard's section output and entry points (nvfp4_attn_input_shard.h),
// behind one shard-only route choice.
#include "ops/attn_input_proj/tp2/nvfp4_attn_input_shard_prelude.h"

#include "ops/attn_input_proj/tp2/nvfp4_attn_input_shard_names.h"
#undef nvfp4_attn_input_a4_launch
#define nvfp4_attn_input_a4_launch nvfp4_attn_input_shard_a4_launch_upstream
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_a4.cu"

namespace ninfer::ops::detail {

// At the 1024-token prefill chunk the shard keeps the 128-token scale tiles of the TMA route,
// where upstream (and the parent) switch to 256 at T=1024: the shard's 56 row tiles give 224 CTAs
// at 256 tokens per tile, 3.2 waves on a 70-SM RTX 5070 Ti, and 448 at 128. Measured there:
// 196.5 -> 183.7 us; from T=1025 the 256-token tile is faster again (tune 2a596191). Every other
// T takes upstream's launcher unchanged.
void nvfp4_attn_input_shard_a4_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                      Tensor& gate, Tensor& k, Tensor& v,
                                      Nvfp4A4Workspace workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (tokens == 1024) {
        constexpr auto layout = Nvfp4ScaleLayout::Tiled128;
        launch_nvfp4_a4_quantize(x, weight, workspace, layout, stream);
        launch_nvfp4_a4_tma_attention_shard(
            nvfp4_a4_operands(weight, workspace, tokens, layout),
            static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(gate.data),
            static_cast<__nv_bfloat16*>(k.data), static_cast<__nv_bfloat16*>(v.data), stream);
        return;
    }
    nvfp4_attn_input_shard_a4_launch_upstream(x, weight, q, gate, k, v, workspace, stream);
}

} // namespace ninfer::ops::detail
