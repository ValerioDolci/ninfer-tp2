#pragma once
#include "core/device.h"
#include "ops/launcher/kernel_attr_once.h"

namespace ninfer::ops::detail {
template <int Bytes>
__device__ __forceinline__ unsigned char* q8_shared_storage() {
    static_assert(Bytes <= 99 * 1024);
    if constexpr (Bytes > 48 * 1024) {
        extern __shared__ __align__(16) unsigned char dynamic_shared[];
        return dynamic_shared;
    } else {
        __shared__ __align__(16) unsigned char storage[Bytes];
        return storage;
    }
}

template <int Bytes, auto Kernel>
int q8_prepare_shared() {
    if constexpr (Bytes > 48 * 1024) {
        // Per device: a function attribute set only on the first device leaves the other rank at
        // the default 48 KiB (kernel_attr_once.h).
        static FuncAttrPerDevice attribute;
        attribute.ensure(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Bytes);
        return Bytes;
    } else
        return 0;
}
} // namespace ninfer::ops::detail
