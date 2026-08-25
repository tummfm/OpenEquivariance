#pragma once

#include <stdint.h>

#if defined(_WIN32)
#define OEQ_FFI_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define OEQ_FFI_EXPORT __attribute__((visibility("default")))
#else
#define OEQ_FFI_EXPORT
#endif

// Stable, Python-free view of JAX handlers. Consumers must reject a table
// with an unknown ABI version or entry layout.
extern "C" {

#define OEQ_FFI_ABI_VERSION 1

struct OeqFfiHandler {
    const char* name;
    void* initialize;
    void* execute;
};

struct OeqFfiHandlerTable {
    uint32_t abi_version;
    uint32_t handler_count;
    const OeqFfiHandler* handlers;
};

OEQ_FFI_EXPORT const OeqFfiHandlerTable* oeq_ffi_handler_table();

}
