#pragma once

#include <stdint.h>

#if defined(_WIN32)
#define OEQ_FFI_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define OEQ_FFI_EXPORT __attribute__((visibility("default")))
#else
#define OEQ_FFI_EXPORT
#endif

// Stable, Python-free view of JAX handlers.  ABI v2 adds the XLA state type
// required by staged instantiate/initialize/execute registrations.  Consumers
// must reject a table with an unknown ABI version or entry layout.
extern "C" {

#define OEQ_FFI_ABI_VERSION 2

struct OeqFfiType {
    // Passed to JAX once when the extension is registered, before a handler
    // can request this state through `Ctx<State<T>>`.
    const char* name;
    void* type_id;
    const void* type_info;
};

struct OeqFfiHandler {
    // Stages are separate because static kernel data is available at
    // instantiation, while the CUDA target is known only at initialization.
    const char* name;
    void* instantiate;
    void* initialize;
    void* execute;
};

struct OeqFfiHandlerTable {
    uint32_t abi_version;
    uint32_t type_count;
    const OeqFfiType* types;
    uint32_t handler_count;
    const OeqFfiHandler* handlers;
};

OEQ_FFI_EXPORT const OeqFfiHandlerTable* oeq_ffi_handler_table();

}
