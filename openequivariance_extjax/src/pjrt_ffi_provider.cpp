#include <cstring>
#include <iostream>
#include <string>

#include "ffi_abi.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_extension.h"

namespace {

const PJRT_FFI* FindFfiExtension(const PJRT_Api* api) {
    if (api == nullptr) {
        return nullptr;
    }

    for (PJRT_Extension_Base* extension = api->extension_start;
         extension != nullptr;
         extension = extension->next) {
        if (extension->type == PJRT_Extension_Type_FFI) {
            return reinterpret_cast<const PJRT_FFI*>(extension);
        }
    }
    return nullptr;
}

int RegisterHandler(
        const PJRT_Api* api,
        const PJRT_FFI* ffi,
        const OeqFfiHandler& handler,
        const char* platform_name) {
    PJRT_FFI_Register_Handler_Args args{};
    args.struct_size = PJRT_FFI_Register_Handler_Args_STRUCT_SIZE;
    args.target_name = handler.name;
    args.target_name_size = std::strlen(handler.name);
    args.handler = handler.execute;
    args.platform_name = platform_name;
    args.platform_name_size = std::strlen(platform_name);
    args.traits = static_cast<PJRT_FFI_Handler_TraitsBits>(
        PJRT_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE);

    PJRT_Error* error = ffi->register_handler(&args);
    if (error == nullptr) {
        std::cout << "[OpenEquivariance] Registered FFI target "
                  << handler.name << " for " << platform_name << std::endl;
        return 0;
    }

    PJRT_Error_Message_Args message_args{};
    message_args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    message_args.error = error;
    api->PJRT_Error_Message(&message_args);
    const std::string message = message_args.message == nullptr
        ? "unknown PJRT error"
        : std::string(message_args.message, message_args.message_size);
    std::cerr << "OpenEquivariance failed to register FFI target "
              << handler.name << ": " << message << std::endl;

    PJRT_Error_Destroy_Args destroy_args{};
    destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    destroy_args.error = error;
    api->PJRT_Error_Destroy(&destroy_args);
    return 1;
}

}  // namespace

extern "C" OEQ_FFI_EXPORT int RegisterFFi(
        const PJRT_Api* api,
        const char* platform_name) {
    if (platform_name == nullptr) {
        return 1;
    }

    const PJRT_FFI* ffi = FindFfiExtension(api);
    if (ffi == nullptr || ffi->register_handler == nullptr) {
        return 1;
    }

    const OeqFfiHandlerTable* handlers = oeq_ffi_handler_table();
    if (handlers == nullptr || handlers->abi_version != OEQ_FFI_ABI_VERSION) {
        return 1;
    }
    for (uint32_t index = 0; index < handlers->handler_count; ++index) {
        if (RegisterHandler(api, ffi, handlers->handlers[index], platform_name) != 0) {
            return 1;
        }
    }
    return 0;
}
