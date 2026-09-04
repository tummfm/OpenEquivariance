#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

#include "xla/ffi/api/ffi.h"
#include "ffi_handler_table.h"
#include "kernel_compilation.h"

#ifdef CUDA_BACKEND
    #include <cuda.h>
    #include <cuda_runtime.h>

    #include "backend/backend_cuda.hpp"
    using stream_t = cudaStream_t;
#endif

#ifdef HIP_BACKEND
    #include "backend/backend_hip.hpp"
    using stream_t = hipStream_t;
#endif

#include "tensorproducts.hpp"
#include "convolution.hpp"

std::string xla_dtype_to_string(xla::ffi::DataType dtype) {
    const std::unordered_map<xla::ffi::DataType, std::string> map = {
        {xla::ffi::DataType::INVALID, "INVALID"},
        {xla::ffi::DataType::PRED, "PRED"},
        {xla::ffi::DataType::S8, "S8"},
        {xla::ffi::DataType::S16, "S16"},
        {xla::ffi::DataType::S32, "S32"},
        {xla::ffi::DataType::S64, "S64"},
        {xla::ffi::DataType::U8, "U8"},
        {xla::ffi::DataType::U16, "U16"},
        {xla::ffi::DataType::U32, "U32"},
        {xla::ffi::DataType::U64, "U64"},
        {xla::ffi::DataType::F16, "F16"},
        {xla::ffi::DataType::F32, "F32"},
        {xla::ffi::DataType::F64, "F64"},
        {xla::ffi::DataType::BF16, "BF16"},
        {xla::ffi::DataType::C64, "C64"},
        {xla::ffi::DataType::C128, "C128"},
        {xla::ffi::DataType::TOKEN, "TOKEN"},
        {xla::ffi::DataType::F8E5M2, "F8E5M2"},
        {xla::ffi::DataType::F8E4M3, "F8E4M3"},
        {xla::ffi::DataType::F8E4M3FN, "F8E4M3FN"},
        {xla::ffi::DataType::F8E4M3B11FNUZ, "F8E4M3B11FNUZ"},
        {xla::ffi::DataType::F8E5M2FNUZ, "F8E5M2FNUZ"},
        {xla::ffi::DataType::F8E4M3FNUZ, "F8E4M3FNUZ"},
        {xla::ffi::DataType::F8E3M4, "F8E3M4"},
        {xla::ffi::DataType::F4E2M1FN, "F4E2M1FN"},
        {xla::ffi::DataType::F8E8M0FNU, "F8E8M0FNU"},
    };
    return map.at(dtype);
}

inline void* data_ptr(ffi::AnyBuffer &buffer) {
    return buffer.untyped_data();
}

inline void* data_ptr(ffi::Result<ffi::AnyBuffer> &buffer) {
    return data_ptr(*buffer);
}

inline int byte_count(ffi::AnyBuffer &buffer) {
    switch (buffer.element_type()) {
        case xla::ffi::DataType::U32:
        case xla::ffi::DataType::S32:
        case xla::ffi::DataType::F32:
            return 4;
        case xla::ffi::DataType::F64:
        case xla::ffi::DataType::S64:
            return 8;
        case xla::ffi::DataType::U8:
            return 1;
        default:
            throw logic_error("Unsupported tensor datatype!");
    }
}

#ifdef CUDA_BACKEND
void zero_buffer(ffi::AnyBuffer &buffer, stream_t stream) {
    cudaMemsetAsync(
        data_ptr(buffer), 
        0, 
        buffer.element_count() * byte_count(buffer),
        stream);
}
#endif
#ifdef HIP_BACKEND
void zero_buffer(ffi::AnyBuffer &buffer, stream_t stream) {
    std::ignore = hipMemsetAsync(
        data_ptr(buffer), 
        0, 
        buffer.element_count() * byte_count(buffer),
        stream);
}
#endif

inline void check_tensor(const ffi::AnyBuffer &buffer, 
                            std::initializer_list<int64_t> expected_shape,
                            xla::ffi::DataType expected_dtype,
                            std::string tensor_name) {
    const ffi::AnyBuffer::Dimensions dims = buffer.dimensions();
    if (dims.size() != expected_shape.size()) {
        throw std::logic_error("Rank mismatch for tensor '"
            + tensor_name 
            + "'. Expected rank " 
            + std::to_string(expected_shape.size()) 
            + ", got rank " 
            + std::to_string(dims.size()));
    }

    for (size_t i = 0; i < dims.size(); i++) {
        if (dims[i] != expected_shape.begin()[i]) {
            throw std::logic_error("Shape mismatch for tensor '"
                + tensor_name 
                + "'. Expected dimension " 
                + std::to_string(expected_shape.begin()[i]) 
                + " at index " 
                + std::to_string(i) 
                + ", got " 
                + std::to_string(dims[i]));
        }
    }

    if (buffer.element_type() != expected_dtype) {
        throw std::logic_error("Datatype mismatch for tensor " + tensor_name +
            ". Expected datatype " + xla_dtype_to_string(expected_dtype) + 
            ", got " + xla_dtype_to_string(buffer.element_type()));
    }
}

// --------------------- Tensor Products --------------------------
ffi::Error tp_forward_impl(
        ffi::AnyBuffer L1_in,
        ffi::AnyBuffer L2_in,
        ffi::AnyBuffer W,
        ffi::Result<ffi::AnyBuffer> L3_out,
        stream_t stream,
        OeqExecutableState* state,
        int32_t device_ordinal,
        std::string_view kernel_json,
        int64_t hash) {
    (void)kernel_json;
    (void)hash;
    auto loaded = tensor_product_kernel(state, device_ordinal);
    if (!loaded) return loaded.error();
    auto [jit_kernel, properties] = *loaded;
    const FfiKernelProperties& k = *properties;
    const int64_t num_batch = L1_in.dimensions()[0];

    check_tensor(L1_in, {num_batch, k.L1_dim}, k.irrep_dtype, "L1_in");
    check_tensor(L2_in, {num_batch, k.L2_dim}, k.irrep_dtype, "L2_in"); 

    if (k.shared_weights)
        check_tensor(W, {k.weight_numel}, k.weight_dtype, "W");
    else 
        check_tensor(W, {num_batch, k.weight_numel}, k.weight_dtype, "W");

    jit_kernel->exec_tensor_product(
            num_batch,
            data_ptr(L1_in),
            data_ptr(L2_in),
            data_ptr(L3_out),
            data_ptr(W),
            stream);

    return ffi::Error::Success();
}

ffi::Error tp_backward_impl(
        ffi::AnyBuffer L1_in,
        ffi::AnyBuffer L2_in,
        ffi::AnyBuffer W,
        ffi::AnyBuffer L3_grad,
        ffi::Result<ffi::AnyBuffer> L1_grad,
        ffi::Result<ffi::AnyBuffer> L2_grad,
        ffi::Result<ffi::AnyBuffer> W_grad, 
        stream_t stream, 
        OeqExecutableState* state,
        int32_t device_ordinal,
        std::string_view kernel_json,
        int64_t hash) {
    (void)kernel_json;
    (void)hash;
    auto loaded = tensor_product_kernel(state, device_ordinal);
    if (!loaded) return loaded.error();
    auto [jit_kernel, properties] = *loaded;
    const FfiKernelProperties& k = *properties;
    const int64_t num_batch = L1_in.dimensions()[0];
    check_tensor(L1_in, {num_batch, k.L1_dim}, k.irrep_dtype, "L1_in");
    check_tensor(L2_in, {num_batch, k.L2_dim}, k.irrep_dtype, "L2_in");
    check_tensor(L3_grad, {num_batch, k.L3_dim}, k.irrep_dtype, "L3_grad");

    if (k.shared_weights) {
        check_tensor(W, {k.weight_numel}, k.weight_dtype, "W");
        check_tensor(*W_grad, {k.weight_numel}, k.weight_dtype, "W_grad");
    }
    else {
        check_tensor(W, {num_batch, k.weight_numel}, k.weight_dtype, "W");
        check_tensor(*W_grad, {num_batch, k.weight_numel}, k.weight_dtype, "W_grad");
    }

    zero_buffer(*L1_grad, stream);
    zero_buffer(*L2_grad, stream);
    zero_buffer(*W_grad, stream);

    jit_kernel->backward(
            num_batch,
            data_ptr(L1_in),
            data_ptr(L1_grad),
            data_ptr(L2_in),
            data_ptr(L2_grad),
            data_ptr(W),
            data_ptr(W_grad),
            data_ptr(L3_grad),
            stream);
    return ffi::Error::Success();
}


ffi::Error tp_double_backward_impl(
        ffi::AnyBuffer L1_in,
        ffi::AnyBuffer L2_in,
        ffi::AnyBuffer W,
        ffi::AnyBuffer L3_grad,
        ffi::AnyBuffer L1_dgrad,
        ffi::AnyBuffer L2_dgrad,
        ffi::AnyBuffer W_dgrad,
        ffi::Result<ffi::AnyBuffer> L1_grad,
        ffi::Result<ffi::AnyBuffer> L2_grad,
        ffi::Result<ffi::AnyBuffer> W_grad,
        ffi::Result<ffi::AnyBuffer> L3_dgrad,
        stream_t stream, 
        OeqExecutableState* state,
        int32_t device_ordinal,
        std::string_view kernel_json,
        int64_t hash) {
    (void)kernel_json;
    (void)hash;
    auto loaded = tensor_product_kernel(state, device_ordinal);
    if (!loaded) return loaded.error();
    auto [jit_kernel, properties] = *loaded;
    const FfiKernelProperties& k = *properties;
    const int64_t num_batch = L1_in.dimensions()[0];
    check_tensor(L1_in, {num_batch, k.L1_dim}, k.irrep_dtype, "L1_in");
    check_tensor(L2_in, {num_batch, k.L2_dim}, k.irrep_dtype, "L2_in");
    check_tensor(L3_grad, {num_batch, k.L3_dim}, k.irrep_dtype, "L3_grad");
    check_tensor(L1_dgrad, {num_batch, k.L1_dim}, k.irrep_dtype, "L1_dgrad");
    check_tensor(L2_dgrad, {num_batch, k.L2_dim}, k.irrep_dtype, "L2_dgrad");

    if (k.shared_weights){
        check_tensor(W, {k.weight_numel}, k.weight_dtype, "W");
        check_tensor(W_dgrad, {k.weight_numel}, k.weight_dtype, "W_dgrad");
    } else {
        check_tensor(W, {num_batch, k.weight_numel}, k.weight_dtype, "W");
        check_tensor(W_dgrad, {num_batch, k.weight_numel}, k.weight_dtype, "W_dgrad");
    }

    zero_buffer(*L1_grad, stream);
    zero_buffer(*L2_grad, stream);
    zero_buffer(*W_grad, stream);
    zero_buffer(*L3_dgrad, stream);

    jit_kernel->double_backward(
            num_batch,
            data_ptr(L1_in),
            data_ptr(L2_in),
            data_ptr(W),
            data_ptr(L3_grad),
            data_ptr(L1_dgrad),
            data_ptr(L2_dgrad),
            data_ptr(W_dgrad),
            data_ptr(L1_grad),
            data_ptr(L2_grad),
            data_ptr(W_grad),
            data_ptr(L3_dgrad),
            stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tp_forward, tp_forward_impl,
    ffi::Ffi::Bind<ffi::ExecutionStage::kExecute>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ctx<ffi::PlatformStream<stream_t>>()
        .Ctx<ffi::State<OeqExecutableState>>()
        .Ctx<ffi::DeviceOrdinal>()
        .Attr<std::string_view>("kernel")
        .Attr<int64_t>("hash"),
        {xla::ffi::Traits::kCmdBufferCompatible});  // cudaGraph enabled

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tp_backward, tp_backward_impl,
    ffi::Ffi::Bind<ffi::ExecutionStage::kExecute>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ctx<ffi::PlatformStream<stream_t>>()
        .Ctx<ffi::State<OeqExecutableState>>()
        .Ctx<ffi::DeviceOrdinal>()
        .Attr<std::string_view>("kernel")
        .Attr<int64_t>("hash"),
        {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tp_double_backward, tp_double_backward_impl,
    ffi::Ffi::Bind<ffi::ExecutionStage::kExecute>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ctx<ffi::PlatformStream<stream_t>>()
        .Ctx<ffi::State<OeqExecutableState>>()
        .Ctx<ffi::DeviceOrdinal>()
        .Attr<std::string_view>("kernel")
        .Attr<int64_t>("hash"),
        {xla::ffi::Traits::kCmdBufferCompatible});

// --------------------- Convolution --------------------------
ffi::Error conv_forward_impl(
        ffi::AnyBuffer L1_in,
        ffi::AnyBuffer L2_in,
        ffi::AnyBuffer W,
        ffi::AnyBuffer rows,
        ffi::AnyBuffer cols,
        ffi::AnyBuffer workspace,
        ffi::AnyBuffer transpose_perm,
        ffi::Result<ffi::AnyBuffer> L3_out,
        stream_t stream, 
        OeqExecutableState* state,
        int32_t device_ordinal,
        std::string_view kernel_json,
        int64_t hash) {
    (void)kernel_json;
    (void)hash;
    auto loaded = convolution_kernel(state, device_ordinal);
    if (!loaded) return loaded.error();
    auto [jit_kernel, properties] = *loaded;
    const FfiKernelProperties& k = *properties;
    const int64_t nnz = rows.dimensions()[0];
    const int64_t node_count = L1_in.dimensions()[0];
    void* workspace_ptr = data_ptr(workspace);

    check_tensor(L1_in, {node_count, k.L1_dim}, k.irrep_dtype, "L1_in");
    check_tensor(L2_in, {nnz, k.L2_dim}, k.irrep_dtype, "L2_in");
    check_tensor(workspace, {k.workspace_size}, k.workspace_dtype, "workspace");
    check_tensor(rows, {nnz}, k.idx_dtype, "rows");
    check_tensor(cols, {nnz}, k.idx_dtype, "cols");

    if (k.deterministic){
        check_tensor(transpose_perm, {nnz}, k.idx_dtype, "transpose perm");
    }
    else {
        workspace_ptr = nullptr;
    }
    zero_buffer(*L3_out, stream);

    if (k.shared_weights)
        check_tensor(W, {k.weight_numel}, k.weight_dtype, "W");
    else 
        check_tensor(W, {nnz, k.weight_numel}, k.weight_dtype, "W");

    jit_kernel->exec_conv(
            data_ptr(L1_in),
            data_ptr(L2_in),
            data_ptr(W),
            data_ptr(L3_out),
            data_ptr(rows),
            data_ptr(cols),
            nnz, node_count,
            workspace_ptr,
            stream);

    return ffi::Error::Success();
}

ffi::Error conv_backward_impl(
        ffi::AnyBuffer L1_in,
        ffi::AnyBuffer L2_in,
        ffi::AnyBuffer W,
        ffi::AnyBuffer L3_grad,
        ffi::Result<ffi::AnyBuffer> L1_grad,
        ffi::Result<ffi::AnyBuffer> L2_grad,
        ffi::Result<ffi::AnyBuffer> W_grad, 
        ffi::AnyBuffer rows,
        ffi::AnyBuffer cols,
        ffi::AnyBuffer workspace,
        ffi::AnyBuffer transpose_perm,
        stream_t stream, 
        OeqExecutableState* state,
        int32_t device_ordinal,
        std::string_view kernel_json,
        int64_t hash) {
    (void)kernel_json;
    (void)hash;
    auto loaded = convolution_kernel(state, device_ordinal);
    if (!loaded) return loaded.error();
    auto [jit_kernel, properties] = *loaded;
    const FfiKernelProperties& k = *properties;
    const int64_t nnz = rows.dimensions()[0];
    const int64_t node_count = L1_in.dimensions()[0];
    void* workspace_ptr = data_ptr(workspace);

    check_tensor(L1_in, {node_count, k.L1_dim}, k.irrep_dtype, "L1_in");
    check_tensor(L2_in, {nnz, k.L2_dim}, k.irrep_dtype, "L2_in");
    check_tensor(L3_grad, {node_count, k.L3_dim}, k.irrep_dtype, "L3_grad");
    check_tensor(workspace, {k.workspace_size}, k.workspace_dtype, "workspace");
    check_tensor(rows, {nnz}, k.idx_dtype, "rows");
    check_tensor(cols, {nnz}, k.idx_dtype, "cols");

    if (k.deterministic) {
        check_tensor(transpose_perm, {nnz}, k.idx_dtype, "transpose perm");
    }
    else {
        workspace_ptr = nullptr;
    }
    zero_buffer(*L1_grad, stream);
    zero_buffer(*L2_grad, stream);
    zero_buffer(*W_grad, stream);

    if (k.shared_weights) {
        check_tensor(W, {k.weight_numel}, k.weight_dtype, "W");
        check_tensor(*W_grad, {k.weight_numel}, k.weight_dtype, "W_grad");
    }
    else {
        check_tensor(W, {nnz, k.weight_numel}, k.weight_dtype, "W");
        check_tensor(*W_grad, {nnz, k.weight_numel}, k.weight_dtype, "W_grad");
    }

    jit_kernel->backward(
            data_ptr(L1_in),
            data_ptr(L1_grad),
            data_ptr(L2_in),
            data_ptr(L2_grad),
            data_ptr(W),
            data_ptr(W_grad),
            data_ptr(L3_grad),
            data_ptr(rows),
            data_ptr(cols),
            nnz, node_count,
            workspace_ptr,
            data_ptr(transpose_perm),
            stream);
    return ffi::Error::Success();
}

ffi::Error conv_double_backward_impl(
        ffi::AnyBuffer L1_in,
        ffi::AnyBuffer L2_in,
        ffi::AnyBuffer W,
        ffi::AnyBuffer L3_grad,
        ffi::AnyBuffer L1_dgrad,
        ffi::AnyBuffer L2_dgrad,
        ffi::AnyBuffer W_dgrad,
        ffi::Result<ffi::AnyBuffer> L1_grad,
        ffi::Result<ffi::AnyBuffer> L2_grad,
        ffi::Result<ffi::AnyBuffer> W_grad,
        ffi::Result<ffi::AnyBuffer> L3_dgrad,
        ffi::AnyBuffer rows,
        ffi::AnyBuffer cols,
        ffi::AnyBuffer workspace,
        ffi::AnyBuffer transpose_perm,
        stream_t stream, 
        OeqExecutableState* state,
        int32_t device_ordinal,
        std::string_view kernel_json,
        int64_t hash) {
    (void)kernel_json;
    (void)hash;
    auto loaded = convolution_kernel(state, device_ordinal);
    if (!loaded) return loaded.error();
    auto [jit_kernel, properties] = *loaded;
    const FfiKernelProperties& k = *properties;
    const int64_t nnz = rows.dimensions()[0];
    const int64_t node_count = L1_in.dimensions()[0];
    void* workspace_ptr = data_ptr(workspace);

    check_tensor(L1_in, {node_count, k.L1_dim}, k.irrep_dtype, "L1_in");
    check_tensor(L2_in, {nnz, k.L2_dim}, k.irrep_dtype, "L2_in");
    check_tensor(L3_grad, {node_count, k.L3_dim}, k.irrep_dtype, "L3_grad");
    check_tensor(L1_dgrad, {node_count, k.L1_dim}, k.irrep_dtype, "L1_dgrad");
    check_tensor(L2_dgrad, {nnz, k.L2_dim}, k.irrep_dtype, "L2_dgrad");
    check_tensor(workspace, {k.workspace_size}, k.workspace_dtype, "workspace");
    check_tensor(rows, {nnz}, k.idx_dtype, "rows");
    check_tensor(cols, {nnz}, k.idx_dtype, "cols");

    if (k.deterministic) {
        check_tensor(transpose_perm, {nnz}, k.idx_dtype, "transpose perm");
    }
    else {
        workspace_ptr = nullptr;
    }
    zero_buffer(*L1_grad, stream);
    zero_buffer(*L2_grad, stream);
    zero_buffer(*W_grad, stream);
    zero_buffer(*L3_dgrad, stream);
    
    if (k.shared_weights) {
        check_tensor(W, {k.weight_numel}, k.weight_dtype, "W");
        check_tensor(W_dgrad, {k.weight_numel}, k.weight_dtype, "W_dgrad");
    } else {
        check_tensor(W, {nnz, k.weight_numel}, k.weight_dtype, "W");
        check_tensor(W_dgrad, {nnz, k.weight_numel}, k.weight_dtype, "W_dgrad");
    }

    jit_kernel->double_backward(
            data_ptr(L1_in),
            data_ptr(L2_in),
            data_ptr(W),
            data_ptr(L3_grad),
            data_ptr(L1_dgrad),
            data_ptr(L2_dgrad),
            data_ptr(W_dgrad),
            data_ptr(L1_grad),
            data_ptr(L2_grad),
            data_ptr(W_grad),
            data_ptr(L3_dgrad),
            data_ptr(rows),
            data_ptr(cols),
            nnz, node_count,
            workspace_ptr,
            data_ptr(transpose_perm),
            stream);
    return ffi::Error::Success();
}

// --------------------- FFI Bindings --------------------------
ffi::ErrorOr<std::unique_ptr<OeqExecutableState>> tp_instantiate_impl(
    ffi::RemainingArgs, ffi::RemainingRets, std::string_view kernel_json,
    int64_t hash) {
    return instantiate_tensor_product(kernel_json, hash);
}

ffi::ErrorOr<std::unique_ptr<OeqExecutableState>> conv_instantiate_impl(
    ffi::RemainingArgs, ffi::RemainingRets, std::string_view kernel_json,
    int64_t hash) {
    return instantiate_convolution(kernel_json, hash);
}

ffi::Error initialize_impl(
    ffi::RemainingArgs, ffi::RemainingRets, OeqExecutableState* state,
    int32_t device_ordinal,
    std::string_view, int64_t) {
    return initialize_kernel_state(state, device_ordinal);
}

#define OEQ_KERNEL_ATTRIBUTES                                                          \
    .Attr<std::string_view>("kernel")                                                  \
        .Attr<int64_t>("hash")

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tp_instantiate, tp_instantiate_impl,
    ffi::Ffi::Bind<ffi::ExecutionStage::kInstantiate>()
        .RemainingArgs()
        .RemainingRets()
        OEQ_KERNEL_ATTRIBUTES);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    conv_instantiate, conv_instantiate_impl,
    ffi::Ffi::Bind<ffi::ExecutionStage::kInstantiate>()
        .RemainingArgs()
        .RemainingRets()
        OEQ_KERNEL_ATTRIBUTES);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tp_initialize, initialize_impl,
    ffi::Ffi::Bind<ffi::ExecutionStage::kInitialize>()
        .RemainingArgs()
        .RemainingRets()
        .Ctx<ffi::State<OeqExecutableState>>()
        .Ctx<ffi::DeviceOrdinal>()
        OEQ_KERNEL_ATTRIBUTES);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    conv_initialize, initialize_impl,
    ffi::Ffi::Bind<ffi::ExecutionStage::kInitialize>()
        .RemainingArgs()
        .RemainingRets()
        .Ctx<ffi::State<OeqExecutableState>>()
        .Ctx<ffi::DeviceOrdinal>()
        OEQ_KERNEL_ATTRIBUTES);

#undef OEQ_KERNEL_ATTRIBUTES

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    conv_forward, conv_forward_impl,
    ffi::Ffi::Bind<ffi::ExecutionStage::kExecute>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ctx<ffi::PlatformStream<stream_t>>()
        .Ctx<ffi::State<OeqExecutableState>>()
        .Ctx<ffi::DeviceOrdinal>()
        .Attr<std::string_view>("kernel")
        .Attr<int64_t>("hash"),
        {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    conv_backward, conv_backward_impl,
    ffi::Ffi::Bind<ffi::ExecutionStage::kExecute>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ctx<ffi::PlatformStream<stream_t>>()
        .Ctx<ffi::State<OeqExecutableState>>()
        .Ctx<ffi::DeviceOrdinal>()
        .Attr<std::string_view>("kernel")
        .Attr<int64_t>("hash"),
        {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    conv_double_backward, conv_double_backward_impl,
    ffi::Ffi::Bind<ffi::ExecutionStage::kExecute>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ctx<ffi::PlatformStream<stream_t>>()
        .Ctx<ffi::State<OeqExecutableState>>()
        .Ctx<ffi::DeviceOrdinal>()
        .Attr<std::string_view>("kernel")
        .Attr<int64_t>("hash"),
        {xla::ffi::Traits::kCmdBufferCompatible});

ffi::TypeId OeqExecutableState::id = {};

namespace {

#define OEQ_FFI_HANDLER(NAME, INSTANTIATE, INITIALIZE)                                \
    {#NAME, reinterpret_cast<void*>(INSTANTIATE), nullptr,                            \
     reinterpret_cast<void*>(INITIALIZE),                                              \
     reinterpret_cast<void*>(NAME), OEQ_FFI_TRAIT_COMMAND_BUFFER_COMPATIBLE}

const OeqFfiHandler kFfiHandlers[] = {
    OEQ_FFI_HANDLER(tp_forward, tp_instantiate, tp_initialize),
    OEQ_FFI_HANDLER(tp_backward, tp_instantiate, tp_initialize),
    OEQ_FFI_HANDLER(tp_double_backward, tp_instantiate, tp_initialize),
    OEQ_FFI_HANDLER(conv_forward, conv_instantiate, conv_initialize),
    OEQ_FFI_HANDLER(conv_backward, conv_instantiate, conv_initialize),
    OEQ_FFI_HANDLER(conv_double_backward, conv_instantiate, conv_initialize),
};

#undef OEQ_FFI_HANDLER

constexpr ffi::TypeInfo kOeqExecutableStateTypeInfo =
    ffi::MakeTypeInfo<OeqExecutableState>();

const OeqFfiType kFfiTypes[] = {{
    "oeq_executable_state",
    &OeqExecutableState::id,
    &kOeqExecutableStateTypeInfo,
}};

const OeqFfiHandlerTable kFfiHandlerTable = {
    OEQ_FFI_ABI_VERSION,
    sizeof(kFfiTypes) / sizeof(kFfiTypes[0]),
    kFfiTypes,
    sizeof(kFfiHandlers) / sizeof(kFfiHandlers[0]),
    kFfiHandlers,
};

}  // namespace

extern "C" const OeqFfiHandlerTable* oeq_ffi_handler_table() {
    return &kFfiHandlerTable;
}
