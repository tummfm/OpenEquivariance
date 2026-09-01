#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

#ifdef CUDA_BACKEND
class CUJITKernel;
using JITKernel = CUJITKernel;
#endif

#ifdef HIP_BACKEND
class HIPJITKernel;
using JITKernel = HIPJITKernel;
#endif

template <typename JIT_IMPL>
class JITTPImpl;

template <typename JIT_IMPL>
class JITConvImpl;

struct FfiKernelProperties {
    int64_t L1_dim = 0;
    int64_t L2_dim = 0;
    int64_t L3_dim = 0;
    int64_t weight_numel = 0;
    bool shared_weights = false;
    ffi::DataType irrep_dtype = ffi::DataType::INVALID;
    ffi::DataType weight_dtype = ffi::DataType::INVALID;
    int64_t workspace_size = 0;
    bool deterministic = false;
    ffi::DataType idx_dtype = ffi::DataType::INVALID;
    ffi::DataType workspace_dtype = ffi::DataType::U8;
};

struct KernelCompilationStatistics {
    uint64_t interner_hits;
    uint64_t interner_misses;
    uint64_t compilations_started;
};

class SharedKernel {
public:
    virtual ~SharedKernel() = default;

    // Prepares work for this device. CUDA queues compilation; HIP loads
    // synchronously.
    virtual void initialize(int32_t device_ordinal) = 0;
};

// XLA owns this state, which holds only a shared pointer. The shared kernel
// owns the parsed plan, compilation work, and loaded launchers.
struct OeqExecutableState {
    static ffi::TypeId id;

    explicit OeqExecutableState(std::shared_ptr<SharedKernel> shared_kernel)
        : shared_kernel(std::move(shared_kernel)) {}

    std::shared_ptr<SharedKernel> shared_kernel;
};

using TensorProductKernel =
    std::pair<JITTPImpl<JITKernel>*, const FfiKernelProperties*>;
using ConvolutionKernel =
    std::pair<JITConvImpl<JITKernel>*, const FfiKernelProperties*>;

ffi::ErrorOr<std::unique_ptr<OeqExecutableState>> instantiate_tensor_product(
    std::string_view payload, int64_t hash);
ffi::ErrorOr<std::unique_ptr<OeqExecutableState>> instantiate_convolution(
    std::string_view payload, int64_t hash);

ffi::Error initialize_kernel_state(
    OeqExecutableState* state, int32_t device_ordinal);
ffi::ErrorOr<TensorProductKernel> tensor_product_kernel(
    OeqExecutableState* state, int32_t device_ordinal);
ffi::ErrorOr<ConvolutionKernel> convolution_kernel(
    OeqExecutableState* state, int32_t device_ordinal);
KernelCompilationStatistics kernel_compilation_statistics();
