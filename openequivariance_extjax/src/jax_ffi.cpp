#include <atomic>
#include <charconv>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#ifndef OEQ_NO_PYTHON_MODULE
#include "nanobind/nanobind.h"
#endif
#include "xla/ffi/api/ffi.h"
#include "json11/json11.hpp"
#include "ffi_abi.h"

#ifndef OEQ_NO_PYTHON_MODULE
namespace nb = nanobind;
#endif
namespace ffi = xla::ffi;
using json = json11::Json;

#ifdef CUDA_BACKEND
    #include <cuda.h>
    #include <cuda_runtime.h>

    #include "backend/backend_cuda.hpp"
    using JITKernel = CUJITKernel;
    using GPU_Allocator = CUDA_Allocator;
    using stream_t = cudaStream_t;
#endif

#ifdef HIP_BACKEND
    #include "backend/backend_hip.hpp"
    using JITKernel = HIPJITKernel;
    using GPU_Allocator = HIP_Allocator;
    using stream_t = hipStream_t;
#endif

#include "tensorproducts.hpp"
#include "convolution.hpp"
#include "factorized_projected.hpp"
#include "symmetric_literal.hpp"

xla::ffi::DataType enum_to_xla_dtype(int64_t i){
    switch(i) {
        case 1:
            return xla::ffi::DataType::F32; 
        case 2: 
            return xla::ffi::DataType::F64;
        case 3: 
            return xla::ffi::DataType::S32;
        case 4: 
            return xla::ffi::DataType::S64;
        case 5: 
            return xla::ffi::DataType::U8;
    }
    throw logic_error("Unsupported tensor datatype!");
}

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

std::unordered_map<std::string, int64_t> parse_json_config(const json &j_obj) {
    std::unordered_map<std::string, int64_t> result;
    for (const auto &kv : j_obj.object_items()) {
        result[kv.first] = static_cast<int64_t>(kv.second.number_value());
    }
    return result;
}

struct KernelProp {
    int64_t L1_dim = 0, L2_dim = 0, L3_dim = 0, weight_numel = 0;
    bool shared_weights = false;
    xla::ffi::DataType irrep_dtype = xla::ffi::DataType::INVALID;
    xla::ffi::DataType weight_dtype = xla::ffi::DataType::INVALID;

    int64_t workspace_size = 0;     // Convolution only
    bool deterministic = false;
    xla::ffi::DataType idx_dtype = xla::ffi::DataType::INVALID;
    xla::ffi::DataType workspace_dtype = xla::ffi::DataType::INVALID;

    KernelProp() {}

    KernelProp(
        const std::unordered_map<string, int64_t> &kernel_dims,
        bool is_convolution):
            L1_dim(kernel_dims.at("L1_dim")),
            L2_dim(kernel_dims.at("L2_dim")),    
            L3_dim(kernel_dims.at("L3_dim")),
            weight_numel(kernel_dims.at("weight_numel")),
            shared_weights(kernel_dims.at("shared_weights")),
            irrep_dtype(enum_to_xla_dtype(kernel_dims.at("irrep_dtype"))),
            weight_dtype(enum_to_xla_dtype(kernel_dims.at("weight_dtype"))),
            workspace_dtype(xla::ffi::DataType::U8) { 
        if(is_convolution) {
            workspace_size = kernel_dims.at("workspace_size");
            deterministic = kernel_dims.at("deterministic");
            idx_dtype = enum_to_xla_dtype(kernel_dims.at("idx_dtype"));
        }
    }
};

namespace {

// Selects the dense tensor-product or sparse-convolution payload layout.
enum class KernelFamily {
    kTensorProduct,
    kConvolution,
    kFactorizedProjected,
    kSymmetricLiteral,
};

// Static JSON decoded and validated once at instantiation. It contains all
// source and launch data needed to create a device-specific launcher later.
struct ParsedKernel {
    std::string source;
    KernelLaunchConfig forward_config;
    KernelLaunchConfig backward_config;
    KernelLaunchConfig double_backward_config;
    KernelProp properties;
    int opt_level;
};

KernelLaunchConfig parse_launch_config(const json& config) {
    auto values = parse_json_config(config);
    return KernelLaunchConfig(
        values.at("num_blocks"), values.at("num_threads"), values.at("smem"));
}

ParsedKernel parse_kernel(std::string_view payload, KernelFamily family) {
    if (family == KernelFamily::kFactorizedProjected ||
        family == KernelFamily::kSymmetricLiteral) {
        return {std::string(payload), {}, {}, {}, {}, 3};
    }

    std::string error;
    json root = json::parse(std::string(payload), error);
    if (!error.empty()) {
        throw std::runtime_error("JSON Parse Error: " + error);
    }

    auto dimensions = parse_json_config(root["kernel_prop"]);
    return {
        root["kernel"].string_value(),
        parse_launch_config(root["forward_config"]),
        parse_launch_config(root["backward_config"]),
        parse_launch_config(root["double_backward_config"]),
        KernelProp(dimensions, family == KernelFamily::kConvolution),
        static_cast<int>(dimensions.at("opt_level")),
    };
}

#ifdef CUDA_BACKEND

int compiler_worker_count() {
    // Read this setting once when the pool is created.
    // The default is 16. The accepted range is 1--64.
    constexpr int kDefaultWorkerCount = 16;
    constexpr int kMaximumWorkerCount = 64;
    const char* setting = std::getenv("OEQ_JAX_COMPILER_THREADS");
    if (setting == nullptr) {
        return kDefaultWorkerCount;
    }

    std::string_view value(setting);
    int worker_count;
    const auto [end, error] = std::from_chars(
        value.data(), value.data() + value.size(), worker_count);
    if (error != std::errc() || end != value.data() + value.size() ||
        worker_count < 1 || worker_count > kMaximumWorkerCount) {
        throw std::invalid_argument(
            "OEQ_JAX_COMPILER_THREADS must be an integer from 1 to 64");
    }
    return worker_count;
}

// Bounded host pool for overlapping independent NVRTC compilations.
class CompilerPool {
public:
    static CompilerPool& instance() {
        static CompilerPool pool;
        return pool;
    }

    void submit(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.push_back(std::move(task));
        }
        ready_.notify_one();
    }

private:
    CompilerPool() {
        const int worker_count = compiler_worker_count();
        try {
            for (int index = 0; index < worker_count; ++index) {
                workers_.emplace_back([this] { run(); });
            }
        } catch (...) {
            stop();
            throw;
        }
    }

    ~CompilerPool() {
        stop();
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    void run() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this] {
                    return stopping_ || !tasks_.empty();
                });
                if (stopping_ && tasks_.empty()) {
                    return;
                }
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            task();
        }
    }

    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::function<void()>> tasks_;
    std::vector<std::thread> workers_;
    bool stopping_ = false;
};

using CompiledImageFuture =
    std::shared_future<CompiledKernelImage>;

using LoadedKernel = std::variant<
    std::unique_ptr<JITTPImpl<JITKernel>>,
    std::unique_ptr<JITConvImpl<JITKernel>>,
    std::unique_ptr<JITFactorizedProjectedImpl<JITKernel>>,
    std::unique_ptr<JITSymmetricLiteralImpl<JITKernel>>>;

// Shared compiled image and loaded launcher for one compute capability.
struct ArchitectureKernel {
    ArchitectureKernel(int architecture, CompiledImageFuture image)
        : architecture(architecture), image(std::move(image)) {}

    const int architecture;
    CompiledImageFuture image;
    // The context-less CUDA library is loaded once per compiled architecture.
    // The image future also carries a compilation error to every operation
    // sharing this entry.
    std::mutex load_mutex;
    std::unique_ptr<LoadedKernel> loaded;
};

int compute_capability(int32_t device_ordinal) {
    // The FFI supplies the target ordinal. Do not infer it from the current
    // CUDA device. Initialization may run outside the eventual launch context.
    CUdevice device;
    if (cuDeviceGet(&device, device_ordinal) != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to resolve the CUDA device ordinal");
    }
    int major;
    int minor;
    if (cuDeviceGetAttribute(
            &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) !=
            CUDA_SUCCESS ||
        cuDeviceGetAttribute(
            &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) !=
            CUDA_SUCCESS) {
        throw std::runtime_error("Failed to query CUDA compute capability");
    }
    return major * 10 + minor;
}

#endif

// Parsed plan and its compute-capability-specific CUDA artifacts.
struct SharedKernel {
    KernelFamily family;
    ParsedKernel parsed;

    SharedKernel(KernelFamily family, ParsedKernel parsed)
        : family(family), parsed(std::move(parsed)) {}

#ifdef CUDA_BACKEND
    std::unique_ptr<LoadedKernel> make_loaded_kernel(
        const CompiledKernelImage& image) const {
        switch (family) {
            case KernelFamily::kTensorProduct:
                return std::make_unique<LoadedKernel>(std::in_place_index<0>,
                    std::make_unique<JITTPImpl<JITKernel>>(
                        image, parsed.forward_config, parsed.backward_config,
                        parsed.double_backward_config, parsed.opt_level));
            case KernelFamily::kConvolution:
                return std::make_unique<LoadedKernel>(std::in_place_index<1>,
                    std::make_unique<JITConvImpl<JITKernel>>(
                        image, parsed.forward_config, parsed.backward_config,
                        parsed.double_backward_config, parsed.opt_level));
            case KernelFamily::kFactorizedProjected:
                return std::make_unique<LoadedKernel>(std::in_place_index<2>,
                    std::make_unique<JITFactorizedProjectedImpl<JITKernel>>(image));
            case KernelFamily::kSymmetricLiteral:
                return std::make_unique<LoadedKernel>(std::in_place_index<3>,
                    std::make_unique<JITSymmetricLiteralImpl<JITKernel>>(image));
        }
        throw std::logic_error("unknown kernel family");
    }

    ArchitectureKernel* schedule(int architecture) {
        std::lock_guard<std::mutex> lock(compilation_mutex);
        auto found = compilations.find(architecture);
        if (found != compilations.end()) {
            return found->second.get();
        }

        // Publish the future before queueing work so equivalent executables
        // share both the in-flight compilation and a possible sticky error.
        auto promise = std::make_shared<std::promise<CompiledKernelImage>>();
        CompiledImageFuture compilation = promise->get_future().share();
        auto compiled =
            std::make_unique<ArchitectureKernel>(architecture, compilation);
        ArchitectureKernel* result = compiled.get();
        compilations.emplace(architecture, std::move(compiled));

        // The worker produces only a context-independent CUDA image. Library
        // loading and handle resolution remain on the execute path. CUkernel
        // launch later selects JAX's context from its stream.
        CompilerPool::instance().submit([this, architecture, promise] {
            try {
                const int major = architecture / 10;
                const int minor = architecture % 10;
                CompiledKernelImage image = [&] {
                    switch (family) {
                        case KernelFamily::kTensorProduct:
                            return JITTPImpl<JITKernel>::compile_image(
                                parsed.source, major, minor, parsed.opt_level);
                        case KernelFamily::kConvolution:
                            return JITConvImpl<JITKernel>::compile_image(
                                parsed.source, major, minor, parsed.opt_level);
                        case KernelFamily::kFactorizedProjected:
                            return JITFactorizedProjectedImpl<JITKernel>::compile_image(
                                parsed.source, major, minor);
                        case KernelFamily::kSymmetricLiteral:
                            return JITSymmetricLiteralImpl<JITKernel>::compile_image(
                                parsed.source, major, minor);
                    }
                    throw std::logic_error("unknown kernel family");
                }();
                promise->set_value(std::move(image));
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        });
        return result;
    }

    ArchitectureKernel* load(int architecture) {
        ArchitectureKernel* compiled = schedule(architecture);
        std::lock_guard<std::mutex> lock(compiled->load_mutex);
        if (!compiled->loaded) {
            // This is the cold-path wait. Once ready, the same loaded launcher
            // is reused by all executable states for this compute capability.
            const CompiledKernelImage& image = compiled->image.get();
            compiled->loaded = make_loaded_kernel(image);
        }
        return compiled;
    }

    std::mutex compilation_mutex;
    std::unordered_map<int, std::unique_ptr<ArchitectureKernel>> compilations;
#endif
};

// One entry in a user-supplied-hash bucket. The payload remains here so the
// interner can reject collisions rather than trusting the hash as identity.
struct CachedKernel {
    KernelFamily family;
    std::string payload;
    std::shared_ptr<SharedKernel> kernel;
};

std::mutex interner_mutex;
std::unordered_map<int64_t, std::vector<CachedKernel>> kernel_interner;

std::shared_ptr<SharedKernel> intern_kernel(
    KernelFamily family, std::string_view payload, int64_t hash) {
    std::lock_guard<std::mutex> lock(interner_mutex);
    // `hash` is an interning bucket only. The complete payload and family are
    // compared before sharing state. A hash collision cannot select a
    // different kernel.
    auto& bucket = kernel_interner[hash];
    for (const CachedKernel& cached : bucket) {
        if (cached.family == family && cached.payload == payload) {
            return cached.kernel;
        }
    }

    // Instantiation parses the static JSON once. XLA then owns a typed state
    // referring to the immutable parsed description.
    auto kernel = std::make_shared<SharedKernel>(
        family, parse_kernel(payload, family));
    bucket.push_back({family, std::string(payload), kernel});
    return kernel;
}

// XLA-owned handle into the process cache, with a warm target lookup.
struct OeqExecutableState {
    static ffi::TypeId id;

    explicit OeqExecutableState(std::shared_ptr<SharedKernel> kernel)
        : kernel(std::move(kernel)) {}

    std::shared_ptr<SharedKernel> kernel;

#ifdef CUDA_BACKEND
    static uint64_t target_key(int32_t device_ordinal, int architecture) {
        return (static_cast<uint64_t>(
                    static_cast<uint32_t>(device_ordinal)) << 32) |
               static_cast<uint32_t>(architecture);
    }

    static int32_t target_device(uint64_t target) {
        return static_cast<int32_t>(target >> 32);
    }

    static int target_architecture(uint64_t target) {
        return static_cast<int32_t>(target);
    }

    void initialize(int32_t device_ordinal) {
        // XLA may call initialization for every execution. The acquire/release
        // pair makes the repeated-ordinal path a lock-free no-op after its
        // compile has been enqueued.
        const uint64_t current =
            initialized_target.load(std::memory_order_acquire);
        if (target_device(current) == device_ordinal) {
            return;
        }
        const int architecture = compute_capability(device_ordinal);
        // Initialization only enqueues host compilation. Execute waits only if
        // the image is still cold.
        kernel->schedule(architecture);
        initialized_target.store(
            target_key(device_ordinal, architecture),
            std::memory_order_release);
    }

    ArchitectureKernel* load(int32_t device_ordinal) {
        const uint64_t current =
            initialized_target.load(std::memory_order_acquire);
        const int architecture = target_device(current) == device_ordinal
            ? target_architecture(current)
            : compute_capability(device_ordinal);
        ArchitectureKernel* recent =
            recent_kernel.load(std::memory_order_acquire);
        // The architecture is immutable, so one published pointer is a
        // race-free warm cache even when devices execute concurrently.
        if (recent != nullptr && recent->architecture == architecture) {
            return recent;
        }
        ArchitectureKernel* loaded = kernel->load(architecture);
        recent_kernel.store(loaded, std::memory_order_release);
        return loaded;
    }

    std::atomic<uint64_t> initialized_target{UINT64_MAX};
    std::atomic<ArchitectureKernel*> recent_kernel{nullptr};
#else
    void initialize(int32_t) {
        std::lock_guard<std::mutex> lock(load_mutex);
        if (tensor_product || convolution || factorized_projected ||
            symmetric_literal) {
            return;
        }
        const ParsedKernel& parsed = kernel->parsed;
        switch (kernel->family) {
            case KernelFamily::kTensorProduct:
                tensor_product = std::make_unique<JITTPImpl<JITKernel>>(
                    parsed.source, parsed.forward_config, parsed.backward_config,
                    parsed.double_backward_config, parsed.opt_level);
                break;
            case KernelFamily::kConvolution:
                convolution = std::make_unique<JITConvImpl<JITKernel>>(
                    parsed.source, parsed.forward_config, parsed.backward_config,
                    parsed.double_backward_config, parsed.opt_level);
                break;
            case KernelFamily::kFactorizedProjected:
                factorized_projected =
                    std::make_unique<JITFactorizedProjectedImpl<JITKernel>>(
                        parsed.source);
                break;
            case KernelFamily::kSymmetricLiteral:
                symmetric_literal =
                    std::make_unique<JITSymmetricLiteralImpl<JITKernel>>(
                        parsed.source);
                break;
        }
    }

    std::mutex load_mutex;
    std::unique_ptr<JITTPImpl<JITKernel>> tensor_product;
    std::unique_ptr<JITConvImpl<JITKernel>> convolution;
    std::unique_ptr<JITFactorizedProjectedImpl<JITKernel>> factorized_projected;
    std::unique_ptr<JITSymmetricLiteralImpl<JITKernel>> symmetric_literal;
#endif
};

ffi::TypeId OeqExecutableState::id = {};
constexpr ffi::TypeInfo kOeqExecutableStateTypeInfo =
    ffi::MakeTypeInfo<OeqExecutableState>();

ffi::ErrorOr<std::unique_ptr<OeqExecutableState>> instantiate_kernel(
    KernelFamily family, std::string_view payload, int64_t hash) {
    try {
        // XLA owns the lightweight executable state. The process cache shares
        // parsed plans and compiled artifacts.
        return std::make_unique<OeqExecutableState>(
            intern_kernel(family, payload, hash));
    } catch (const std::exception& error) {
        return ffi::Unexpected(ffi::Error::InvalidArgument(error.what()));
    }
}

// Family-specific launcher plus validated static dimensions.
using TensorProductKernel =
    std::pair<JITTPImpl<JITKernel>*, const KernelProp*>;
using ConvolutionKernel =
    std::pair<JITConvImpl<JITKernel>*, const KernelProp*>;

ffi::ErrorOr<TensorProductKernel> tensor_product_kernel(
    OeqExecutableState* state, int32_t device_ordinal) {
    try {
#ifdef CUDA_BACKEND
        ArchitectureKernel* compiled = state->load(device_ordinal);
        LoadedKernel& loaded = *compiled->loaded;
        return TensorProductKernel{
            std::get<std::unique_ptr<JITTPImpl<JITKernel>>>(loaded).get(),
            &state->kernel->parsed.properties};
#else
        state->initialize(device_ordinal);
        return TensorProductKernel{
            state->tensor_product.get(), &state->kernel->parsed.properties};
#endif
    } catch (const std::exception& error) {
        // `future::get()` rethrows NVRTC failures here. Convert them to the
        // FFI error channel instead of allowing an exception through XLA.
        return ffi::Unexpected(ffi::Error::Internal(error.what()));
    }
}

ffi::ErrorOr<ConvolutionKernel> convolution_kernel(
    OeqExecutableState* state, int32_t device_ordinal) {
    try {
#ifdef CUDA_BACKEND
        ArchitectureKernel* compiled = state->load(device_ordinal);
        LoadedKernel& loaded = *compiled->loaded;
        return ConvolutionKernel{
            std::get<std::unique_ptr<JITConvImpl<JITKernel>>>(loaded).get(),
            &state->kernel->parsed.properties};
#else
        state->initialize(device_ordinal);
        return ConvolutionKernel{
            state->convolution.get(), &state->kernel->parsed.properties};
#endif
    } catch (const std::exception& error) {
        // Compilation errors are shared by all users of the future and
        // reported through the FFI boundary.
        return ffi::Unexpected(ffi::Error::Internal(error.what()));
    }
}

ffi::ErrorOr<JITFactorizedProjectedImpl<JITKernel>*>
factorized_projected_kernel(
    OeqExecutableState* state, int32_t device_ordinal) {
    try {
#ifdef CUDA_BACKEND
        LoadedKernel& loaded = *state->load(device_ordinal)->loaded;
        return std::get<
            std::unique_ptr<JITFactorizedProjectedImpl<JITKernel>>>(loaded).get();
#else
        state->initialize(device_ordinal);
        return state->factorized_projected.get();
#endif
    } catch (const std::exception& error) {
        return ffi::Unexpected(ffi::Error::Internal(error.what()));
    }
}

ffi::ErrorOr<JITSymmetricLiteralImpl<JITKernel>*> symmetric_literal_kernel(
    OeqExecutableState* state, int32_t device_ordinal) {
    try {
#ifdef CUDA_BACKEND
        LoadedKernel& loaded = *state->load(device_ordinal)->loaded;
        return std::get<
            std::unique_ptr<JITSymmetricLiteralImpl<JITKernel>>>(loaded).get();
#else
        state->initialize(device_ordinal);
        return state->symmetric_literal.get();
#endif
    } catch (const std::exception& error) {
        return ffi::Unexpected(ffi::Error::Internal(error.what()));
    }
}

}  // namespace

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

// JAX lowers an inactive tangent to a rank-zero buffer.  Generated JVP kernels
// render these operands as scalar_t(0), so their data pointer is never read.
inline void check_tensor_or_scalar_placeholder(
    const ffi::AnyBuffer &buffer, std::initializer_list<int64_t> expected_shape,
    xla::ffi::DataType expected_dtype, std::string tensor_name) {
    if (buffer.dimensions().size() == 0) {
        check_tensor(buffer, {}, expected_dtype, std::move(tensor_name));
        return;
    }
    check_tensor(buffer, expected_shape, expected_dtype, std::move(tensor_name));
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
    const KernelProp& k = *properties;
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
    const KernelProp& k = *properties;
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
    const KernelProp& k = *properties;
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
    ffi::Ffi::Bind()
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
    ffi::Ffi::Bind()
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
    ffi::Ffi::Bind()
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
    const KernelProp& k = *properties;
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
    const KernelProp& k = *properties;
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
    const KernelProp& k = *properties;
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

bool is_hip() {
#ifdef HIP_BACKEND
    return true;
#else
    return false;
#endif
}

// --------------------- FFI Bindings --------------------------
// Instantiation interns static state. Initialization queues NVRTC work.
// Execution loads a cold image if needed and launches it.

ffi::ErrorOr<std::unique_ptr<OeqExecutableState>> tp_instantiate_impl(
    std::string_view kernel_json, int64_t hash) {
    return instantiate_kernel(KernelFamily::kTensorProduct, kernel_json, hash);
}

ffi::ErrorOr<std::unique_ptr<OeqExecutableState>> conv_instantiate_impl(
    std::string_view kernel_json, int64_t hash) {
    return instantiate_kernel(KernelFamily::kConvolution, kernel_json, hash);
}

ffi::Error initialize_impl(
    OeqExecutableState* state, int32_t device_ordinal,
    std::string_view, int64_t) {
    try {
        state->initialize(device_ordinal);
        return ffi::Error::Success();
    } catch (const std::exception& error) {
        return ffi::Error::Internal(error.what());
    }
}

#define OEQ_KERNEL_ATTRIBUTES                                                         \
    .Attr<std::string_view>("kernel")                                                \
        .Attr<int64_t>("hash")

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tp_instantiate, tp_instantiate_impl,
    ffi::Ffi::BindInstantiate() OEQ_KERNEL_ATTRIBUTES);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    conv_instantiate, conv_instantiate_impl,
    ffi::Ffi::BindInstantiate() OEQ_KERNEL_ATTRIBUTES);

#define OEQ_INITIALIZE_BINDING                                                        \
    ffi::Ffi::BindInitialize()                                                       \
        .Ctx<ffi::State<OeqExecutableState>>()                                       \
        .Ctx<ffi::DeviceOrdinal>()                                                   \
        OEQ_KERNEL_ATTRIBUTES

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tp_initialize, initialize_impl, OEQ_INITIALIZE_BINDING);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    conv_initialize, initialize_impl, OEQ_INITIALIZE_BINDING);

#undef OEQ_INITIALIZE_BINDING
#undef OEQ_KERNEL_ATTRIBUTES

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    conv_forward, conv_forward_impl,
    ffi::Ffi::Bind()
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
    ffi::Ffi::Bind()
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
    ffi::Ffi::Bind()
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

// ------------------- Generated factorized convolution -------------------
void validate_projected_inputs(ffi::AnyBuffer &x, ffi::AnyBuffer &sh,
                               ffi::AnyBuffer &senders, int64_t input_dim,
                               int64_t edge_dim) {
    if (x.dimensions().size() != 2 || sh.dimensions().size() != 2) {
        throw std::logic_error("projected factorized inputs must have rank two");
    }
    if (x.element_type() != xla::ffi::DataType::F32 &&
        x.element_type() != xla::ffi::DataType::F64) {
        throw std::logic_error("projected factorized kernels support only f32 and f64");
    }
    const int64_t edges = sh.dimensions()[0];
    check_tensor(x, {x.dimensions()[0], input_dim}, x.element_type(), "x");
    check_tensor(sh, {edges, edge_dim}, x.element_type(), "sh");
    check_tensor(senders, {edges}, xla::ffi::DataType::S32, "senders");
}

ffi::Error factorized_projected_forward_impl(
    ffi::AnyBuffer x, ffi::AnyBuffer sh, ffi::AnyBuffer weights, ffi::AnyBuffer senders,
    ffi::AnyBuffer row_ptr, ffi::Result<ffi::AnyBuffer> out, stream_t stream,
    OeqExecutableState* state, int32_t device_ordinal,
    std::string_view source, int64_t hash, int64_t channels,
    int64_t input_dim, int64_t edge_dim, int64_t weight_dim, int64_t output_dim) {
    validate_projected_inputs(x, sh, senders, input_dim, edge_dim);
    const int64_t nodes = x.dimensions()[0], edges = sh.dimensions()[0];
    check_tensor(weights, {edges, weight_dim}, x.element_type(), "weights");
    check_tensor(row_ptr, {nodes + 1}, xla::ffi::DataType::S32, "row_ptr");
    check_tensor(*out, {nodes, output_dim}, x.element_type(), "out");
    auto loaded = factorized_projected_kernel(state, device_ordinal);
    if (!loaded) return loaded.error();
    auto* jit_kernel = *loaded;
    jit_kernel->forward(
        nodes, edges, channels, data_ptr(x), data_ptr(sh), data_ptr(weights),
        data_ptr(senders), data_ptr(row_ptr), data_ptr(out), stream);
    return ffi::Error::Success();
}

ffi::Error factorized_projected_forward_jvp_impl(
    ffi::AnyBuffer x, ffi::AnyBuffer sh, ffi::AnyBuffer weights, ffi::AnyBuffer senders,
    ffi::AnyBuffer row_ptr, ffi::AnyBuffer tx, ffi::AnyBuffer tsh,
    ffi::AnyBuffer tweights, ffi::Result<ffi::AnyBuffer> out, stream_t stream,
    OeqExecutableState* state, int32_t device_ordinal,
    std::string_view source, int64_t hash, int64_t channels,
    int64_t input_dim, int64_t edge_dim, int64_t weight_dim, int64_t output_dim) {
    validate_projected_inputs(x, sh, senders, input_dim, edge_dim);
    const int64_t nodes = x.dimensions()[0], edges = sh.dimensions()[0];
    check_tensor(weights, {edges, weight_dim}, x.element_type(), "weights");
    check_tensor(row_ptr, {nodes + 1}, xla::ffi::DataType::S32, "row_ptr");
    check_tensor_or_scalar_placeholder(
        tx, {nodes, input_dim}, x.element_type(), "tx");
    check_tensor_or_scalar_placeholder(
        tsh, {edges, edge_dim}, x.element_type(), "tsh");
    check_tensor_or_scalar_placeholder(
        tweights, {edges, weight_dim}, x.element_type(), "tweights");
    check_tensor(*out, {nodes, output_dim}, x.element_type(), "out");
    auto loaded = factorized_projected_kernel(state, device_ordinal);
    if (!loaded) return loaded.error();
    auto* jit_kernel = *loaded;
    jit_kernel->forward_jvp(
        nodes, edges, channels, data_ptr(x), data_ptr(sh), data_ptr(weights),
        data_ptr(senders), data_ptr(row_ptr), data_ptr(tx), data_ptr(tsh),
        data_ptr(tweights), data_ptr(out), stream);
    return ffi::Error::Success();
}

ffi::Error factorized_projected_backward_impl(
    ffi::AnyBuffer x, ffi::AnyBuffer sh, ffi::AnyBuffer weights, ffi::AnyBuffer senders,
    ffi::AnyBuffer receivers, ffi::AnyBuffer dout, ffi::Result<ffi::AnyBuffer> dx,
    ffi::Result<ffi::AnyBuffer> dsh, ffi::Result<ffi::AnyBuffer> dweights,
    stream_t stream, OeqExecutableState* state, int32_t device_ordinal,
    std::string_view source, int64_t hash,
    int64_t channels, int64_t input_dim, int64_t edge_dim, int64_t weight_dim,
    int64_t output_dim) {
    validate_projected_inputs(x, sh, senders, input_dim, edge_dim);
    const int64_t nodes = x.dimensions()[0], edges = sh.dimensions()[0];
    check_tensor(weights, {edges, weight_dim}, x.element_type(), "weights");
    check_tensor(receivers, {edges}, xla::ffi::DataType::S32, "receivers");
    check_tensor(dout, {nodes, output_dim}, x.element_type(), "dout");
    check_tensor(*dx, {nodes, input_dim}, x.element_type(), "dx");
    check_tensor(*dsh, {edges, edge_dim}, x.element_type(), "dsh");
    check_tensor(*dweights, {edges, weight_dim}, x.element_type(), "dweights");
    zero_buffer(*dx, stream);
    zero_buffer(*dsh, stream);
    zero_buffer(*dweights, stream);
    auto loaded = factorized_projected_kernel(state, device_ordinal);
    if (!loaded) return loaded.error();
    auto* jit_kernel = *loaded;
    jit_kernel->spatial_backward(
        nodes, edges, data_ptr(x), data_ptr(sh), data_ptr(weights), data_ptr(senders),
        data_ptr(receivers), data_ptr(dout), data_ptr(dx), data_ptr(dsh),
        data_ptr(dweights), stream);
    return ffi::Error::Success();
}

ffi::Error factorized_projected_weight_backward_impl(
    ffi::AnyBuffer x, ffi::AnyBuffer sh, ffi::AnyBuffer senders,
    ffi::AnyBuffer receivers, ffi::AnyBuffer dout, ffi::Result<ffi::AnyBuffer> dweights,
    stream_t stream, OeqExecutableState* state, int32_t device_ordinal,
    std::string_view source, int64_t hash,
    int64_t channels, int64_t input_dim, int64_t edge_dim, int64_t weight_dim,
    int64_t output_dim) {
    validate_projected_inputs(x, sh, senders, input_dim, edge_dim);
    const int64_t nodes = x.dimensions()[0], edges = sh.dimensions()[0];
    check_tensor(receivers, {edges}, xla::ffi::DataType::S32, "receivers");
    check_tensor(dout, {nodes, output_dim}, x.element_type(), "dout");
    check_tensor(*dweights, {edges, weight_dim}, x.element_type(), "dweights");
    zero_buffer(*dweights, stream);
    auto loaded = factorized_projected_kernel(state, device_ordinal);
    if (!loaded) return loaded.error();
    auto* jit_kernel = *loaded;
    jit_kernel->weight_backward(
        nodes, edges, channels, data_ptr(x), data_ptr(sh), data_ptr(senders),
        data_ptr(receivers), data_ptr(dout), data_ptr(dweights), stream);
    return ffi::Error::Success();
}

ffi::Error factorized_projected_backward_jvp_impl(
    ffi::AnyBuffer x, ffi::AnyBuffer sh, ffi::AnyBuffer weights, ffi::AnyBuffer senders,
    ffi::AnyBuffer receivers, ffi::AnyBuffer dout, ffi::AnyBuffer tx,
    ffi::AnyBuffer tsh, ffi::AnyBuffer tweights, ffi::AnyBuffer tdout,
    ffi::Result<ffi::AnyBuffer> tdx, ffi::Result<ffi::AnyBuffer> tdsh,
    ffi::Result<ffi::AnyBuffer> tdweights, stream_t stream,
    OeqExecutableState* state, int32_t device_ordinal,
    std::string_view source, int64_t hash, int64_t channels, int64_t input_dim,
    int64_t edge_dim, int64_t weight_dim, int64_t output_dim) {
    validate_projected_inputs(x, sh, senders, input_dim, edge_dim);
    const int64_t nodes = x.dimensions()[0], edges = sh.dimensions()[0];
    check_tensor(weights, {edges, weight_dim}, x.element_type(), "weights");
    check_tensor(receivers, {edges}, xla::ffi::DataType::S32, "receivers");
    check_tensor(dout, {nodes, output_dim}, x.element_type(), "dout");
    check_tensor_or_scalar_placeholder(
        tx, {nodes, input_dim}, x.element_type(), "tx");
    check_tensor_or_scalar_placeholder(
        tsh, {edges, edge_dim}, x.element_type(), "tsh");
    check_tensor_or_scalar_placeholder(
        tweights, {edges, weight_dim}, x.element_type(), "tweights");
    check_tensor_or_scalar_placeholder(
        tdout, {nodes, output_dim}, x.element_type(), "tdout");
    check_tensor(*tdx, {nodes, input_dim}, x.element_type(), "tdx");
    check_tensor(*tdsh, {edges, edge_dim}, x.element_type(), "tdsh");
    check_tensor(*tdweights, {edges, weight_dim}, x.element_type(), "tdweights");
    zero_buffer(*tdx, stream);
    zero_buffer(*tdsh, stream);
    zero_buffer(*tdweights, stream);
    auto loaded = factorized_projected_kernel(state, device_ordinal);
    if (!loaded) return loaded.error();
    auto* jit_kernel = *loaded;
    jit_kernel->spatial_backward_jvp(
        nodes, edges, data_ptr(x), data_ptr(sh), data_ptr(weights), data_ptr(senders),
        data_ptr(receivers), data_ptr(dout), data_ptr(tx), data_ptr(tsh),
        data_ptr(tweights), data_ptr(tdout), data_ptr(tdx), data_ptr(tdsh),
        data_ptr(tdweights), stream);
    return ffi::Error::Success();
}

// ------------------- Generated symmetric contractions -------------------

void validate_symmetric_inputs(const ffi::AnyBuffer &x, const ffi::AnyBuffer &selector,
                               const ffi::AnyBuffer &weights,
                               int64_t channels, int64_t feature_dim,
                               int64_t feature_layout, int64_t weight_dim) {
    if (x.dimensions().size() != 3) {
        throw std::logic_error("symmetric x must have rank 3");
    }
    const int64_t nodes = x.dimensions()[0];
    const auto dtype = x.element_type();
    if (dtype != xla::ffi::DataType::F32 && dtype != xla::ffi::DataType::F64) {
        throw std::logic_error("symmetric contraction supports only f32 and f64");
    }
    if (feature_layout == 0) {
        check_tensor(x, {nodes, channels, feature_dim}, dtype, "symmetric x");
    } else if (feature_layout == 1) {
        check_tensor(x, {nodes, feature_dim, channels}, dtype, "symmetric x");
    } else {
        throw std::logic_error("invalid symmetric feature layout");
    }
    if (weights.dimensions().size() != 2 || weights.dimensions()[1] != weight_dim ||
        weights.element_type() != dtype) {
        throw std::logic_error(
            "symmetric weights must have shape [S,weight_dim] and match x dtype");
    }
    const int64_t num_elements = weights.dimensions()[0];
    if (num_elements <= 0) {
        throw std::logic_error("symmetric weights require a nonempty species table");
    }
    check_tensor(selector, {nodes}, xla::ffi::DataType::S32, "symmetric species");
}

void validate_symmetric_feature_buffer(const ffi::AnyBuffer &value,
                                       const ffi::AnyBuffer &x, std::string_view name) {
    const auto value_dimensions = value.dimensions();
    const auto x_dimensions = x.dimensions();
    bool dimensions_match = value_dimensions.size() == x_dimensions.size();
    for (size_t index = 0; dimensions_match && index < x_dimensions.size(); ++index) {
        dimensions_match = value_dimensions[index] == x_dimensions[index];
    }
    if (!dimensions_match || value.element_type() != x.element_type()) {
        throw std::logic_error(std::string(name) +
                               " must match symmetric x shape and dtype");
    }
}

ffi::Error symmetric_literal_forward_species_impl(
    ffi::AnyBuffer x, ffi::AnyBuffer species, ffi::AnyBuffer weights,
    ffi::Result<ffi::AnyBuffer> out, stream_t stream,
    OeqExecutableState* state, int32_t device_ordinal,
    std::string_view source, int64_t hash, int64_t channels, int64_t feature_dim,
    int64_t feature_layout, int64_t output_dim, int64_t weight_dim) {
    validate_symmetric_inputs(x, species, weights, channels, feature_dim,
                              feature_layout, weight_dim);
    const int64_t num_elements = weights.dimensions()[0];
    const int64_t nodes = x.dimensions()[0];
    check_tensor(*out, {nodes, output_dim}, x.element_type(),
                 "symmetric literal output");
    zero_buffer(*out, stream);
    auto loaded = symmetric_literal_kernel(state, device_ordinal);
    if (!loaded) return loaded.error();
    auto* jit_kernel = *loaded;
    jit_kernel->forward_species(
        nodes, num_elements, channels, data_ptr(x), data_ptr(species), data_ptr(weights),
        data_ptr(out), stream);
    return ffi::Error::Success();
}

ffi::Error symmetric_literal_forward_jvp_x_species_impl(
    ffi::AnyBuffer x, ffi::AnyBuffer species, ffi::AnyBuffer weights, ffi::AnyBuffer tx,
    ffi::Result<ffi::AnyBuffer> tout, stream_t stream,
    OeqExecutableState* state, int32_t device_ordinal,
    std::string_view source, int64_t hash, int64_t channels, int64_t feature_dim,
    int64_t feature_layout, int64_t output_dim, int64_t weight_dim) {
    validate_symmetric_inputs(x, species, weights, channels, feature_dim,
                              feature_layout, weight_dim);
    const int64_t num_elements = weights.dimensions()[0];
    const int64_t nodes = x.dimensions()[0];
    validate_symmetric_feature_buffer(tx, x, "symmetric literal tx");
    check_tensor(*tout, {nodes, output_dim}, x.element_type(),
                 "symmetric literal JVP output");
    zero_buffer(*tout, stream);
    auto loaded = symmetric_literal_kernel(state, device_ordinal);
    if (!loaded) return loaded.error();
    auto* jit_kernel = *loaded;
    jit_kernel->forward_jvp_x_species(
        nodes, num_elements, channels, data_ptr(x), data_ptr(species), data_ptr(weights),
        data_ptr(tx), data_ptr(tout), stream);
    return ffi::Error::Success();
}

ffi::Error symmetric_literal_backward_x_species_impl(
    ffi::AnyBuffer x, ffi::AnyBuffer species, ffi::AnyBuffer weights,
    ffi::AnyBuffer dout, ffi::Result<ffi::AnyBuffer> dx, stream_t stream,
    OeqExecutableState* state, int32_t device_ordinal,
    std::string_view source, int64_t hash, int64_t channels,
    int64_t feature_dim, int64_t feature_layout, int64_t output_dim,
    int64_t weight_dim) {
    validate_symmetric_inputs(x, species, weights, channels, feature_dim,
                              feature_layout, weight_dim);
    const int64_t num_elements = weights.dimensions()[0];
    const int64_t nodes = x.dimensions()[0];
    check_tensor(dout, {nodes, output_dim}, x.element_type(),
                 "symmetric literal dout");
    validate_symmetric_feature_buffer(*dx, x, "symmetric literal dx");
    zero_buffer(*dx, stream);
    auto loaded = symmetric_literal_kernel(state, device_ordinal);
    if (!loaded) return loaded.error();
    auto* jit_kernel = *loaded;
    jit_kernel->backward_x_species(
        nodes, num_elements, channels, data_ptr(x), data_ptr(species), data_ptr(weights),
        data_ptr(dout), data_ptr(dx), stream);
    return ffi::Error::Success();
}

ffi::Error symmetric_literal_backward_jvp_x_species_impl(
    ffi::AnyBuffer x, ffi::AnyBuffer species, ffi::AnyBuffer weights,
    ffi::AnyBuffer dout, ffi::AnyBuffer tx, ffi::AnyBuffer tdout,
    ffi::Result<ffi::AnyBuffer> tdx, stream_t stream,
    OeqExecutableState* state, int32_t device_ordinal,
    std::string_view source, int64_t hash, int64_t channels, int64_t feature_dim,
    int64_t feature_layout, int64_t output_dim, int64_t weight_dim) {
    validate_symmetric_inputs(x, species, weights, channels, feature_dim,
                              feature_layout, weight_dim);
    const int64_t num_elements = weights.dimensions()[0];
    const int64_t nodes = x.dimensions()[0];
    check_tensor(dout, {nodes, output_dim}, x.element_type(),
                 "symmetric literal dout");
    validate_symmetric_feature_buffer(tx, x, "symmetric literal tx");
    check_tensor(tdout, {nodes, output_dim}, x.element_type(),
                 "symmetric literal tdout");
    validate_symmetric_feature_buffer(*tdx, x, "symmetric literal tdx");
    zero_buffer(*tdx, stream);
    auto loaded = symmetric_literal_kernel(state, device_ordinal);
    if (!loaded) return loaded.error();
    auto* jit_kernel = *loaded;
    jit_kernel->backward_jvp_x_species(
        nodes, num_elements, channels, data_ptr(x), data_ptr(species), data_ptr(weights),
        data_ptr(dout), data_ptr(tx), data_ptr(tdout), data_ptr(tdx), stream);
    return ffi::Error::Success();
}

ffi::Error symmetric_literal_backward_hvp_x_species_impl(
    ffi::AnyBuffer x, ffi::AnyBuffer species, ffi::AnyBuffer weights,
    ffi::AnyBuffer dout, ffi::AnyBuffer tx, ffi::Result<ffi::AnyBuffer> tdx,
    stream_t stream, OeqExecutableState* state, int32_t device_ordinal,
    std::string_view source, int64_t hash,
    int64_t channels, int64_t feature_dim, int64_t feature_layout,
    int64_t output_dim, int64_t weight_dim) {
    validate_symmetric_inputs(x, species, weights, channels, feature_dim,
                              feature_layout, weight_dim);
    const int64_t num_elements = weights.dimensions()[0];
    const int64_t nodes = x.dimensions()[0];
    check_tensor(dout, {nodes, output_dim}, x.element_type(),
                 "symmetric literal dout");
    validate_symmetric_feature_buffer(tx, x, "symmetric literal tx");
    validate_symmetric_feature_buffer(*tdx, x, "symmetric literal tdx");
    zero_buffer(*tdx, stream);
    auto loaded = symmetric_literal_kernel(state, device_ordinal);
    if (!loaded) return loaded.error();
    auto* jit_kernel = *loaded;
    jit_kernel->backward_hvp_x_species(
        nodes, num_elements, channels, data_ptr(x), data_ptr(species), data_ptr(weights),
        data_ptr(dout), data_ptr(tx), data_ptr(tdx), stream);
    return ffi::Error::Success();
}

ffi::Error symmetric_literal_backward_jvp_xw_species_impl(
    ffi::AnyBuffer x, ffi::AnyBuffer species, ffi::AnyBuffer weights,
    ffi::AnyBuffer dout, ffi::AnyBuffer tx, ffi::AnyBuffer tweights,
    ffi::AnyBuffer tdout, ffi::Result<ffi::AnyBuffer> tdx, stream_t stream,
    OeqExecutableState* state, int32_t device_ordinal,
    std::string_view source, int64_t hash, int64_t channels,
    int64_t feature_dim, int64_t feature_layout, int64_t output_dim,
    int64_t weight_dim) {
    validate_symmetric_inputs(x, species, weights, channels, feature_dim,
                              feature_layout, weight_dim);
    const int64_t num_elements = weights.dimensions()[0];
    const int64_t nodes = x.dimensions()[0];
    check_tensor(dout, {nodes, output_dim}, x.element_type(),
                 "symmetric literal dout");
    validate_symmetric_feature_buffer(tx, x, "symmetric literal tx");
    check_tensor(tweights, {num_elements, weight_dim}, x.element_type(),
                 "symmetric literal tweights");
    check_tensor(tdout, {nodes, output_dim}, x.element_type(),
                 "symmetric literal tdout");
    validate_symmetric_feature_buffer(*tdx, x, "symmetric literal tdx");
    zero_buffer(*tdx, stream);
    auto loaded = symmetric_literal_kernel(state, device_ordinal);
    if (!loaded) return loaded.error();
    auto* jit_kernel = *loaded;
    jit_kernel->backward_jvp_xw_species(
        nodes, num_elements, channels, data_ptr(x), data_ptr(species), data_ptr(weights),
        data_ptr(dout), data_ptr(tx), data_ptr(tweights), data_ptr(tdout), data_ptr(tdx),
        stream);
    return ffi::Error::Success();
}

ffi::Error symmetric_literal_backward_jvp_xw_transpose_species_impl(
    ffi::AnyBuffer x, ffi::AnyBuffer species, ffi::AnyBuffer weights,
    ffi::AnyBuffer dout, ffi::AnyBuffer ctdx, ffi::Result<ffi::AnyBuffer> ctx,
    ffi::Result<ffi::AnyBuffer> ctweights, ffi::Result<ffi::AnyBuffer> ctdout,
    stream_t stream, OeqExecutableState* state, int32_t device_ordinal,
    std::string_view source, int64_t hash,
    int64_t channels, int64_t feature_dim, int64_t feature_layout,
    int64_t output_dim, int64_t weight_dim) {
    validate_symmetric_inputs(x, species, weights, channels, feature_dim,
                              feature_layout, weight_dim);
    const int64_t num_elements = weights.dimensions()[0];
    const int64_t nodes = x.dimensions()[0];
    check_tensor(dout, {nodes, output_dim}, x.element_type(),
                 "symmetric literal dout");
    validate_symmetric_feature_buffer(ctdx, x, "symmetric literal ctdx");
    validate_symmetric_feature_buffer(*ctx, x, "symmetric literal ctx");
    check_tensor(*ctweights, {num_elements, weight_dim}, x.element_type(),
                 "symmetric literal ctweights");
    check_tensor(*ctdout, {nodes, output_dim}, x.element_type(),
                 "symmetric literal ctdout");
    zero_buffer(*ctx, stream);
    zero_buffer(*ctweights, stream);
    zero_buffer(*ctdout, stream);
    auto loaded = symmetric_literal_kernel(state, device_ordinal);
    if (!loaded) return loaded.error();
    auto* jit_kernel = *loaded;
    jit_kernel->backward_jvp_xw_transpose_species(
        nodes, num_elements, channels, data_ptr(x), data_ptr(species), data_ptr(weights),
        data_ptr(dout), data_ptr(ctdx), data_ptr(ctx), data_ptr(ctweights),
        data_ptr(ctdout), stream);
    return ffi::Error::Success();
}

ffi::ErrorOr<std::unique_ptr<OeqExecutableState>>
factorized_projected_instantiate_impl(
    std::string_view source, int64_t hash, int64_t, int64_t, int64_t,
    int64_t, int64_t) {
    return instantiate_kernel(KernelFamily::kFactorizedProjected, source, hash);
}

ffi::ErrorOr<std::unique_ptr<OeqExecutableState>>
symmetric_literal_instantiate_impl(
    std::string_view source, int64_t hash, int64_t, int64_t, int64_t,
    int64_t, int64_t) {
    return instantiate_kernel(KernelFamily::kSymmetricLiteral, source, hash);
}

ffi::Error generated_initialize_impl(
    OeqExecutableState* state, int32_t device_ordinal,
    std::string_view source, int64_t hash, int64_t, int64_t, int64_t,
    int64_t, int64_t) {
    return initialize_impl(state, device_ordinal, source, hash);
}

#define OEQ_GENERATED_CONTEXTS                                                         \
    .Ctx<ffi::PlatformStream<stream_t>>()                                             \
        .Ctx<ffi::State<OeqExecutableState>>()                                       \
        .Ctx<ffi::DeviceOrdinal>()

#define OEQ_GENERATED_ATTRIBUTES                                                      \
    .Attr<std::string_view>("source")                                               \
        .Attr<int64_t>("hash")

#define OEQ_PROJECTED_ATTRIBUTES                                                       \
    OEQ_GENERATED_ATTRIBUTES                                                         \
        .Attr<int64_t>("channels")                                                     \
        .Attr<int64_t>("input_dim")                                                    \
        .Attr<int64_t>("edge_dim")                                                     \
        .Attr<int64_t>("weight_dim")                                                   \
        .Attr<int64_t>("output_dim")

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    factorized_projected_instantiate, factorized_projected_instantiate_impl,
    ffi::Ffi::BindInstantiate() OEQ_PROJECTED_ATTRIBUTES);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    factorized_projected_initialize, generated_initialize_impl,
    ffi::Ffi::BindInitialize()
        .Ctx<ffi::State<OeqExecutableState>>()
        .Ctx<ffi::DeviceOrdinal>()
        OEQ_PROJECTED_ATTRIBUTES);

XLA_FFI_DEFINE_HANDLER_SYMBOL(factorized_projected_forward,
                              factorized_projected_forward_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>() OEQ_GENERATED_CONTEXTS
                                      OEQ_PROJECTED_ATTRIBUTES,
                              {xla::ffi::Traits::kCmdBufferCompatible});
XLA_FFI_DEFINE_HANDLER_SYMBOL(factorized_projected_forward_jvp,
                              factorized_projected_forward_jvp_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>() OEQ_GENERATED_CONTEXTS
                                      OEQ_PROJECTED_ATTRIBUTES,
                              {xla::ffi::Traits::kCmdBufferCompatible});
XLA_FFI_DEFINE_HANDLER_SYMBOL(factorized_projected_spatial_backward,
                              factorized_projected_backward_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>() OEQ_GENERATED_CONTEXTS
                                      OEQ_PROJECTED_ATTRIBUTES,
                              {xla::ffi::Traits::kCmdBufferCompatible});
XLA_FFI_DEFINE_HANDLER_SYMBOL(factorized_projected_weight_backward,
                              factorized_projected_weight_backward_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>() OEQ_GENERATED_CONTEXTS
                                      OEQ_PROJECTED_ATTRIBUTES,
                              {xla::ffi::Traits::kCmdBufferCompatible});
XLA_FFI_DEFINE_HANDLER_SYMBOL(factorized_projected_spatial_backward_jvp,
                              factorized_projected_backward_jvp_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>() OEQ_GENERATED_CONTEXTS
                                      OEQ_PROJECTED_ATTRIBUTES,
                              {xla::ffi::Traits::kCmdBufferCompatible});

#undef OEQ_PROJECTED_ATTRIBUTES

#define OEQ_SYMMETRIC_ATTRIBUTES                                                       \
    OEQ_GENERATED_ATTRIBUTES                                                         \
        .Attr<int64_t>("channels")                                                     \
        .Attr<int64_t>("feature_dim")                                                  \
        .Attr<int64_t>("feature_layout")                                               \
        .Attr<int64_t>("output_dim")                                                   \
        .Attr<int64_t>("weight_dim")

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    symmetric_literal_instantiate, symmetric_literal_instantiate_impl,
    ffi::Ffi::BindInstantiate() OEQ_SYMMETRIC_ATTRIBUTES);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    symmetric_literal_initialize, generated_initialize_impl,
    ffi::Ffi::BindInitialize()
        .Ctx<ffi::State<OeqExecutableState>>()
        .Ctx<ffi::DeviceOrdinal>()
        OEQ_SYMMETRIC_ATTRIBUTES);

XLA_FFI_DEFINE_HANDLER_SYMBOL(symmetric_literal_forward_species,
                              symmetric_literal_forward_species_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  OEQ_GENERATED_CONTEXTS OEQ_SYMMETRIC_ATTRIBUTES,
                              {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(symmetric_literal_forward_jvp_x_species,
                              symmetric_literal_forward_jvp_x_species_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  OEQ_GENERATED_CONTEXTS OEQ_SYMMETRIC_ATTRIBUTES,
                              {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(symmetric_literal_backward_x_species,
                              symmetric_literal_backward_x_species_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  OEQ_GENERATED_CONTEXTS OEQ_SYMMETRIC_ATTRIBUTES,
                              {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(symmetric_literal_backward_jvp_x_species,
                              symmetric_literal_backward_jvp_x_species_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  OEQ_GENERATED_CONTEXTS OEQ_SYMMETRIC_ATTRIBUTES,
                              {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(symmetric_literal_backward_hvp_x_species,
                              symmetric_literal_backward_hvp_x_species_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  OEQ_GENERATED_CONTEXTS OEQ_SYMMETRIC_ATTRIBUTES,
                              {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(symmetric_literal_backward_jvp_xw_species,
                              symmetric_literal_backward_jvp_xw_species_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  OEQ_GENERATED_CONTEXTS OEQ_SYMMETRIC_ATTRIBUTES,
                              {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(symmetric_literal_backward_jvp_xw_transpose_species,
                              symmetric_literal_backward_jvp_xw_transpose_species_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  OEQ_GENERATED_CONTEXTS OEQ_SYMMETRIC_ATTRIBUTES,
                              {xla::ffi::Traits::kCmdBufferCompatible});

#undef OEQ_SYMMETRIC_ATTRIBUTES
#undef OEQ_GENERATED_ATTRIBUTES
#undef OEQ_GENERATED_CONTEXTS

#define OEQ_FFI_HANDLER(NAME, INSTANTIATE, INITIALIZE)                                \
    {#NAME, reinterpret_cast<void*>(INSTANTIATE),                                    \
     reinterpret_cast<void*>(INITIALIZE), reinterpret_cast<void*>(NAME)}

const OeqFfiHandler kFfiHandlers[] = {
    OEQ_FFI_HANDLER(tp_forward, tp_instantiate, tp_initialize),
    OEQ_FFI_HANDLER(tp_backward, tp_instantiate, tp_initialize),
    OEQ_FFI_HANDLER(tp_double_backward, tp_instantiate, tp_initialize),
    OEQ_FFI_HANDLER(conv_forward, conv_instantiate, conv_initialize),
    OEQ_FFI_HANDLER(conv_backward, conv_instantiate, conv_initialize),
    OEQ_FFI_HANDLER(
        conv_double_backward, conv_instantiate, conv_initialize),
    OEQ_FFI_HANDLER(
        factorized_projected_forward,
        factorized_projected_instantiate,
        factorized_projected_initialize),
    OEQ_FFI_HANDLER(
        factorized_projected_forward_jvp,
        factorized_projected_instantiate,
        factorized_projected_initialize),
    OEQ_FFI_HANDLER(
        factorized_projected_spatial_backward,
        factorized_projected_instantiate,
        factorized_projected_initialize),
    OEQ_FFI_HANDLER(
        factorized_projected_weight_backward,
        factorized_projected_instantiate,
        factorized_projected_initialize),
    OEQ_FFI_HANDLER(
        factorized_projected_spatial_backward_jvp,
        factorized_projected_instantiate,
        factorized_projected_initialize),
    OEQ_FFI_HANDLER(
        symmetric_literal_forward_species,
        symmetric_literal_instantiate,
        symmetric_literal_initialize),
    OEQ_FFI_HANDLER(
        symmetric_literal_forward_jvp_x_species,
        symmetric_literal_instantiate,
        symmetric_literal_initialize),
    OEQ_FFI_HANDLER(
        symmetric_literal_backward_x_species,
        symmetric_literal_instantiate,
        symmetric_literal_initialize),
    OEQ_FFI_HANDLER(
        symmetric_literal_backward_jvp_x_species,
        symmetric_literal_instantiate,
        symmetric_literal_initialize),
    OEQ_FFI_HANDLER(
        symmetric_literal_backward_hvp_x_species,
        symmetric_literal_instantiate,
        symmetric_literal_initialize),
    OEQ_FFI_HANDLER(
        symmetric_literal_backward_jvp_xw_species,
        symmetric_literal_instantiate,
        symmetric_literal_initialize),
    OEQ_FFI_HANDLER(
        symmetric_literal_backward_jvp_xw_transpose_species,
        symmetric_literal_instantiate,
        symmetric_literal_initialize),
};

#undef OEQ_FFI_HANDLER

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

extern "C" const OeqFfiHandlerTable* oeq_ffi_handler_table() {
    return &kFfiHandlerTable;
}

#ifndef OEQ_NO_PYTHON_MODULE
NB_MODULE(openequivariance_extjax, m) {
    m.def("type_registrations", []() {
        nb::dict registrations;
        const auto* table = oeq_ffi_handler_table();
        for (uint32_t index = 0; index < table->type_count; ++index) {
            const auto& type = table->types[index];
            nb::dict registration;
            registration["type_id"] = nb::capsule(type.type_id);
            registration["type_info"] = nb::capsule(
                const_cast<void*>(type.type_info));
            registrations[type.name] = registration;
        }
        return registrations;
    });
    m.def("registrations", []() {
        nb::dict registrations;
        const auto* handlers = oeq_ffi_handler_table();
        for (uint32_t index = 0; index < handlers->handler_count; ++index) {
            const auto& handler = handlers->handlers[index];
            nb::dict stages;
            stages["instantiate"] = nb::capsule(handler.instantiate);
            stages["initialize"] = nb::capsule(handler.initialize);
            stages["execute"] = nb::capsule(handler.execute);
            registrations[handler.name] = stages;
        }
        return registrations;
    });
    m.def("is_hip", &is_hip);

    nb::class_<DeviceProp>(m, "DeviceProp")
        .def(nb::init<int>())
        .def_ro("name", &DeviceProp::name)
        .def_ro("warpsize", &DeviceProp::warpsize)
        .def_ro("major", &DeviceProp::major)
        .def_ro("minor", &DeviceProp::minor)
        .def_ro("multiprocessorCount", &DeviceProp::multiprocessorCount)
        .def_ro("maxSharedMemPerBlock", &DeviceProp::maxSharedMemPerBlock);

    nb::class_<GPUTimer>(m, "GPUTimer")
        .def(nb::init<>())
        .def("start", &GPUTimer::start)
        .def("stop_clock_get_elapsed", &GPUTimer::stop_clock_get_elapsed)
        .def("clear_L2_cache", &GPUTimer::clear_L2_cache);
}
#endif
