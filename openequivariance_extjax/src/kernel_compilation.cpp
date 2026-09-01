#include "kernel_compilation.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "json11/json11.hpp"

#ifdef CUDA_BACKEND
#include <cuda.h>
#include <cuda_runtime.h>

#include "backend/backend_cuda.hpp"
#endif

#ifdef HIP_BACKEND
#include "backend/backend_hip.hpp"
#endif

#include "convolution.hpp"
#include "tensorproducts.hpp"

namespace {

using json = json11::Json;

std::atomic<uint64_t> interner_hits{0};
std::atomic<uint64_t> interner_misses{0};
std::atomic<uint64_t> compilations_started{0};

ffi::DataType decode_serialized_dtype(int64_t value) {
    switch (value) {
        case 1:
            return ffi::DataType::F32;
        case 2:
            return ffi::DataType::F64;
        case 3:
            return ffi::DataType::S32;
        case 4:
            return ffi::DataType::S64;
        case 5:
            return ffi::DataType::U8;
        default:
            throw std::logic_error("Unsupported tensor datatype!");
    }
}

std::unordered_map<std::string, int64_t> parse_json_config(const json& object) {
    std::unordered_map<std::string, int64_t> values;
    for (const auto& item : object.object_items()) {
        values[item.first] = static_cast<int64_t>(item.second.number_value());
    }
    return values;
}

KernelLaunchConfig parse_launch_config(const json& config) {
    const auto values = parse_json_config(config);
    return KernelLaunchConfig(
        values.at("num_blocks"), values.at("num_threads"), values.at("smem"));
}

FfiKernelProperties parse_kernel_properties(
    const std::unordered_map<std::string, int64_t>& dimensions,
    bool is_convolution) {
    FfiKernelProperties properties;
    properties.L1_dim = dimensions.at("L1_dim");
    properties.L2_dim = dimensions.at("L2_dim");
    properties.L3_dim = dimensions.at("L3_dim");
    properties.weight_numel = dimensions.at("weight_numel");
    properties.shared_weights = dimensions.at("shared_weights");
    properties.irrep_dtype = decode_serialized_dtype(dimensions.at("irrep_dtype"));
    properties.weight_dtype = decode_serialized_dtype(dimensions.at("weight_dtype"));
    if (is_convolution) {
        properties.workspace_size = dimensions.at("workspace_size");
        properties.deterministic = dimensions.at("deterministic");
        properties.idx_dtype = decode_serialized_dtype(dimensions.at("idx_dtype"));
    }
    return properties;
}

struct KernelPlan {
    std::string source;
    KernelLaunchConfig forward_config;
    KernelLaunchConfig backward_config;
    KernelLaunchConfig double_backward_config;
    FfiKernelProperties properties;
    int opt_level;
};

KernelPlan parse_kernel_plan(std::string_view payload, bool is_convolution) {
    std::string error;
    const json root = json::parse(std::string(payload), error);
    if (!error.empty()) {
        throw std::runtime_error("JSON Parse Error: " + error);
    }
    const auto dimensions = parse_json_config(root["kernel_prop"]);
    const std::string source = root["kernel"].string_value();
    if (source.empty()) {
        throw std::runtime_error("Kernel JSON must contain nonempty source");
    }
    return {
        source,
        parse_launch_config(root["forward_config"]),
        parse_launch_config(root["backward_config"]),
        parse_launch_config(root["double_backward_config"]),
        parse_kernel_properties(dimensions, is_convolution),
        static_cast<int>(dimensions.at("opt_level")),
    };
}

#ifdef CUDA_BACKEND

int parse_bounded_environment_integer(
    const char* name, int default_value, int minimum, int maximum) {
    const char* setting = std::getenv(name);
    if (setting == nullptr) {
        return default_value;
    }
    const std::string_view value(setting);
    int parsed;
    const auto [end, error] = std::from_chars(
        value.data(), value.data() + value.size(), parsed);
    if (error != std::errc() || end != value.data() + value.size() ||
        parsed < minimum || parsed > maximum) {
        throw std::invalid_argument(
            std::string(name) + " must be an integer from " +
            std::to_string(minimum) + " to " + std::to_string(maximum));
    }
    return parsed;
}

// This bounded pool runs independent NVRTC jobs outside XLA execution threads.
class CompilerPool {
public:
    static CompilerPool& instance() {
        static CompilerPool pool;
        return pool;
    }

    // Submit waits for queue space and returns false only during shutdown.
    bool submit(std::function<void()> task) {
        std::unique_lock<std::mutex> lock(mutex_);
        space_available_.wait(lock, [this] {
            return stopping_ || tasks_.size() < queue_capacity_;
        });
        if (stopping_) {
            return false;
        }
        tasks_.push_back(std::move(task));
        lock.unlock();
        ready_.notify_one();
        return true;
    }

private:
    CompilerPool()
        : queue_capacity_(parse_bounded_environment_integer(
              "OEQ_JAX_COMPILER_QUEUE_CAPACITY", 32, 1, 256)) {
        const int worker_count = parse_bounded_environment_integer(
            "OEQ_JAX_COMPILER_THREADS", 8, 1, 64);
        try {
            for (int index = 0; index < worker_count; ++index) {
                workers_.emplace_back([this] { run(); });
            }
        } catch (...) {
            stop();
            throw;
        }
    }

    ~CompilerPool() { stop(); }

    // Shutdown wakes the workers, lets them finish queued jobs, and joins
    // their threads.
    void stop() {
        std::unique_lock<std::mutex> lock(mutex_);
        stopping_ = true;
        lock.unlock();
        ready_.notify_all();
        space_available_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    // Each worker removes one queued job and runs it without holding the queue
    // lock.
    void run() {
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
            if (stopping_ && tasks_.empty()) {
                return;
            }
            std::function<void()> task = std::move(tasks_.front());
            tasks_.pop_front();
            lock.unlock();
            space_available_.notify_one();
            task();
        }
    }

    std::mutex mutex_;
    std::condition_variable ready_;
    std::condition_variable space_available_;
    std::deque<std::function<void()>> tasks_;
    std::vector<std::thread> workers_;
    const size_t queue_capacity_;
    bool stopping_ = false;
};

enum class CompilationBackend : uint8_t { kCuda = 1 };

// This key identifies one compiled image by backend and architecture.
struct CompilationTarget {
    CompilationBackend backend;
    int architecture;

    bool operator==(const CompilationTarget& other) const {
        return backend == other.backend && architecture == other.architecture;
    }
};

struct CompilationTargetHash {
    size_t operator()(const CompilationTarget& target) const {
        size_t hash = static_cast<size_t>(target.backend);
        return hash * 31 + static_cast<size_t>(target.architecture);
    }
};

CompilationTarget target_for_device(int32_t device_ordinal) {
    if (cuInit(0) != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to initialize the CUDA driver");
    }
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
    return {CompilationBackend::kCuda, major * 10 + minor};
}

template <typename Implementation>
class TypedSharedKernel;

// A worker compiles an image for one target. The first execution loads its
// launcher; later executions reuse it.
template <typename Implementation>
struct TargetArtifact {
    explicit TargetArtifact(
        CompilationTarget target,
        std::shared_future<CompiledKernelImage> image)
        : target(std::move(target)), image(std::move(image)) {}

    std::shared_ptr<Implementation> load(const KernelPlan& plan) {
        std::shared_future<std::shared_ptr<Implementation>> future;
        std::shared_ptr<std::promise<std::shared_ptr<Implementation>>> promise;
        {
            std::lock_guard<std::mutex> lock(load_mutex);
            if (!loaded.valid()) {
                promise = std::make_shared<std::promise<std::shared_ptr<Implementation>>>();
                loaded = promise->get_future().share();
            }
            future = loaded;
        }
        // The first caller waits for the image and loads the launcher. Other
        // callers wait for that result.
        if (promise != nullptr) {
            try {
                const CompiledKernelImage& compiled = image.get();
                promise->set_value(std::make_shared<Implementation>(
                    compiled, plan.forward_config, plan.backward_config,
                    plan.double_backward_config, plan.opt_level));
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        }
        return future.get();
    }

    const CompilationTarget target;
    const std::shared_future<CompiledKernelImage> image;
    std::mutex load_mutex;
    std::shared_future<std::shared_ptr<Implementation>> loaded;
};

#endif

template <typename Implementation>
class TypedSharedKernel final
    : public SharedKernel,
      public std::enable_shared_from_this<TypedSharedKernel<Implementation>> {
public:
    explicit TypedSharedKernel(KernelPlan plan) : plan_(std::move(plan)) {}

    void initialize(int32_t device_ordinal) override {
#ifdef CUDA_BACKEND
        get_or_schedule(target_for_device(device_ordinal));
#else
        static_cast<void>(load_host());
#endif
    }

    Implementation* load(int32_t device_ordinal) {
#ifdef CUDA_BACKEND
        return get_or_schedule(target_for_device(device_ordinal))
            ->load(plan_)
            .get();
#else
        (void)device_ordinal;
        return load_host().get();
#endif
    }

    const FfiKernelProperties& properties() const { return plan_.properties; }

private:
#ifdef CUDA_BACKEND
    class CompilationJob {
    public:
        CompilationJob(
            std::shared_ptr<TypedSharedKernel> owner,
            CompilationTarget target,
            std::shared_ptr<std::promise<CompiledKernelImage>> completion)
            : owner_(std::move(owner)),
              target_(std::move(target)),
              completion_(std::move(completion)) {}

        // A job always completes its promise with an image or an exception.
        void run() const {
            try {
                compilations_started.fetch_add(1, std::memory_order_relaxed);
                completion_->set_value(Implementation::compile_image(
                    owner_->plan_.source, target_.architecture / 10,
                    target_.architecture % 10, owner_->plan_.opt_level));
            } catch (...) {
                completion_->set_exception(std::current_exception());
            }
        }

        void fail(std::exception_ptr error) const {
            completion_->set_exception(std::move(error));
        }

    private:
        std::shared_ptr<TypedSharedKernel> owner_;
        CompilationTarget target_;
        std::shared_ptr<std::promise<CompiledKernelImage>> completion_;
    };

    std::shared_ptr<TargetArtifact<Implementation>> get_or_schedule(
        const CompilationTarget& target) {
        std::shared_ptr<TargetArtifact<Implementation>> artifact;
        std::shared_ptr<std::promise<CompiledKernelImage>> promise;
        {
            // Publish the artifact before scheduling so all callers share one future.
            std::lock_guard<std::mutex> lock(artifacts_mutex_);
            const auto found = artifacts_.find(target);
            if (found != artifacts_.end()) {
                return found->second;
            }
            promise = std::make_shared<std::promise<CompiledKernelImage>>();
            artifact = std::make_shared<TargetArtifact<Implementation>>(
                target, promise->get_future().share());
            artifacts_.emplace(target, artifact);
        }

        CompilationJob job(this->shared_from_this(), target, promise);
        try {
            const bool submitted = CompilerPool::instance().submit(
                [job] { job.run(); });
            if (!submitted) {
                job.fail(std::make_exception_ptr(std::runtime_error(
                    "OpenEquivariance compiler pool is shutting down")));
            }
        } catch (...) {
            job.fail(std::current_exception());
        }
        return artifact;
    }

    std::mutex artifacts_mutex_;
    std::unordered_map<CompilationTarget,
                       std::shared_ptr<TargetArtifact<Implementation>>,
                       CompilationTargetHash>
        artifacts_;
#else
    std::shared_ptr<Implementation> load_host() {
        std::shared_future<std::shared_ptr<Implementation>> future;
        std::shared_ptr<std::promise<std::shared_ptr<Implementation>>> promise;
        {
            std::lock_guard<std::mutex> lock(host_load_mutex_);
            if (!host_loaded_.valid()) {
                promise = std::make_shared<std::promise<std::shared_ptr<Implementation>>>();
                host_loaded_ = promise->get_future().share();
            }
            future = host_loaded_;
        }
        if (promise != nullptr) {
            try {
                promise->set_value(std::make_shared<Implementation>(
                    plan_.source, plan_.forward_config, plan_.backward_config,
                    plan_.double_backward_config, plan_.opt_level));
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        }
        return future.get();
    }

    std::mutex host_load_mutex_;
    std::shared_future<std::shared_ptr<Implementation>> host_loaded_;
#endif
    KernelPlan plan_;
};

using TensorProductSharedKernel = TypedSharedKernel<JITTPImpl<JITKernel>>;
using ConvolutionSharedKernel = TypedSharedKernel<JITConvImpl<JITKernel>>;

// This factory constructs one typed shared kernel from a serialized plan.
struct SharedKernelFactory {
    const char* name;
    std::shared_ptr<SharedKernel> (*create)(std::string_view payload);
};

std::shared_ptr<SharedKernel> create_tensor_product_shared_kernel(
    std::string_view payload) {
    return std::make_shared<TensorProductSharedKernel>(
        parse_kernel_plan(payload, false));
}

std::shared_ptr<SharedKernel> create_convolution_shared_kernel(
    std::string_view payload) {
    return std::make_shared<ConvolutionSharedKernel>(
        parse_kernel_plan(payload, true));
}

const SharedKernelFactory kTensorProductFactory{
    "tensor_product", create_tensor_product_shared_kernel};
const SharedKernelFactory kConvolutionFactory{
    "convolution", create_convolution_shared_kernel};

struct InternedKernel {
    const SharedKernelFactory* factory;
    std::string payload;
    std::shared_ptr<SharedKernel> shared_kernel;
};

std::mutex interner_mutex;
std::unordered_map<int64_t, std::vector<InternedKernel>> kernel_interner;

std::shared_ptr<SharedKernel> intern_kernel(
    const SharedKernelFactory& factory, std::string_view payload, int64_t hash) {
    std::lock_guard<std::mutex> lock(interner_mutex);
    auto& bucket = kernel_interner[hash];
    // The hash chooses a bucket, but only exact family and payload matches share work.
    for (const InternedKernel& interned : bucket) {
        if (interned.factory == &factory && interned.payload == payload) {
            interner_hits.fetch_add(1, std::memory_order_relaxed);
            return interned.shared_kernel;
        }
    }
    auto shared_kernel = factory.create(payload);
    interner_misses.fetch_add(1, std::memory_order_relaxed);
    bucket.push_back({&factory, std::string(payload), shared_kernel});
    return shared_kernel;
}

ffi::ErrorOr<std::unique_ptr<OeqExecutableState>> instantiate_kernel(
    const SharedKernelFactory& factory, std::string_view payload, int64_t hash) {
    try {
        // Instantiation is the only lifecycle step that consults the global
        // interner. The returned state holds only the shared pointer.
        return std::make_unique<OeqExecutableState>(
            intern_kernel(factory, payload, hash));
    } catch (const std::exception& error) {
        return ffi::Unexpected(ffi::Error::InvalidArgument(error.what()));
    }
}

template <typename SharedKernelType, typename Kernel>
ffi::ErrorOr<Kernel> load_typed_kernel(
    OeqExecutableState* state, int32_t device_ordinal) {
    try {
        // Execution uses only the state. It schedules missing target work,
        // waits for it, and reuses the loaded launcher.
        if (state == nullptr || state->shared_kernel == nullptr) {
            throw std::logic_error("OpenEquivariance executable state is missing");
        }
        const auto shared_kernel =
            std::dynamic_pointer_cast<SharedKernelType>(state->shared_kernel);
        if (shared_kernel == nullptr) {
            throw std::logic_error("OpenEquivariance executable state has the wrong family");
        }
        return Kernel{shared_kernel->load(device_ordinal),
                      &shared_kernel->properties()};
    } catch (const std::exception& error) {
        return ffi::Unexpected(ffi::Error::Internal(error.what()));
    }
}

}  // namespace

ffi::ErrorOr<std::unique_ptr<OeqExecutableState>> instantiate_tensor_product(
    std::string_view payload, int64_t hash) {
    return instantiate_kernel(kTensorProductFactory, payload, hash);
}

ffi::ErrorOr<std::unique_ptr<OeqExecutableState>> instantiate_convolution(
    std::string_view payload, int64_t hash) {
    return instantiate_kernel(kConvolutionFactory, payload, hash);
}

ffi::Error initialize_kernel_state(
    OeqExecutableState* state, int32_t device_ordinal) {
    try {
        // Prepares work for this device. CUDA queues compilation; HIP loads
        // synchronously.
        if (state == nullptr || state->shared_kernel == nullptr) {
            throw std::logic_error("OpenEquivariance executable state is missing");
        }
        state->shared_kernel->initialize(device_ordinal);
        return ffi::Error::Success();
    } catch (const std::exception& error) {
        return ffi::Error::Internal(error.what());
    }
}

ffi::ErrorOr<TensorProductKernel> tensor_product_kernel(
    OeqExecutableState* state, int32_t device_ordinal) {
    return load_typed_kernel<TensorProductSharedKernel, TensorProductKernel>(
        state, device_ordinal);
}

ffi::ErrorOr<ConvolutionKernel> convolution_kernel(
    OeqExecutableState* state, int32_t device_ordinal) {
    return load_typed_kernel<ConvolutionSharedKernel, ConvolutionKernel>(
        state, device_ordinal);
}

KernelCompilationStatistics kernel_compilation_statistics() {
    return {
        interner_hits.load(std::memory_order_relaxed),
        interner_misses.load(std::memory_order_relaxed),
        compilations_started.load(std::memory_order_relaxed),
    };
}
