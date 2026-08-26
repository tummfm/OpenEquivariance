#pragma once

#include <cstdint>
#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm> 
#include <stdexcept>
#include <utility>

using namespace std;
using Stream = cudaStream_t;

#define NVRTC_SAFE_CALL(x)                                      \
do {                                                            \
   nvrtcResult result = x;                                      \
   if (result != NVRTC_SUCCESS) {                               \
      std::cerr << "\nerror: " #x " failed with error "         \
               << nvrtcGetErrorString(result) << '\n';          \
      exit(1);                                                  \
   }                                                            \
} while(0)

#define CUDA_SAFE_CALL(x)                                     \
do {                                                            \
   CUresult result = x;                                         \
   if (result != CUDA_SUCCESS) {                                \
      const char *msg;                                          \
      cuGetErrorName(result, &msg);                             \
      std::cerr << "\nerror: " #x " failed with error "         \
               << msg << '\n';                                  \
      exit(1);                                                  \
   }                                                            \
} while(0)

#define CUDA_ERRCHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

class CUDA_Allocator {
public:
    static void* gpu_alloc (size_t size) {
        void* ptr;
        CUDA_ERRCHK( cudaMalloc((void**) &ptr, size ))
        return ptr;
    }

    static void gpu_free (void* ptr) {
        CUDA_ERRCHK( cudaFree(ptr))
    }

    static void copy_host_to_device (void* host, void* device, size_t size) {
        CUDA_ERRCHK( cudaMemcpy(device, host, size, cudaMemcpyHostToDevice));
    }

    static void copy_device_to_host (void* host, void* device, size_t size) {
        CUDA_ERRCHK( cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
    }
};

class GPUTimer {
    cudaEvent_t start_evt, stop_evt;

public:
    GPUTimer() {  
        cudaEventCreate(&start_evt);
        cudaEventCreate(&stop_evt);
    }

    void start() {
        cudaEventRecord(start_evt);
    }

    float stop_clock_get_elapsed() {
        float time_millis;
        cudaEventRecord(stop_evt);
        cudaEventSynchronize(stop_evt);
        cudaEventElapsedTime(&time_millis, start_evt, stop_evt);
        return time_millis; 
    }

    void clear_L2_cache() {
        size_t element_count = 25000000;

        int* ptr = (int*) (CUDA_Allocator::gpu_alloc(element_count * sizeof(int)));
        CUDA_ERRCHK(cudaMemset(ptr, 42, element_count * sizeof(int)))
        CUDA_Allocator::gpu_free(ptr);
        cudaDeviceSynchronize();
    }
    
    ~GPUTimer() {
        cudaEventDestroy(start_evt);
        cudaEventDestroy(stop_evt);
    }
};

class __attribute__((visibility("default"))) DeviceProp {
public:
    std::string name; 
    int warpsize;
    int major, minor;
    int multiprocessorCount;
    int maxSharedMemPerBlock;
    int maxSharedMemoryPerMultiprocessor; 

    DeviceProp(int device_id) {
        cudaDeviceProp prop; 
        cudaGetDeviceProperties(&prop, device_id);
        name = std::string(prop.name);
        CUDA_ERRCHK(cudaDeviceGetAttribute(&maxSharedMemoryPerMultiprocessor, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device_id));
        CUDA_ERRCHK(cudaDeviceGetAttribute(&maxSharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id));
        CUDA_ERRCHK(cudaDeviceGetAttribute(&warpsize, cudaDevAttrWarpSize, device_id));
        CUDA_ERRCHK(cudaDeviceGetAttribute(&multiprocessorCount, cudaDevAttrMultiProcessorCount, device_id));
        CUDA_ERRCHK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id));
        CUDA_ERRCHK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id));
    }
};

class __attribute__((visibility("default"))) KernelLaunchConfig {
public:
    uint32_t num_blocks = 0;
    uint32_t num_threads = 0;
    uint32_t warp_size = 32;
    uint32_t smem = 0;
    CUstream hStream = NULL;

    KernelLaunchConfig() = default;
    ~KernelLaunchConfig() = default;

    KernelLaunchConfig(uint32_t num_blocks, uint32_t num_threads_per_block, uint32_t smem) :
        num_blocks(num_blocks),
        num_threads(num_threads_per_block),
        smem(smem) 
    { }

    KernelLaunchConfig(int64_t num_blocks_i, int64_t num_threads_i, int64_t smem_i) :
        KernelLaunchConfig( static_cast<uint32_t>(num_blocks_i),
                            static_cast<uint32_t>(num_threads_i),
                            static_cast<uint32_t>(smem_i)) 
    { }
};

/*
* This page is a useful resource on NVRTC: 
* https://docs.nvidia.com/cuda/nvrtc/index.html#example-using-nvrtcgettypename
*/

class __attribute__((visibility("default"))) CompiledKernelImage {
private:
    // Worker-side NVRTC failures must throw so an asynchronous caller can
    // propagate them through its future instead of terminating the process.
    static void check_nvrtc(nvrtcResult result, const char* operation) {
        if(result != NVRTC_SUCCESS) {
            throw std::runtime_error(
                string(operation) + " failed with error " +
                nvrtcGetErrorString(result));
        }
    }

    // Destroy the NVRTC program on success and on every exceptional exit.
    struct Program {
        nvrtcProgram handle = nullptr;

        ~Program() {
            if(handle != nullptr) {
                nvrtcDestroyProgram(&handle);
            }
        }
    };

public:
    vector<char> cubin;
    vector<string> lowered_kernel_names;
    // Record the cubin target so dynamic shared-memory attributes can be
    // applied to every compatible device before the launcher is shared.
    int cu_major;
    int cu_minor;

    CompiledKernelImage(
        vector<char> cubin_i,
        vector<string> lowered_kernel_names_i,
        int cu_major_i,
        int cu_minor_i) :
        cubin(std::move(cubin_i)),
        lowered_kernel_names(std::move(lowered_kernel_names_i)),
        cu_major(cu_major_i),
        cu_minor(cu_minor_i) { }

    // NVRTC compilation and name lowering are host-only. Compiler workers can
    // produce this image without a CUDA context. CUDA-library handle creation
    // is deferred to CUJITKernel outside those workers.
    static CompiledKernelImage compile(
        const string& kernel_plaintext,
        const vector<string>& kernel_names_i,
        const vector<vector<int>>& template_param_list,
        int cu_major,
        int cu_minor,
        int opt_level=3,
        bool fast_math=true) {
        (void)opt_level;

        if(kernel_names_i.size() != template_param_list.size()) {
            throw std::logic_error("Kernel names and template parameters must have the same size!");
        }

        int num_supported_archs;
        check_nvrtc(
            nvrtcGetNumSupportedArchs(&num_supported_archs),
            "nvrtcGetNumSupportedArchs");
        vector<int> supported_archs(num_supported_archs);
        check_nvrtc(
            nvrtcGetSupportedArchs(supported_archs.data()),
            "nvrtcGetSupportedArchs");

        int device_arch = cu_major * 10 + cu_minor;
        if (std::find(supported_archs.begin(), supported_archs.end(), device_arch) == supported_archs.end()){
            int nvrtc_version_major, nvrtc_version_minor;
            check_nvrtc(
                nvrtcVersion(&nvrtc_version_major, &nvrtc_version_minor),
                "nvrtcVersion");

            throw std::runtime_error("NVRTC version "
                + std::to_string(nvrtc_version_major)
                + "."
                + std::to_string(nvrtc_version_minor)
                + " does not support device architecture "
                + std::to_string(device_arch));
        }

        vector<string> kernel_names;
        kernel_names.reserve(kernel_names_i.size());
        for(size_t kernel = 0; kernel < kernel_names_i.size(); kernel++) {
            const string& kernel_name = kernel_names_i[kernel];
            const vector<int>& template_params = template_param_list[kernel];

            if(template_params.empty()) {
                kernel_names.push_back(kernel_name);
                continue;
            }

            string result = kernel_name + "<";
            for(size_t i = 0; i < template_params.size(); i++) {
                result += std::to_string(template_params[i]);
                if(i + 1 != template_params.size()) {
                    result += ",";
                }
            }
            kernel_names.push_back(result + ">");
        }

        Program program;
        check_nvrtc(nvrtcCreateProgram(
            &program.handle,
            kernel_plaintext.c_str(),
            "kernel.cu",
            0,
            NULL,
            NULL), "nvrtcCreateProgram");

        string sm = "-arch=sm_" + std::to_string(cu_major) + std::to_string(cu_minor);
        vector<const char*> opts = {
            "--std=c++17",
            sm.c_str(),
            "--split-compile=0"
        };
        if(fast_math) {
            opts.push_back("--use_fast_math");
        }

        for(const string& kernel_name : kernel_names) {
            check_nvrtc(
                nvrtcAddNameExpression(program.handle, kernel_name.c_str()),
                "nvrtcAddNameExpression");
        }

        nvrtcResult compileResult = nvrtcCompileProgram(
            program.handle, static_cast<int>(opts.size()), opts.data());

        size_t log_size;
        check_nvrtc(
            nvrtcGetProgramLogSize(program.handle, &log_size),
            "nvrtcGetProgramLogSize");
        vector<char> log(log_size);
        check_nvrtc(
            nvrtcGetProgramLog(program.handle, log.data()),
            "nvrtcGetProgramLog");

        if (compileResult != NVRTC_SUCCESS) {
            throw std::logic_error("NVRTC Fail, log: " + string(log.data()));
        }

        size_t cubin_size;
        check_nvrtc(
            nvrtcGetCUBINSize(program.handle, &cubin_size),
            "nvrtcGetCUBINSize");
        vector<char> cubin(cubin_size);
        check_nvrtc(
            nvrtcGetCUBIN(program.handle, cubin.data()),
            "nvrtcGetCUBIN");

        vector<string> lowered_kernel_names;
        lowered_kernel_names.reserve(kernel_names.size());
        for(const string& kernel_name : kernel_names) {
            const char* lowered_name;
            check_nvrtc(nvrtcGetLoweredName(
                program.handle, kernel_name.c_str(), &lowered_name),
                "nvrtcGetLoweredName");
            lowered_kernel_names.emplace_back(lowered_name);
        }

        return CompiledKernelImage(
            std::move(cubin), std::move(lowered_kernel_names),
            cu_major, cu_minor);
    }
};

class __attribute__((visibility("default"))) CUJITKernel {
private:
    bool loaded = false;
    int cu_major = 0;
    int cu_minor = 0;
    CUlibrary library = nullptr;
    vector<CUkernel> kernels;

    void load(const CompiledKernelImage& image) {
        if(loaded) {
            throw std::logic_error("JIT object has already been loaded!");
        }

        CUDA_SAFE_CALL(cuInit(0));
        cu_major = image.cu_major;
        cu_minor = image.cu_minor;
        CUDA_SAFE_CALL(cuLibraryLoadData(
            &library, image.cubin.data(), 0, 0, 0, 0, 0, 0));

        kernels.reserve(image.lowered_kernel_names.size());
        for(const string& lowered_name : image.lowered_kernel_names) {
            CUkernel kernel;
            CUDA_SAFE_CALL(cuLibraryGetKernel(&kernel, library, lowered_name.c_str()));
            kernels.push_back(kernel);
        }
        loaded = true;
    }

public:
    string kernel_plaintext;
    bool fast_math = true;

    CUJITKernel(string plaintext, bool fast_math_i=true) :
        kernel_plaintext(std::move(plaintext)), fast_math(fast_math_i) { }

    // Load context-less CUkernel handles from a host-compiled CUDA image. The
    // caller-provided launch stream later selects the CUDA context.
    CUJITKernel(const CompiledKernelImage& image) {
        load(image);
    }

    void compile(string kernel_name, const vector<int> template_params, int opt_level=3) {
        vector<string> kernel_names = {kernel_name};
        vector<vector<int>> template_param_list = {template_params};
        compile(kernel_names, template_param_list, opt_level);
    }

    void compile(
        const vector<string>& kernel_names_i,
        const vector<vector<int>>& template_param_list,
        int opt_level=3) {
        if(loaded) {
            throw std::logic_error("JIT object has already been compiled!");
        }

        // Preserve the synchronous constructor/compile path used by Torch and
        // other callers that do not use the staged FFI lifecycle.
        int device_id;
        CUDA_ERRCHK(cudaGetDevice(&device_id));
        DeviceProp device_prop(device_id);
        CompiledKernelImage image = CompiledKernelImage::compile(
            kernel_plaintext,
            kernel_names_i,
            template_param_list,
            device_prop.major,
            device_prop.minor,
            opt_level,
            fast_math);
        load(image);
    }

    void set_max_smem(int kernel_id, uint32_t max_smem_bytes) {
        if(!loaded)
            throw std::logic_error("JIT object has not been compiled!");
        if(kernel_id >= kernels.size())
            throw std::logic_error("Kernel index out of range!");

        int device_count;
        CUDA_SAFE_CALL(cuDeviceGetCount(&device_count));
        for(int device_id = 0; device_id < device_count; device_id++) {
            DeviceProp device_prop(device_id);
            if(device_prop.major == cu_major && device_prop.minor == cu_minor) {
                // The image is shared by matching architectures. Request the
                // dynamic shared-memory limit on every compatible device.
                CUdevice device;
                CUDA_SAFE_CALL(cuDeviceGet(&device, device_id));
                CUDA_SAFE_CALL(cuKernelSetAttribute(
                    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    max_smem_bytes,
                    kernels[kernel_id],
                    device));
            }
        }
    }

    void execute(int kernel_id, void* args[], KernelLaunchConfig config) {
        if(!loaded)
            throw std::logic_error("JIT object has not been compiled!");
        if(kernel_id >= kernels.size())
            throw std::logic_error("Kernel index out of range!");

        // The caller-provided stream selects the CUDA context and ordering.
        CUDA_SAFE_CALL(
            cuLaunchKernel( (CUfunction) (kernels[kernel_id]),
                            config.num_blocks, 1, 1,    // grid dim
                            config.num_threads, 1, 1,   // block dim
                            config.smem, config.hStream,       // shared mem and stream
                            args, NULL)          // arguments
        );            
    }

    ~CUJITKernel() {
        if(loaded) {
            auto result = cuLibraryUnload(library);
            if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) {
                std::cout << "Failed to unload CUDA library, error code: " << ((int) result) << std::endl; 
            }
        }
    }
};

KernelLaunchConfig with_stream(const KernelLaunchConfig& config, Stream stream) {
    KernelLaunchConfig new_config = config;
    new_config.hStream = stream;
    return new_config;
}
