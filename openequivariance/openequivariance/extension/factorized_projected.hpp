#pragma once

#include <cstdint>
#include <string>
#include <vector>

template<typename JIT_IMPL>
class __attribute__ ((visibility ("default"))) JITFactorizedProjectedImpl {
public:
    JIT_IMPL jit;

    static std::vector<std::string> kernel_entry_points() {
        return {
            "oeq_projected_forward", "oeq_projected_forward_jvp",
            "oeq_projected_spatial_backward", "oeq_projected_weight_backward",
            "oeq_projected_spatial_backward_jvp"};
    }

    static std::vector<std::vector<int>> kernel_template_parameters() {
        return std::vector<std::vector<int>>(kernel_entry_points().size());
    }

#ifdef CUDA_BACKEND
    static CompiledKernelImage compile_image(
            const std::string& source, int cu_major, int cu_minor) {
        return CompiledKernelImage::compile(
            source, kernel_entry_points(), kernel_template_parameters(),
            cu_major, cu_minor, 3, false);
    }
#endif

    JITFactorizedProjectedImpl(std::string source) : jit(std::move(source), false) {
        jit.compile(kernel_entry_points(), kernel_template_parameters());
    }

#ifdef CUDA_BACKEND
    explicit JITFactorizedProjectedImpl(const CompiledKernelImage& image)
        : jit(image) {}
#endif

    void forward(
            int64_t nodes, int64_t edges, int64_t channels,
            void* x, void* sh, void* weights, void* senders, void* row_ptr,
            void* out, Stream stream) {
        void* args[] = {&nodes, &edges, &x, &sh, &weights, &senders, &row_ptr, &out};
        execute(0, nodes * channels, args, stream);
    }

    void forward_jvp(
            int64_t nodes, int64_t edges, int64_t channels,
            void* x, void* sh, void* weights, void* senders, void* row_ptr,
            void* tx, void* tsh, void* tweights, void* out, Stream stream) {
        void* args[] = {
            &nodes, &edges, &x, &sh, &weights, &senders, &row_ptr,
            &tx, &tsh, &tweights, &out};
        execute(1, nodes * channels, args, stream);
    }

    void spatial_backward(
            int64_t nodes, int64_t edges,
            void* x, void* sh, void* weights, void* senders, void* receivers,
            void* dout, void* dx, void* dsh, void* dweights, Stream stream) {
        void* args[] = {
            &nodes, &edges, &x, &sh, &weights, &senders, &receivers,
            &dout, &dx, &dsh, &dweights};
        execute(2, edges * kLogicalCohortWidth, args, stream);
    }

    void weight_backward(
            int64_t nodes, int64_t edges, int64_t channels,
            void* x, void* sh, void* senders, void* receivers, void* dout,
            void* dweights, Stream stream) {
        void* args[] = {
            &nodes, &edges, &x, &sh, &senders, &receivers, &dout, &dweights};
        execute(3, edges * channels, args, stream);
    }

    void spatial_backward_jvp(
            int64_t nodes, int64_t edges,
            void* x, void* sh, void* weights, void* senders, void* receivers,
            void* dout, void* tx, void* tsh, void* tweights, void* tdout,
            void* tdx, void* tdsh, void* tdweights, Stream stream) {
        void* args[] = {
            &nodes, &edges, &x, &sh, &weights, &senders, &receivers, &dout,
            &tx, &tsh, &tweights, &tdout, &tdx, &tdsh, &tdweights};
        execute(4, edges * kLogicalCohortWidth, args, stream);
    }

private:
    static constexpr int64_t kLogicalCohortWidth = 32;
    static constexpr int64_t kThreadsPerBlock = 128;

    void execute(int kernel_index, int64_t work_items, void* args[], Stream stream) {
        if (work_items == 0)
            return;
        const int64_t blocks =
            (work_items + kThreadsPerBlock - 1) / kThreadsPerBlock;
        jit.execute(
            kernel_index, args,
            with_stream(KernelLaunchConfig(blocks, kThreadsPerBlock, 0), stream));
    }
};
