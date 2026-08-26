#pragma once

#include <cstdint>
#include <string>
#include <vector>

template<typename JIT_IMPL>
class __attribute__ ((visibility ("default"))) JITSymmetricLiteralImpl {
public:
    JIT_IMPL jit;

    static std::vector<std::string> kernel_entry_points() {
        return {
            "oeq_symmetric_literal_forward_species",
            "oeq_symmetric_literal_forward_jvp_x_species",
            "oeq_symmetric_literal_backward_x_species",
            "oeq_symmetric_literal_backward_jvp_x_species",
            "oeq_symmetric_literal_backward_hvp_x_species",
            "oeq_symmetric_literal_backward_jvp_xw_species",
            "oeq_symmetric_literal_backward_jvp_xw_transpose_species"};
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

    JITSymmetricLiteralImpl(std::string source) : jit(std::move(source), false) {
        jit.compile(kernel_entry_points(), kernel_template_parameters());
    }

#ifdef CUDA_BACKEND
    explicit JITSymmetricLiteralImpl(const CompiledKernelImage& image)
        : jit(image) {}
#endif

    void forward_species(
            int64_t nodes, int64_t num_elements, int64_t channels,
            void* x, void* species, void* weights, void* out, Stream stream) {
        void* args[] = {&nodes, &num_elements, &x, &species, &weights, &out};
        execute(0, work_items(nodes, channels, 1), args, stream);
    }

    void forward_jvp_x_species(
            int64_t nodes, int64_t num_elements, int64_t channels,
            void* x, void* species, void* weights, void* tx, void* tout,
            Stream stream) {
        void* args[] = {
            &nodes, &num_elements, &x, &species, &weights, &tx, &tout};
        execute(1, work_items(nodes, channels, 2), args, stream);
    }

    void backward_x_species(
            int64_t nodes, int64_t num_elements, int64_t channels,
            void* x, void* species, void* weights, void* dout, void* dx,
            Stream stream) {
        void* args[] = {
            &nodes, &num_elements, &x, &species, &weights, &dout, &dx};
        execute(2, work_items(nodes, channels, 1), args, stream);
    }

    void backward_jvp_x_species(
            int64_t nodes, int64_t num_elements, int64_t channels,
            void* x, void* species, void* weights, void* dout, void* tx,
            void* tdout, void* tdx, Stream stream) {
        void* args[] = {
            &nodes, &num_elements, &x, &species, &weights, &dout,
            &tx, &tdout, &tdx};
        execute(3, work_items(nodes, channels, 2), args, stream);
    }

    void backward_hvp_x_species(
            int64_t nodes, int64_t num_elements, int64_t channels,
            void* x, void* species, void* weights, void* dout, void* tx,
            void* tdx, Stream stream) {
        void* args[] = {
            &nodes, &num_elements, &x, &species, &weights, &dout, &tx, &tdx};
        execute(4, work_items(nodes, channels, 1), args, stream);
    }

    void backward_jvp_xw_species(
            int64_t nodes, int64_t num_elements, int64_t channels,
            void* x, void* species, void* weights, void* dout, void* tx,
            void* tweights, void* tdout, void* tdx, Stream stream) {
        void* args[] = {
            &nodes, &num_elements, &x, &species, &weights, &dout,
            &tx, &tweights, &tdout, &tdx};
        execute(5, work_items(nodes, channels, 2), args, stream);
    }

    void backward_jvp_xw_transpose_species(
            int64_t nodes, int64_t num_elements, int64_t channels,
            void* x, void* species, void* weights, void* dout, void* ctdx,
            void* ctx, void* ctweights, void* ctdout, Stream stream) {
        void* args[] = {
            &nodes, &num_elements, &x, &species, &weights, &dout,
            &ctdx, &ctx, &ctweights, &ctdout};
        execute(6, work_items(nodes, channels, 2), args, stream);
    }

private:
    static constexpr int64_t kLogicalCohortWidth = 32;
    static constexpr int64_t kThreadsPerBlock = 128;

    int64_t work_items(int64_t nodes, int64_t channels, int64_t path_lanes) {
        const int64_t channels_per_cohort = kLogicalCohortWidth / path_lanes;
        return nodes * ((channels + channels_per_cohort - 1) / channels_per_cohort) *
               kLogicalCohortWidth;
    }

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
