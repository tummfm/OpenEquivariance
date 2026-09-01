#include "nanobind/nanobind.h"

#include "ffi_handler_table.h"
#include "kernel_compilation.h"

#ifdef CUDA_BACKEND
#include "backend/backend_cuda.hpp"
#endif

#ifdef HIP_BACKEND
#include "backend/backend_hip.hpp"
#endif

namespace nb = nanobind;

namespace {

bool IsHip() {
#ifdef HIP_BACKEND
    return true;
#else
    return false;
#endif
}

void AddStage(nb::dict& stages, const char* name, void* handler) {
    if (handler != nullptr) {
        stages[name] = nb::capsule(handler);
    }
}

}  // namespace

NB_MODULE(openequivariance_extjax, m) {
    m.def("compilation_statistics", []() {
        const KernelCompilationStatistics statistics =
            kernel_compilation_statistics();
        nb::dict result;
        result["interner_hits"] = statistics.interner_hits;
        result["interner_misses"] = statistics.interner_misses;
        result["compilations_started"] = statistics.compilations_started;
        return result;
    });
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
            AddStage(stages, "instantiate", handler.instantiate);
            AddStage(stages, "prepare", handler.prepare);
            AddStage(stages, "initialize", handler.initialize);
            AddStage(stages, "execute", handler.execute);
            registrations[handler.name] = stages;
        }
        return registrations;
    });
    m.def("is_hip", &IsHip);

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
