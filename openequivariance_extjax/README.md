# OpenEquivariance JAX Extension

The JAX extension module for OpenEquivariance.

## PJRT FFI provider

The `libjcn_ffi_openequivariance.so` Bazel target builds a Python-independent
FFI provider against the XLA and CUDA repositories selected by the caller's
Bazel configuration:

```text
bazel build @openequivariance_src//openequivariance_extjax:libjcn_ffi_openequivariance.so
```

The shared library exports `RegisterFFi(const PJRT_Api*, const char*)`. A
runtime loader calls this entry point after loading its PJRT plugin. The
provider registers every target in the OpenEquivariance handler table and
prints each registered target.

The current PJRT C FFI extension accepts an execute handler for each target,
but it does not accept an XLA FFI handler bundle. The provider therefore uses
the execute handlers and retains their lazy initialization. JAX registration
continues to use the separate initialization handlers from the same table.
