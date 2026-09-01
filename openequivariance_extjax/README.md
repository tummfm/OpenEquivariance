# OpenEquivariance JAX Extension

The JAX extension module for OpenEquivariance.

## PJRT FFI provider

The `libjcn_ffi_openequivariance.so` Bazel target builds a Python-independent
FFI provider against the XLA and CUDA repositories selected by the caller's
Bazel configuration:

```text
bazel build @openequivariance_src//openequivariance_extjax:libjcn_ffi_openequivariance.so
```

The handler implementation owns one ABI-versioned table with every XLA stage
(`instantiate`, `prepare`, `initialize`, and `execute`) plus each target's
traits. The nanobind extension reads that table and publishes only stages that
have a handler. The shared library exports `RegisterFFi(const PJRT_Api*, const
char*)`; a runtime loader calls this entry point after loading its PJRT plugin.
The provider registers every target in the same table and prints each
registered target.

The provider prefers PJRT's handler-bundle registration when its trailing
function pointer is present in the advertised extension size. That path passes
every non-null stage and the traits from the table. Older PJRT extensions can
register execute-only targets, but the provider rejects a staged target rather
than silently losing its initialization. JAX registration uses every non-null
stage from the same table.
