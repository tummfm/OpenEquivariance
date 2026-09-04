# OpenEquivariance JAX Extension

The JAX extension module for OpenEquivariance.

## XLA FFI provider

The `libjcn_ffi_openequivariance.so` Bazel target builds a Python-independent
FFI provider against the XLA and CUDA repositories selected by the caller's
Bazel configuration:

```text
bazel build @openequivariance_src//openequivariance_extjax:libjcn_ffi_openequivariance.so
```

The handler implementation owns one ABI-versioned table with slots for every
XLA stage (`instantiate`, `prepare`, `initialize`, and `execute`) plus each
target's traits. The nanobind extension reads that table and publishes only
stages that have a handler. The shared library exports `RegisterFFi(const
XLA_FFI_Api*, const char*)`; a runtime loader obtains the API from its PJRT
plugin and calls this entry point. The provider uses
`XLA_FFI_Handler_Register` to register every target as a complete handler
bundle, passing every non-null stage and the traits from the table. JAX
registration uses the stages from the same table.

## JAX kernel compilation

On CUDA, JAX compiles generated kernels in a bounded worker pool. The following
environment variables control its parallelism:

- `OEQ_JAX_COMPILER_THREADS` sets the number of kernels compiled in parallel.
  It accepts values from 1 to 64 and defaults to 8.
- `OEQ_JAX_COMPILER_QUEUE_CAPACITY` sets the number of waiting compilation
  jobs. It accepts values from 1 to 256 and defaults to 32. Submission waits
  when the queue is full.

For example, the following configuration compiles up to 16 kernels at once:

```text
OEQ_JAX_COMPILER_THREADS=16 python application.py
```

These variables are read once when compilation first starts. The HIP path
continues to compile synchronously.
