import threading

import jax
import openequivariance_extjax as oeq_extjax

IS_HIP = oeq_extjax.is_hip()

platform = "CUDA"
if IS_HIP:
    platform = "ROCM"

if not hasattr(jax.ffi, "register_ffi_type"):
    raise RuntimeError("OpenEquivariance JAX support requires JAX 0.8.2 or newer")

_registration_lock = threading.Lock()
_ffi_registered = False


def ensure_ffi_registered() -> None:
    """Register OEQ state types and handlers once the GPU backend is ready."""
    global _ffi_registered
    with _registration_lock:
        if _ffi_registered:
            return
        # The public registration calls become ordered direct calls after the
        # selected PJRT plugin is initialized. This avoids JAX's independent
        # pending queues for state types and handlers.
        jax.devices(platform.lower())
        for name, registration in oeq_extjax.type_registrations().items():
            jax.ffi.register_ffi_type(name, registration, platform=platform)
        for name, target in oeq_extjax.registrations().items():
            jax.ffi.register_ffi_target(
                name, target, platform=platform, api_version=1
            )
        _ffi_registered = True

GPUTimer = oeq_extjax.GPUTimer
DeviceProp = oeq_extjax.DeviceProp

__all__ = [
    "GPUTimer",
    "DeviceProp",
    "ensure_ffi_registered",
]
