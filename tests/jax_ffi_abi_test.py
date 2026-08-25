"""Runtime checks for the OEQ-owned C handler table and nanobind registry."""

import ctypes
from pathlib import Path
import runpy

import pytest


FFI_TARGETS = runpy.run_path(
    Path(__file__).parents[1]
    / "openequivariance"
    / "openequivariance"
    / "jax"
    / "ffi_targets.py"
)["FFI_TARGETS"]


class _Handler(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("initialize", ctypes.c_void_p),
        ("execute", ctypes.c_void_p),
    ]


class _HandlerTable(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("handler_count", ctypes.c_uint32),
        ("handlers", ctypes.POINTER(_Handler)),
    ]


def _handler_table():
    ext = pytest.importorskip("openequivariance_extjax")
    library = ctypes.CDLL(ext.__file__)
    table_fn = library.oeq_ffi_handler_table
    table_fn.restype = ctypes.POINTER(_HandlerTable)
    return ext, table_fn().contents


def _capsule_pointer(capsule):
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    get_pointer.restype = ctypes.c_void_p
    return get_pointer(capsule, None)


def test_exported_handler_table_matches_manifest(with_jax):
    """The ABI-v1 table is the ordered source of truth for every target."""
    if not with_jax:
        pytest.skip("JAX ABI checks require --jax")

    _, table = _handler_table()
    names = [table.handlers[index].name.decode() for index in range(table.handler_count)]
    assert table.abi_version == 1
    assert tuple(names) == FFI_TARGETS
    assert len(names) == len(set(names)) == 6
    assert all(table.handlers[index].initialize for index in range(table.handler_count))
    assert all(table.handlers[index].execute for index in range(table.handler_count))


def test_nanobind_registrations_match_handler_table(with_jax):
    """Nanobind exposes exactly the ABI table's staged handler values."""
    if not with_jax:
        pytest.skip("JAX ABI checks require --jax")

    ext, table = _handler_table()
    registrations = ext.registrations()
    assert tuple(registrations) == FFI_TARGETS
    assert len(registrations) == table.handler_count == 6
    for index, name in enumerate(FFI_TARGETS):
        handler = table.handlers[index]
        registration = registrations[name]
        assert set(registration) == {"initialize", "execute"}
        assert _capsule_pointer(registration["initialize"]) == handler.initialize
        assert _capsule_pointer(registration["execute"]) == handler.execute


def test_handler_families_share_one_initializer(with_jax):
    """Each compiled kernel family exposes one common initialization stage."""
    if not with_jax:
        pytest.skip("JAX ABI checks require --jax")

    _, table = _handler_table()
    handlers = {
        table.handlers[index].name.decode(): table.handlers[index]
        for index in range(table.handler_count)
    }
    for family in (
        ("tp_forward", "tp_backward", "tp_double_backward"),
        ("conv_forward", "conv_backward", "conv_double_backward"),
    ):
        assert len({handlers[name].initialize for name in family}) == 1
