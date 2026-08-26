"""Numerical and lowering coverage for projected factorized convolution."""

from types import SimpleNamespace

import numpy as np
import pytest


_TOLERANCES = {
    np.dtype(np.float32): {"atol": 5e-3, "rtol": 5e-4},
    np.dtype(np.float64): {"atol": 4e-11, "rtol": 4e-11},
}


@pytest.fixture(scope="module")
def gpu_context(with_jax):
    """Provide JAX only for explicitly enabled GPU tests."""
    if not with_jax:
        pytest.skip("requires --jax")
    import jax
    import jax.numpy as jnp

    if jax.default_backend() != "gpu":
        pytest.skip("requires a JAX GPU backend")
    return jax, jnp


def _case(gpu_context, dtype):
    jax, jnp = gpu_context
    from openequivariance.benchmark.problems import ChannelwiseTPP
    from openequivariance.jax.FactorizedTensorProductConv import (
        FactorizedTensorProductConv,
    )
    from openequivariance.jax.TensorProductConv import TensorProductConv

    problem = ChannelwiseTPP(
        "3x0e + 3x1e",
        "1x0e + 1x1e",
        "3x0e + 3x1e",
        irrep_dtype=dtype,
        weight_dtype=dtype,
    )
    generated = FactorizedTensorProductConv(problem)
    reference = TensorProductConv(problem)
    keys = jax.random.split(jax.random.key(991), 12)
    nodes, edges, radial_dim = 5, 9, 4
    values = (
        jax.random.normal(keys[0], (nodes, problem.irreps_in1.dim), dtype=dtype),
        jax.random.normal(keys[1], (edges, problem.irreps_in2.dim), dtype=dtype),
        jax.random.normal(keys[2], (edges, radial_dim), dtype=dtype),
        jax.random.normal(keys[3], (radial_dim, problem.weight_numel), dtype=dtype),
    )
    tangents = tuple(
        jax.random.normal(key, value.shape, dtype=dtype)
        for key, value in zip(keys[4:8], values, strict=True)
    )
    return SimpleNamespace(
        generated=generated,
        reference=reference,
        values=values,
        tangents=tangents,
        rows=jnp.array([3, 0, 4, 1, 0, 2, 3, 1, 4], jnp.int32),
        cols=jnp.array([0, 2, 1, 3, 4, 0, 2, 4, 3], jnp.int32),
        dout=jax.random.normal(keys[8], (nodes, problem.irreps_out.dim), dtype=dtype),
        force_cotangents=tuple(
            jax.random.normal(key, value.shape, dtype=dtype)
            for key, value in zip(keys[9:], values[:3], strict=True)
        ),
    )


def _operators(case, jax, jnp):
    def generated(*args):
        return case.generated(*args, case.rows, case.cols)

    def reference(x, sh, radial, projection):
        return case.reference(
            x,
            sh,
            jnp.matmul(radial, projection, precision=jax.lax.Precision.HIGHEST),
            case.rows,
            case.cols,
        )

    return generated, reference


def _tolerance(dtype, multiplier=1):
    values = _TOLERANCES[np.dtype(dtype)]
    return {name: value * multiplier for name, value in values.items()}


@pytest.mark.parametrize("dtype", (np.float32, np.float64), ids=("f32", "f64"))
def test_projected_forward_jvp_vjp_and_grad(gpu_context, dtype):
    """Match f32/f64 forward and first-order transforms to TensorProductConv."""
    jax, jnp = gpu_context
    case = _case(gpu_context, dtype)
    generated, reference = _operators(case, jax, jnp)

    np.testing.assert_allclose(
        generated(*case.values), reference(*case.values), **_tolerance(dtype)
    )
    got_jvp = jax.jvp(generated, case.values, case.tangents)[1]
    want_jvp = jax.jvp(reference, case.values, case.tangents)[1]
    np.testing.assert_allclose(got_jvp, want_jvp, **_tolerance(dtype, 5))

    for active_indices in ((0,), (1,), (2, 3), (0, 1, 2, 3)):

        def bind(operator, *active_values):
            values = list(case.values)
            for index, value in zip(active_indices, active_values, strict=True):
                values[index] = value
            return operator(*values)

        primals = tuple(case.values[index] for index in active_indices)
        tangents = tuple(case.tangents[index] for index in active_indices)
        got = jax.jvp(lambda *values: bind(generated, *values), primals, tangents)[1]
        want = jax.jvp(lambda *values: bind(reference, *values), primals, tangents)[1]
        np.testing.assert_allclose(got, want, **_tolerance(dtype, 5))

    got_vjp = jax.vjp(generated, *case.values)[1](case.dout)
    want_vjp = jax.vjp(reference, *case.values)[1](case.dout)
    for got, want in zip(got_vjp, want_vjp, strict=True):
        np.testing.assert_allclose(got, want, **_tolerance(dtype, 6))

    def energy(operator, *args):
        return jnp.vdot(operator(*args), case.dout)

    got_grad = jax.grad(lambda *args: energy(generated, *args), (0, 1, 2, 3))(
        *case.values
    )
    want_grad = jax.grad(lambda *args: energy(reference, *args), (0, 1, 2, 3))(
        *case.values
    )
    for got, want in zip(got_grad, want_grad, strict=True):
        np.testing.assert_allclose(got, want, **_tolerance(dtype, 6))


@pytest.mark.parametrize("dtype", (np.float32, np.float64), ids=("f32", "f64"))
def test_projected_hvp_and_mixed_gradients(gpu_context, dtype):
    """Match f32/f64 HVP and force-loss mixed derivatives."""
    jax, jnp = gpu_context
    case = _case(gpu_context, dtype)
    generated, reference = _operators(case, jax, jnp)

    def energy(operator, *args):
        return jnp.vdot(operator(*args), case.dout)

    def hessian_vector(operator):
        gradient = jax.grad(lambda *args: energy(operator, *args), (0, 1, 2, 3))
        return jax.jvp(gradient, case.values, case.tangents)[1]

    for got, want in zip(
        hessian_vector(generated), hessian_vector(reference), strict=True
    ):
        np.testing.assert_allclose(got, want, **_tolerance(dtype, 12))

    def force_loss(operator, *args):
        force = jax.grad(lambda *values: energy(operator, *values), argnums=(0, 1, 2))(
            *args
        )
        return sum(
            jnp.vdot(value, cotangent)
            for value, cotangent in zip(force, case.force_cotangents, strict=True)
        )

    got_mixed = jax.grad(lambda *args: force_loss(generated, *args), (2, 3))(
        *case.values
    )
    want_mixed = jax.grad(lambda *args: force_loss(reference, *args), (2, 3))(
        *case.values
    )
    for got, want in zip(got_mixed, want_mixed, strict=True):
        np.testing.assert_allclose(got, want, **_tolerance(dtype, 12))


def test_projected_mixed_third_derivative(gpu_context):
    """Match an f32 mixed x, edge, and radial third derivative."""
    jax, jnp = gpu_context
    dtype = np.float32
    case = _case(gpu_context, dtype)
    generated, reference = _operators(case, jax, jnp)
    x, sh, radial, projection = case.values
    tx, tsh, tradial, _ = case.tangents

    def third_derivative(operator):
        def energy(x_value, sh_value, radial_value):
            return jnp.vdot(
                operator(x_value, sh_value, radial_value, projection), case.dout
            )

        return jax.jvp(
            lambda radial_value: jax.jvp(
                lambda sh_value: jax.jvp(
                    lambda x_value: energy(x_value, sh_value, radial_value),
                    (x,),
                    (tx,),
                )[1],
                (sh,),
                (tsh,),
            )[1],
            (radial,),
            (tradial,),
        )[1]

    np.testing.assert_allclose(
        third_derivative(generated),
        third_derivative(reference),
        **_tolerance(dtype, 12),
    )


def test_projected_padded_endpoints_are_zero_in_values_and_gradients(gpu_context):
    """Mask f32 negative and upper-bound padded endpoints and their gradients."""
    jax, jnp = gpu_context
    dtype = np.float32
    case = _case(gpu_context, dtype)
    x, sh, radial, projection = case.values
    valid_edges = sh.shape[0] - 2
    rows = case.rows.at[-2].set(-1).at[-1].set(x.shape[0])
    cols = case.cols.at[-2].set(x.shape[0]).at[-1].set(-1)

    def generated(x_value, sh_value, radial_value, projection_value):
        return case.generated(
            x_value, sh_value, radial_value, projection_value, rows, cols
        )

    def reference(x_value, sh_value, radial_value, projection_value):
        return case.reference(
            x_value,
            sh_value[:valid_edges],
            radial_value[:valid_edges] @ projection_value,
            case.rows[:valid_edges],
            case.cols[:valid_edges],
        )

    np.testing.assert_allclose(
        generated(x, sh, radial, projection),
        reference(x, sh, radial, projection),
        **_tolerance(dtype),
    )
    got = jax.grad(lambda *args: jnp.sum(generated(*args)), (0, 1, 2, 3))(
        x, sh, radial, projection
    )
    want = jax.grad(lambda *args: jnp.sum(reference(*args)), (0, 1, 2, 3))(
        x, sh, radial, projection
    )
    np.testing.assert_allclose(got[0], want[0], **_tolerance(dtype))
    np.testing.assert_allclose(
        got[1][:valid_edges], want[1][:valid_edges], **_tolerance(dtype)
    )
    np.testing.assert_allclose(got[1][valid_edges:], 0, **_tolerance(dtype))
    np.testing.assert_allclose(
        got[2][:valid_edges], want[2][:valid_edges], **_tolerance(dtype)
    )
    np.testing.assert_allclose(got[2][valid_edges:], 0, **_tolerance(dtype))
    np.testing.assert_allclose(got[3], want[3], **_tolerance(dtype))


def test_projected_runtime_topology_is_order_stable_and_empty_is_zero(gpu_context):
    """Preserve graph results under edge reordering and empty edge sets."""
    _, jnp = gpu_context
    case = _case(gpu_context, np.float32)
    x, sh, radial, projection = case.values
    order = jnp.array([2, 6, 0, 8, 3, 5, 1, 7, 4], dtype=jnp.int32)
    expected = case.generated(x, sh, radial, projection, case.rows, case.cols)
    actual = case.generated(
        x, sh[order], radial[order], projection, case.rows[order], case.cols[order]
    )
    np.testing.assert_allclose(actual, expected, **_tolerance(np.float32))
    empty = jnp.empty((0,), dtype=jnp.int32)
    np.testing.assert_allclose(
        case.generated(x, sh[:0], radial[:0], projection, empty, empty),
        jnp.zeros_like(expected),
        **_tolerance(np.float32),
    )


def test_projected_hlo_requests_representative_targets(gpu_context):
    """Lower forward, adjoint, and HVP paths to their required targets."""
    jax, jnp = gpu_context
    from openequivariance.jax.ffi_targets import (
        FACTORIZED_FORWARD,
        FACTORIZED_SPATIAL_BACKWARD,
        FACTORIZED_SPATIAL_BACKWARD_JVP,
        FACTORIZED_WEIGHT_BACKWARD,
    )

    case = _case(gpu_context, np.float32)
    generated, _ = _operators(case, jax, jnp)

    def energy(*args):
        return jnp.vdot(generated(*args), case.dout)

    def target_text(function, args):
        return str(jax.jit(function).lower(*args).compiler_ir(dialect="stablehlo"))

    forward_hlo = target_text(energy, case.values)
    assert FACTORIZED_FORWARD in forward_hlo
    assert FACTORIZED_SPATIAL_BACKWARD not in forward_hlo

    force = jax.grad(energy, (0, 1, 2))
    force_hlo = target_text(force, case.values)
    assert FACTORIZED_SPATIAL_BACKWARD in force_hlo

    parameter_hlo = target_text(jax.grad(energy, 3), case.values)
    assert FACTORIZED_WEIGHT_BACKWARD in parameter_hlo
    assert FACTORIZED_SPATIAL_BACKWARD not in parameter_hlo

    def hvp(*args):
        return jax.jvp(force, args, case.tangents)[1]

    assert FACTORIZED_SPATIAL_BACKWARD_JVP in target_text(hvp, case.values)


def test_projected_force_export_has_dynamic_nodes_and_edges(gpu_context):
    """Export one force transform for two runtime graph shapes."""
    jax, jnp = gpu_context
    from jax import export
    from openequivariance.jax.ffi_targets import (
        FACTORIZED_FORWARD,
        FACTORIZED_SPATIAL_BACKWARD,
    )

    case = _case(gpu_context, np.float32)
    radial_dim = case.values[2].shape[1]

    def force(x, sh, radial, projection, rows, cols):
        return jax.grad(
            lambda a, b, c: jnp.sum(case.generated(a, b, c, projection, rows, cols)),
            (0, 1, 2),
        )(x, sh, radial)

    nodes, edges = export.symbolic_shape(
        "nodes,edges", constraints=("nodes >= 1", "edges >= 1")
    )
    specs = (
        jax.ShapeDtypeStruct(
            (nodes, case.generated.config.irreps_in1.dim), jnp.float32
        ),
        jax.ShapeDtypeStruct(
            (edges, case.generated.config.irreps_in2.dim), jnp.float32
        ),
        jax.ShapeDtypeStruct((edges, radial_dim), jnp.float32),
        jax.ShapeDtypeStruct(
            (radial_dim, case.generated.config.weight_numel), jnp.float32
        ),
        jax.ShapeDtypeStruct((edges,), jnp.int32),
        jax.ShapeDtypeStruct((edges,), jnp.int32),
    )
    restored = export.deserialize(
        export.export(
            jax.jit(force),
            disabled_checks=(
                export.DisabledSafetyCheck.custom_call(FACTORIZED_FORWARD),
                export.DisabledSafetyCheck.custom_call(FACTORIZED_SPATIAL_BACKWARD),
            ),
        )(*specs).serialize()
    )

    for node_count, edge_count in ((5, 9), (8, 17)):
        keys = jax.random.split(jax.random.key(node_count + edge_count), 4)
        rows = jnp.arange(edge_count, dtype=jnp.int32) % node_count
        cols = (jnp.arange(edge_count, dtype=jnp.int32) * 3 + 1) % node_count
        args = (
            jax.random.normal(
                keys[0],
                (node_count, case.generated.config.irreps_in1.dim),
                dtype=jnp.float32,
            ),
            jax.random.normal(
                keys[1],
                (edge_count, case.generated.config.irreps_in2.dim),
                dtype=jnp.float32,
            ),
            jax.random.normal(keys[2], (edge_count, radial_dim), dtype=jnp.float32),
            jax.random.normal(
                keys[3],
                (radial_dim, case.generated.config.weight_numel),
                dtype=jnp.float32,
            ),
            rows,
            cols,
        )
        restored_force = restored.call(*args)
        reference_force = force(*args)
        for actual, expected in zip(
            restored_force, reference_force, strict=True
        ):
            np.testing.assert_allclose(
                actual, expected, **_tolerance(np.float32, 2)
            )
