"""Generated symmetric-contraction numerical and lowering coverage."""

from types import SimpleNamespace

import numpy as np
import pytest


_TOLERANCES = {
    np.dtype(np.float32): {"atol": 8e-4, "rtol": 8e-4},
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


def _plan(feature_layout="channel_feature"):
    from openequivariance.core.SymmetricContractionPlan import (
        SymmetricContractionPlan,
        SymmetricPath,
    )

    return SymmetricContractionPlan(
        channels=2,
        feature_dim=4,
        output_dim=4,
        num_elements=4,
        weight_dim=8,
        weight_shapes=(),
        paths=(
            SymmetricPath(0, 1, 0, 0, 2, 0, (0,), 1.5),
            SymmetricPath(0, 1, 0, 0, 2, 1, (1, 1), -0.25),
            SymmetricPath(2, 1, 0, 4, 2, 0, (0, 2, 3), 0.5),
            SymmetricPath(2, 1, 0, 4, 2, 1, (3,), -0.75),
        ),
        feature_layout=feature_layout,
    )


def _operator_pair(dtype, *, feature_layout="channel_feature"):
    """Build a generated operator and an explicit polynomial oracle."""
    import jax.numpy as jnp

    from openequivariance.jax.SymmetricContraction import SymmetricContraction

    generated = SymmetricContraction.from_plan(
        _plan(feature_layout), dtype=dtype, algorithm="generated"
    )

    def reference(features, selector, weights):
        channel_features = (
            features
            if feature_layout == "channel_feature"
            else jnp.swapaxes(features, 1, 2)
        )
        if selector.ndim == 1:
            valid = (selector >= 0) & (selector < weights.shape[0])
            selected = weights[jnp.where(valid, selector, 0)]
            selected = jnp.where(valid[:, None], selected, 0)
        else:
            selected = selector @ weights
        first = 1.5 * selected[:, (0, 2)] * channel_features[:, :, 0]
        first -= 0.25 * selected[:, (1, 3)] * channel_features[:, :, 1] ** 2
        second = 0.5 * selected[:, (4, 6)] * channel_features[:, :, 0]
        second *= channel_features[:, :, 2] * channel_features[:, :, 3]
        second -= 0.75 * selected[:, (5, 7)] * channel_features[:, :, 3]
        return jnp.concatenate((first, second), axis=1)

    return generated, reference


def _case(gpu_context, dtype):
    """Create one f32/f64 operator pair with distinct primal and tangent keys."""
    jax, jnp = gpu_context
    generated, reference = _operator_pair(dtype)
    keys = jax.random.split(jax.random.key(412), 6)
    features = jax.random.normal(keys[0], (6, 2, 4), dtype=dtype)
    species = jnp.arange(6, dtype=jnp.int32) % 4
    attributes = jax.nn.one_hot(species, 4, dtype=dtype)
    weights = jax.random.normal(keys[1], (4, 8), dtype=dtype)
    return SimpleNamespace(
        generated=generated,
        reference=reference,
        features=features,
        species=species,
        attributes=attributes,
        weights=weights,
        feature_tangent=jax.random.normal(keys[2], features.shape, dtype=dtype),
        attribute_tangent=jax.random.normal(keys[3], attributes.shape, dtype=dtype),
        weight_tangent=jax.random.normal(keys[4], weights.shape, dtype=dtype),
        cotangent=jax.random.normal(keys[5], (6, 4), dtype=dtype),
    )


def _tolerance(dtype, multiplier=1):
    return {
        name: value * multiplier
        for name, value in _TOLERANCES[np.dtype(dtype)].items()
    }


def _assert_tree_close(actual, expected, tolerance):
    actual_leaves = actual if isinstance(actual, tuple) else (actual,)
    expected_leaves = expected if isinstance(expected, tuple) else (expected,)
    for got, want in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_allclose(got, want, **tolerance)


def test_generated_species_invalid_ids_are_zero(gpu_context):
    """Map negative and upper-bound species IDs to zero rows."""
    _, jnp = gpu_context
    generated, reference = _operator_pair(np.float32)
    features = jnp.ones((4, 2, 4), jnp.float32)
    weights = jnp.arange(32, dtype=jnp.float32).reshape(4, 8) / 7
    species = jnp.asarray((-1, 0, 3, 4), jnp.int32)

    actual = generated(features, species, weights)
    np.testing.assert_allclose(
        actual, reference(features, species, weights), **_tolerance(np.float32)
    )
    np.testing.assert_allclose(
        np.asarray(actual)[(0, 3)], 0, **_tolerance(np.float32)
    )


@pytest.mark.parametrize("feature_layout", ("channel_feature", "feature_channel"))
def test_generated_feature_layouts_match_their_declared_ffi_abi(
    gpu_context, feature_layout
):
    """Use each declared [N,C,F]/[N,F,C] layout without reinterpretation."""
    jax, jnp = gpu_context
    generated, reference = _operator_pair(
        np.float32, feature_layout=feature_layout
    )
    channel_features = jax.random.normal(
        jax.random.key(7), (3, 2, 4), dtype=jnp.float32
    )
    features = (
        channel_features
        if feature_layout == "channel_feature"
        else jnp.swapaxes(channel_features, 1, 2)
    )
    species = jnp.asarray((0, 1, 0), dtype=jnp.int32)
    weights = jax.random.normal(jax.random.key(8), (2, 8), dtype=jnp.float32)

    np.testing.assert_allclose(
        jax.jit(generated)(features, species, weights),
        reference(features, species, weights),
        **_tolerance(np.float32),
    )


def test_generated_ffi_rejects_shape_for_wrong_declared_feature_layout(gpu_context):
    """Native validation rejects [N,C,F] when the ABI claims [N,F,C]."""
    _, jnp = gpu_context
    import jax

    from openequivariance.jax.ffi_targets import (
        SYMMETRIC_FORWARD,
        SYMMETRIC_LAYOUT_FEATURE_CHANNEL,
    )
    from openequivariance.jax.symmetric_literal_codegen import (
        generate_symmetric_literal_source,
    )

    plan = _plan()
    generated = generate_symmetric_literal_source(plan, dtype=np.float32)
    x = jnp.ones((2, 2, 4), dtype=jnp.float32)
    species = jnp.zeros((2,), dtype=jnp.int32)
    weights = jnp.ones((1, 8), dtype=jnp.float32)
    call = jax.ffi.ffi_call(
        SYMMETRIC_FORWARD, jax.ShapeDtypeStruct((2, plan.output_dim), x.dtype)
    )

    with pytest.raises(
        jax.errors.JaxRuntimeError, match="symmetric x"
    ):
        call(
            x,
            species,
            weights,
            source=generated.source,
            hash=generated.source_hash,
            channels=plan.channels,
            feature_dim=plan.feature_dim,
            output_dim=plan.output_dim,
            weight_dim=plan.weight_dim,
            feature_layout=SYMMETRIC_LAYOUT_FEATURE_CHANNEL,
        ).block_until_ready()


@pytest.mark.parametrize("dtype", (np.float32, np.float64), ids=("f32", "f64"))
@pytest.mark.parametrize("selector_name", ("species", "attributes"))
def test_generated_forward_and_ad_match_explicit_polynomial(
    gpu_context, dtype, selector_name
):
    """Match f32/f64 species and dense-attribute forward and AD transforms."""
    jax, jnp = gpu_context
    case = _case(gpu_context, dtype)
    selector = getattr(case, selector_name)
    values = (case.features, selector, case.weights)
    active = (0, 2) if selector_name == "species" else (0, 1, 2)
    primals = tuple(values[index] for index in active)
    tangents = (case.feature_tangent, case.weight_tangent)
    if selector_name == "attributes":
        tangents = (case.feature_tangent, case.attribute_tangent, case.weight_tangent)

    def bind(operator, *active_values):
        bound = list(values)
        for index, value in zip(active, active_values, strict=True):
            bound[index] = value
        return operator(*bound)

    def generated(*args):
        return bind(case.generated, *args)

    def reference(*args):
        return bind(case.reference, *args)
    np.testing.assert_allclose(
        jax.jit(generated)(*primals),
        jax.jit(reference)(*primals),
        **_tolerance(dtype),
    )
    np.testing.assert_allclose(
        jax.jvp(generated, primals, tangents)[1],
        jax.jvp(reference, primals, tangents)[1],
        **_tolerance(dtype, 5),
    )
    _assert_tree_close(
        jax.vjp(generated, *primals)[1](case.cotangent),
        jax.vjp(reference, *primals)[1](case.cotangent),
        _tolerance(dtype, 6),
    )

    def energy(operator, *args):
        return jnp.vdot(operator(*args), case.cotangent)

    def gradient(operator):
        return jax.grad(
            lambda *args: energy(operator, *args), tuple(range(len(primals)))
        )
    _assert_tree_close(
        jax.jvp(gradient(generated), primals, tangents)[1],
        jax.jvp(gradient(reference), primals, tangents)[1],
        _tolerance(dtype, 12),
    )
    _assert_tree_close(
        jax.grad(
            lambda *args: jnp.vdot(
                jax.jvp(generated, args, tangents)[1], case.cotangent
            ),
            tuple(range(len(primals))),
        )(*primals),
        jax.grad(
            lambda *args: jnp.vdot(
                jax.jvp(reference, args, tangents)[1], case.cotangent
            ),
            tuple(range(len(primals))),
        )(*primals),
        _tolerance(dtype, 12),
    )


def test_generated_hlo_has_representative_forward_adjoint_and_hvp_targets(gpu_context):
    """Lower representative f32 forward, adjoint, and HVP targets."""
    jax, jnp = gpu_context
    from openequivariance.jax.ffi_targets import (
        SYMMETRIC_BACKWARD,
        SYMMETRIC_BACKWARD_HVP,
        SYMMETRIC_FORWARD,
    )

    case = _case(gpu_context, np.float32)

    def energy(features):
        return jnp.vdot(
            case.generated(features, case.species, case.weights), case.cotangent
        )

    def hlo(function, *args):
        return str(jax.jit(function).lower(*args).compiler_ir(dialect="stablehlo"))

    assert SYMMETRIC_FORWARD in hlo(
        case.generated, case.features, case.species, case.weights
    )
    force = jax.grad(energy)
    assert SYMMETRIC_BACKWARD in hlo(force, case.features)
    assert SYMMETRIC_BACKWARD_HVP in hlo(
        lambda features, tangent: jax.jvp(force, (features,), (tangent,))[1],
        case.features,
        case.feature_tangent,
    )


def _exported(gpu_context, *, dynamic_species):
    jax, jnp = gpu_context
    from jax import export
    from openequivariance.jax.ffi_targets import SYMMETRIC_TARGETS

    generated, reference = _operator_pair(np.float32)
    shape = "nodes,s" if dynamic_species else "nodes"
    constraints = ("nodes >= 1", "s >= 1") if dynamic_species else ("nodes >= 1",)
    symbols = export.symbolic_shape(shape, constraints=constraints)
    nodes = symbols[0]
    species_count = symbols[1] if dynamic_species else 4
    specs = (
        jax.ShapeDtypeStruct((nodes, 2, 4), jnp.float32),
        jax.ShapeDtypeStruct((nodes,), jnp.int32),
        jax.ShapeDtypeStruct((species_count, 8), jnp.float32),
    )
    restored = export.deserialize(
        export.export(
            jax.jit(generated),
            disabled_checks=tuple(
                export.DisabledSafetyCheck.custom_call(target)
                for target in SYMMETRIC_TARGETS
            ),
        )(*specs).serialize()
    )
    return restored, reference


def test_generated_export_has_dynamic_node_count(gpu_context):
    """Export generated f32 evaluation for two runtime node counts."""
    _, jnp = gpu_context
    restored, reference = _exported(gpu_context, dynamic_species=False)
    weights = jnp.arange(32, dtype=jnp.float32).reshape(4, 8) / 13
    for nodes in (2, 6):
        features = jnp.full((nodes, 2, 4), 0.125, jnp.float32)
        species = jnp.arange(nodes, dtype=jnp.int32) % 4
        np.testing.assert_allclose(
            restored.call(features, species, weights),
            reference(features, species, weights),
            **_tolerance(np.float32, 2),
        )


def test_generated_external_plan_export_has_dynamic_nodes_and_species(gpu_context):
    """Export an external plan with runtime node and species-table dimensions."""
    _, jnp = gpu_context
    restored, reference = _exported(gpu_context, dynamic_species=True)
    for nodes, species_count in ((3, 2), (7, 5)):
        features = (
            jnp.arange(nodes * 8, dtype=jnp.float32).reshape(nodes, 2, 4) / 11
        )
        species = jnp.arange(nodes, dtype=jnp.int32) % species_count
        weights = (
            jnp.arange(species_count * 8, dtype=jnp.float32).reshape(
                species_count, 8
            )
            / 17
        )
        np.testing.assert_allclose(
            restored.call(features, species, weights),
            reference(features, species, weights),
            **_tolerance(np.float32, 2),
        )
