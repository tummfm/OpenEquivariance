"""Reference evaluation and generated dispatch for symmetric contractions."""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from openequivariance.core.SymmetricContractionPlan import SymmetricContractionPlan
from openequivariance.jax.ffi_targets import symmetric_feature_layout_abi_value


def pack_symmetric_weights(
    weights: Sequence[Sequence[jax.Array]],
    plan: SymmetricContractionPlan,
) -> jax.Array:
    """Pack symmetric-contraction weights into the generated-kernel layout.

    ``weights`` has one outer group per output irrep and one array within that
    group per correlation order.  Every array has shape ``[S, P, C]``, where
    ``S`` is the number of species, ``P`` the number of coupling paths, and
    ``C`` the number of feature channels.  ``plan.weight_shapes`` defines the
    required group order and shapes.

    The returned array has shape ``[S, plan.weight_dim]``.  Within each species
    row, arrays are concatenated in plan order.  Each array is stored with the
    channel index before the path index, so its local flat index is ``c * P +
    p``. Packing only transposes and reshapes the supplied values. It does not
    change their dtype or numerical values.

    :param weights: Nested symmetric weights in external ``[S, P, C]`` layout.
    :param plan: Static contraction plan which defines the expected layout.
    :return: Weight table in the flat layout consumed by generated kernels.
    """
    if len(weights) != len(plan.weight_shapes):
        raise ValueError(
            f"expected {len(plan.weight_shapes)} output weight groups, "
            f"got {len(weights)}"
        )
    packed = []
    for actual_group, expected_group in zip(weights, plan.weight_shapes, strict=True):
        if len(actual_group) != len(expected_group):
            raise ValueError(
                f"expected {len(expected_group)} correlation weights, "
                f"got {len(actual_group)}"
            )
        for value, shape in zip(actual_group, expected_group, strict=True):
            value = jnp.asarray(value)
            if value.shape != shape:
                raise ValueError(
                    f"expected symmetric weight {shape}, got {value.shape}"
                )
            packed.append(jnp.swapaxes(value, 1, 2).reshape(plan.num_elements, -1))
    if not packed:
        raise ValueError("symmetric contraction has no weights")
    dtype = packed[0].dtype
    if any(value.dtype != dtype for value in packed):
        raise TypeError("all symmetric weights must have the same dtype")
    return jnp.concatenate(packed, axis=1)


def evaluate_symmetric_plan(
    plan: SymmetricContractionPlan,
    features: jax.Array,
    species_or_node_attrs: jax.Array,
    packed_weights: jax.Array,
) -> jax.Array:
    """Evaluate a symmetric plan using ordinary exportable JAX operations.

    ``features`` contains one row per node.  ``species_or_node_attrs`` either
    selects one packed weight row with an integer species ID or forms a soft
    mixture of rows with dense node attributes.

    :param plan: Static sparse polynomial and buffer-layout description.
    :param features: Canonical node features in the layout named by ``plan``.
    :param species_or_node_attrs: Species IDs ``[N]`` or attributes ``[N, S]``.
    :param packed_weights: Flat per-species table ``[S, plan.weight_dim]``.
    :return: Contracted node features ``[N, plan.output_dim]``.
    """
    features = jnp.asarray(features)
    selector = jnp.asarray(species_or_node_attrs)
    packed_weights = jnp.asarray(packed_weights)
    trailing_shape = (
        (plan.feature_dim, plan.channels)
        if plan.feature_layout == "feature_channel"
        else (plan.channels, plan.feature_dim)
    )
    if features.ndim != 3 or features.shape[1:] != trailing_shape:
        raise ValueError(
            f"features must have shape [nodes,{trailing_shape[0]},{trailing_shape[1]}]"
        )
    if packed_weights.ndim != 2 or packed_weights.shape[1] != plan.weight_dim:
        raise ValueError(f"packed weights must have shape [S,{plan.weight_dim}]")
    if selector.ndim == 1:
        if selector.dtype != jnp.int32:
            raise TypeError("species must use int32")
        if selector.shape[0] != features.shape[0]:
            raise ValueError("species must contain one entry per node")
        valid_species = (selector >= 0) & (selector < packed_weights.shape[0])
        safe_species = jnp.where(valid_species, selector, 0)
        selected_weights = packed_weights[safe_species]
        selected_weights = jnp.where(valid_species[:, None], selected_weights, 0)
    elif selector.ndim == 2 and selector.shape[1] == packed_weights.shape[0]:
        if selector.shape[0] != features.shape[0]:
            raise ValueError("node attributes must contain one row per node")
        if selector.dtype != features.dtype:
            raise TypeError("node attributes and features must have the same dtype")
        selected_weights = jnp.matmul(
            selector,
            packed_weights,
            precision=jax.lax.Precision.HIGHEST,
        )
    else:
        raise ValueError(
            "selector must be integer species [nodes] or dense node "
            "attributes [nodes,S] matching weights"
        )
    if features.dtype != packed_weights.dtype:
        raise TypeError("features and symmetric weights must have the same dtype")

    result = jnp.zeros((features.shape[0], plan.output_dim), features.dtype)
    channels = jnp.arange(plan.channels)
    for path in plan.paths:
        weight_index = (
            path.weight_base
            + channels * path.couplings_per_channel
            + path.coupling_index
        )
        value = path.coefficient * selected_weights[:, weight_index]
        for component in path.feature_components:
            value = value * (
                features[:, component, :]
                if plan.feature_layout == "feature_channel"
                else features[:, :, component]
            )
        output_index = (
            path.output_base + channels * path.output_irrep_dim + path.output_component
        )
        result = result.at[:, output_index].add(value)
    return result


def _materialize_symbolic_zero(value):
    """Materialize a JAX symbolic zero using its primal abstract value."""
    if ad.is_undefined_primal(value) or type(value) is ad.Zero:
        return jnp.zeros(value.aval.shape, value.aval.dtype)
    return value


def _plan_attributes(plan):
    """Return static dimensions and exact feature layout for a FFI handler."""
    return {
        "channels": plan.channels,
        "feature_dim": plan.feature_dim,
        "output_dim": plan.output_dim,
        "weight_dim": plan.weight_dim,
        "feature_layout": symmetric_feature_layout_abi_value(plan.feature_layout),
    }


def _validate(plan, x, selector, weights, *, attributes, dout=None):
    trailing_shape = (
        (plan.feature_dim, plan.channels)
        if plan.feature_layout == "feature_channel"
        else (plan.channels, plan.feature_dim)
    )
    if x.ndim != 3 or x.shape[1:] != trailing_shape:
        raise ValueError(
            f"x must have shape [N, {trailing_shape[0]}, {trailing_shape[1]}]"
        )
    if weights.ndim != 2 or weights.shape[1] != plan.weight_dim:
        raise ValueError(f"weights must have shape [S, {plan.weight_dim}]")
    if x.dtype != weights.dtype:
        raise ValueError("x and weights must have the same dtype")
    if attributes:
        if selector.shape != (x.shape[0], weights.shape[0]):
            raise ValueError("attributes must have shape [N, S] matching weights")
        if selector.dtype != x.dtype:
            raise ValueError("attributes and x must have the same dtype")
    else:
        if selector.shape != (x.shape[0],):
            raise ValueError("species must contain one entry per node")
        if selector.dtype != np.dtype(np.int32):
            raise TypeError("species must use int32")
    if dout is not None and (
        dout.shape != (x.shape[0], plan.output_dim) or dout.dtype != x.dtype
    ):
        raise ValueError(f"dout must have shape [N, {plan.output_dim}]")


def _reference_backward(plan, x, selector, weights, dout, *, attributes):
    if attributes:
        _, pullback = jax.vjp(
            lambda a, b, c: evaluate_symmetric_plan(plan, a, b, c),
            x,
            selector,
            weights,
        )
    else:
        _, pullback = jax.vjp(
            lambda a, c: evaluate_symmetric_plan(plan, a, selector, c),
            x,
            weights,
        )
    return pullback(dout)


def symmetric_literal(plan, x, selector, weights):
    """Use the literal species kernel when eligible, otherwise ordinary JAX.

    Unsupported plans and dense attributes use the reference evaluator, which
    supports all ordinary JAX transformations.

    :param plan: Canonical sparse symmetric-contraction plan.
    :param x: Canonical features ``[nodes, channels, feature_dim]`` or the
        feature-channel layout named by ``plan``.
    :param selector: int32 species IDs ``[nodes]`` or dense attributes
        ``[nodes, species]``.
    :param weights: Packed per-species table ``[species, plan.weight_dim]``.
    :return: Contracted node features ``[nodes, plan.output_dim]``.
    """
    if selector.ndim == 1:
        from openequivariance.jax.symmetric_literal_codegen import (
            symmetric_literal_eligibility,
        )

        if symmetric_literal_eligibility(plan).eligible:
            from openequivariance.jax.jvp.symmetric_literal_prim import (
                symmetric_literal_generated,
            )

            return symmetric_literal_generated(plan, x, selector, weights)
    return evaluate_symmetric_plan(plan, x, selector, weights)


__all__ = [
    "evaluate_symmetric_plan",
    "pack_symmetric_weights",
    "symmetric_literal",
]
