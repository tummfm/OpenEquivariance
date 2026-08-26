"""Factorized radial-weight interface for JAX tensor-product convolutions."""

import jax
import jax.numpy as jnp

from openequivariance.core.e3nn_lite import TPProblem
from openequivariance.core.FactorizedConvPlan import factorized_plan_from_problem


def _sort_factorized_topology(rows, cols, num_nodes):
    """Sort graph topology for the generated receiver-row kernel.

    Invalid endpoints are sorted after valid edges and represented with receiver
    ``num_nodes``. Generated forward and adjoint kernels consistently skip them.
    """
    if rows.ndim != 1 or cols.ndim != 1 or rows.shape != cols.shape:
        raise ValueError("rows and cols must be equal-length vectors")
    if rows.dtype != jnp.int32 or cols.dtype != jnp.int32:
        raise ValueError("generated convolution topology must use int32 indices")
    edge_is_valid = (
        (rows >= 0) & (rows < num_nodes) & (cols >= 0) & (cols < num_nodes)
    )
    # Invalid edges sort after every receiver row.
    receiver_or_sentinel = jnp.where(edge_is_valid, rows, num_nodes)
    edge_order = jnp.argsort(receiver_or_sentinel, stable=True)
    edges_per_receiver = jnp.bincount(
        jnp.where(edge_is_valid, rows, 0),
        weights=edge_is_valid.astype(rows.dtype),
        length=num_nodes,
    ).astype(rows.dtype)
    row_offsets = jnp.concatenate(
        (
            jnp.zeros((1,), dtype=rows.dtype),
            jnp.cumsum(edges_per_receiver, dtype=rows.dtype),
        )
    )
    return (
        edge_order,
        receiver_or_sentinel[edge_order],
        cols[edge_order],
        row_offsets,
    )


class FactorizedTensorProductConv:
    """Tensor-product convolution with a differentiable radial projection.

    ``radial @ projection`` is evaluated by JAX at highest precision and the
    generated kernel consumes the resulting per-edge weights. Every floating
    operand supports the registered reverse and higher-order transforms.

    ``projection`` columns use OpenEquivariance's internal weight order.

    :param config: External, unshared ``uvu`` tensor-product problem in
                   ``mul_ir`` layout.
    :type config: TPProblem
    """

    def __init__(self, config: TPProblem):
        if config.shared_weights:
            raise ValueError(
                "FactorizedTensorProductConv requires unshared per-edge weights"
            )
        if config.internal_weights:
            raise ValueError("FactorizedTensorProductConv requires external weights")
        self.config = config
        self.weight_numel = config.weight_numel
        self.plan = factorized_plan_from_problem(config)

    def forward(
        self,
        X: jax.Array,
        Y: jax.Array,
        radial: jax.Array,
        projection: jax.Array,
        rows: jax.Array,
        cols: jax.Array,
    ) -> jax.Array:
        """Apply the projected tensor-product convolution.

        ``radial @ projection`` forms per-edge external weights with highest
        JAX matmul precision.  Valid edges are then sorted into contiguous
        receiver rows for the generated receiver-owned kernel.  Negative or
        out-of-range padded endpoints are explicitly masked and contribute
        zero to both values and floating derivatives.

        :param X: Node features ``[nodes, plan.input_dim]``.
        :param Y: Edge features ``[edges, plan.edge_dim]``.
        :param radial: Radial basis values ``[edges, radial_dim]``.
        :param projection: Learnable radial projection
            ``[radial_dim, plan.weight_numel]``.
        :param rows: int32 receiver indices ``[edges]``.
        :param cols: int32 sender indices ``[edges]``.
        :return: Output node features ``[nodes, plan.output_dim]``.
        :raises ValueError: If radial/projection ranks or compatible dimensions
            do not match, or topology indices are not int32.
        """

        radial = jnp.asarray(radial)
        projection = jnp.asarray(projection)
        if radial.ndim != 2 or projection.ndim != 2:
            raise ValueError("radial and projection must both be rank-2 arrays")
        if radial.shape[0] != Y.shape[0]:
            raise ValueError("radial and edge features must have the same edge count")
        if radial.shape[1] != projection.shape[0]:
            raise ValueError("radial and projection contraction dimensions disagree")
        if projection.shape[1] != self.weight_numel:
            raise ValueError(
                f"projection must have shape [radial_dim, {self.weight_numel}]"
            )
        if radial.dtype != projection.dtype:
            raise ValueError("radial and projection must have matching dtypes")
        if rows.dtype != jnp.int32 or cols.dtype != jnp.int32:
            raise ValueError("generated convolution topology must use int32 indices")
        if Y.shape[0] == 0:
            # Return the zero result for an empty edge set.
            return jnp.zeros((X.shape[0], self.plan.output_dim), dtype=X.dtype)

        from openequivariance.jax.jvp.factorized_projected_prim import (
            factorized_projected,
        )

        # Mask invalid padded edges before sorting so their values and
        # floating-point gradients are zero.
        valid_edge = (
            (rows >= 0) & (rows < X.shape[0]) & (cols >= 0) & (cols < X.shape[0])
        )
        safe_y = jnp.where(valid_edge[:, None], Y, jnp.zeros_like(Y))
        safe_radial = jnp.where(valid_edge[:, None], radial, jnp.zeros_like(radial))
        order, sorted_rows, sorted_cols, row_ptr = _sort_factorized_topology(
            rows, cols, X.shape[0]
        )

        # Apply the receiver permutation before projecting the radial basis.
        sorted_y = safe_y[order]
        sorted_radial = safe_radial[order]
        weights = jnp.matmul(
            sorted_radial, projection, precision=jax.lax.Precision.HIGHEST
        )
        return factorized_projected(
            self.plan,
            X,
            sorted_y,
            weights,
            sorted_cols,
            sorted_rows,
            row_ptr,
        )

    __call__ = forward


__all__ = ["FactorizedTensorProductConv"]
