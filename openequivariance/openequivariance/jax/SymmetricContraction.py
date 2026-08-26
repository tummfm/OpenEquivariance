"""Generic full-AD JAX symmetric contractions."""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from openequivariance.core.e3nn_lite import Irreps
from openequivariance.core.SymmetricContractionPlan import (
    SymmetricContractionPlan,
    build_symmetric_contraction_plan,
)
from openequivariance.jax.symmetric_literal import (
    _validate,
    evaluate_symmetric_plan,
    pack_symmetric_weights,
)


class SymmetricContraction:
    """Plan-driven symmetric polynomial with full JAX autodiff.

    Integer species IDs are runtime operands. Dense floating node attributes
    are also accepted and remain differentiable, including soft mixtures.
    Weight-table trailing dimensions are static properties of the operator.
    the checkpoint vocabulary dimension may remain symbolic through export.

    :param irreps_in: Input irreducible representations.
    :param irreps_out: Output irreducible representations.
    :param correlation: Polynomial degree, globally or per output irrep.
    :param irrep_normalization: Clebsch--Gordan normalisation convention.
    :param path_normalization: Path normalisation. The supported value is
        ``"element"``.
    :param num_elements: Number of rows in the species weight table.
    :param dtype: Floating dtype, either float32 or float64.
    :param input_layout: External feature layout, ``"mul_ir"`` or ``"ir_mul"``.
    :param algorithm: ``"reference"`` for ordinary JAX or ``"generated"`` for
                      eligible generated kernels with an ordinary-JAX fallback.
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        correlation: int | Mapping,
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        num_elements: int | None = None,
        dtype=jnp.float32,
        input_layout: str = "mul_ir",
        algorithm: str = "reference",
    ):
        if path_normalization != "element":
            raise ValueError("only path_normalization='element' is supported")
        if input_layout not in ("mul_ir", "ir_mul"):
            raise ValueError("input_layout must be 'mul_ir' or 'ir_mul'")
        if algorithm not in ("reference", "generated"):
            raise ValueError("algorithm must be 'reference' or 'generated'")
        if num_elements is None:
            raise ValueError("num_elements is required")
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.dtype = jnp.dtype(dtype)
        if self.dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
            raise TypeError("symmetric contraction supports float32 and float64")
        self.input_layout = input_layout
        self.algorithm = algorithm
        self.path_normalization = path_normalization
        self.plan = build_symmetric_contraction_plan(
            self.irreps_in,
            self.irreps_out,
            correlation,
            num_elements=num_elements,
            irrep_normalization=irrep_normalization,
            dtype=np.dtype(self.dtype),
        )
        self.num_elements = self.plan.num_elements
        self.num_features = self.plan.channels
        self.feature_dim = self.plan.feature_dim
        self.weight_shapes = self.plan.weight_shapes
        self._external_plan = False

    @classmethod
    def from_plan(
        cls,
        plan: SymmetricContractionPlan,
        *,
        dtype=jnp.float32,
        algorithm: str = "reference",
    ):
        """Construct an evaluator for an externally canonicalised static plan.

        The call accepts canonical rank-three features and an already packed
        ``[S, weight_dim]`` table. This interface preserves external checkpoint
        layouts and keeps descriptor dependencies out of runtime evaluation and
        generated code.

        :param plan: Canonical sparse polynomial and packed-buffer layout.
        :param dtype: Floating dtype, either float32 or float64.
        :param algorithm: ``"reference"`` or ``"generated"``.
        :return: Symmetric contraction configured for canonical inputs.
        """
        if not isinstance(plan, SymmetricContractionPlan):
            raise TypeError("plan must be a SymmetricContractionPlan")
        if algorithm not in ("reference", "generated"):
            raise ValueError("algorithm must be 'reference' or 'generated'")
        instance = cls.__new__(cls)
        instance.irreps_in = None
        instance.irreps_out = None
        instance.dtype = jnp.dtype(dtype)
        if instance.dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
            raise TypeError("symmetric contraction supports float32 and float64")
        instance.input_layout = "mul_ir"
        instance.algorithm = algorithm
        instance.path_normalization = "external"
        instance.plan = plan
        instance.num_elements = plan.num_elements
        instance.num_features = plan.channels
        instance.feature_dim = plan.feature_dim
        instance.weight_shapes = plan.weight_shapes
        instance._external_plan = True
        return instance

    def init_weights(self, key, scale: float = 1.0):
        """Initialise external symmetric-contraction weight tables.

        :param key: JAX random key used to sample every weight array.
        :param scale: Multiplicative scale applied after channel normalisation.
        :return: Nested ``[species, parameters, channels]`` arrays matching
                 :attr:`weight_shapes`.
        """
        keys = iter(
            jax.random.split(key, sum(len(group) for group in self.weight_shapes))
        )
        return tuple(
            tuple(
                scale
                * jax.random.normal(next(keys), shape, dtype=self.dtype)
                / max(shape[1], 1)
                for shape in group
            )
            for group in self.weight_shapes
        )

    def _canonical_features(self, features):
        features = jnp.asarray(features)
        if features.dtype != self.dtype:
            raise TypeError(
                f"expected feature dtype {self.dtype}, got {features.dtype}"
            )
        if features.ndim == 3:
            expected = (
                (self.plan.feature_dim, self.plan.channels)
                if self.plan.feature_layout == "feature_channel"
                else (self.plan.channels, self.plan.feature_dim)
            )
            if features.shape[1:] != expected:
                raise ValueError("rank-3 feature shape does not match irreps_in")
            return features
        if self._external_plan:
            raise ValueError("external symmetric plans require rank-3 features")
        if features.ndim != 2 or features.shape[1] != self.irreps_in.dim:
            raise ValueError("features must be rank 2 or canonical rank 3")
        blocks = []
        for mul_ir, sl in zip(self.irreps_in, self.irreps_in.slices(), strict=True):
            block = features[:, sl]
            if self.input_layout == "mul_ir":
                block = block.reshape(
                    features.shape[0], self.plan.channels, mul_ir.ir.dim
                )
            else:
                block = block.reshape(
                    features.shape[0], mul_ir.ir.dim, self.plan.channels
                ).swapaxes(1, 2)
            blocks.append(block)
        return jnp.concatenate(blocks, axis=2)

    def __call__(self, features, species_or_node_attrs, weights):
        """Evaluate features using species IDs or dense node attributes.

        Normal instances accept weights in the nested external layout returned
        by :meth:`init_weights`.  Instances created with :meth:`from_plan`
        accept an already packed ``[species, weight_dim]`` table.

        :param features: Node features in external or canonical layout.
        :param species_or_node_attrs: Species IDs ``[N]`` or attributes
                                      ``[N, species]``.
        :param weights: Nested or packed symmetric-contraction weights.
        :return: Contracted node features ``[N, output_dim]``.
        """
        features = self._canonical_features(features)
        if self._external_plan:
            packed_weights = jnp.asarray(weights)
            if (
                packed_weights.ndim != 2
                or packed_weights.shape[1] != self.plan.weight_dim
            ):
                raise ValueError(
                    f"packed weights must have shape [S,{self.plan.weight_dim}]"
                )
        else:
            packed_weights = pack_symmetric_weights(weights, self.plan)
        selector = jnp.asarray(species_or_node_attrs)
        if selector.ndim == 1:
            if selector.dtype != jnp.int32:
                raise TypeError("symmetric species must use int32")
            attributes = False
        elif selector.ndim == 2:
            if not jnp.issubdtype(selector.dtype, jnp.floating):
                raise TypeError("symmetric node attributes must be floating point")
            attributes = True
        else:
            raise ValueError("selector must be species [N] or node attributes [N,Z]")
        _validate(
            self.plan,
            features,
            selector,
            packed_weights,
            attributes=attributes,
        )
        if self.algorithm == "generated":
            if features.shape[0] == 0:
                return jnp.zeros((0, self.plan.output_dim), dtype=features.dtype)
            from openequivariance.jax.symmetric_literal import (
                symmetric_literal,
            )

            return symmetric_literal(self.plan, features, selector, packed_weights)
        return evaluate_symmetric_plan(self.plan, features, selector, packed_weights)


__all__ = [
    "SymmetricContraction",
    "evaluate_symmetric_plan",
    "pack_symmetric_weights",
]
