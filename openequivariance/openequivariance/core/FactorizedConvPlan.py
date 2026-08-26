"""Static, framework-neutral plans for generated factorized convolutions."""

from dataclasses import dataclass, field
from math import prod

import numpy as np

from openequivariance.core.e3nn_lite import TPProblem, wigner_3j


@dataclass(frozen=True, slots=True, eq=False)
class FactorizedConvPath:
    """Sparse coupling data and flat offsets for one weighted instruction.

    The four offsets address the external flattened input, edge, output, and
    per-edge weight arrays. The ``cg_*`` arrays contain aligned nonzero real
    Clebsch--Gordan component indices and values.
    """

    input_start: int
    edge_start: int
    output_start: int
    weight_start: int
    input_irrep_dim: int
    output_irrep_dim: int
    cg_input: np.ndarray = field(repr=False, compare=False)
    cg_edge: np.ndarray = field(repr=False, compare=False)
    cg_output: np.ndarray = field(repr=False, compare=False)
    cg_value: np.ndarray = field(repr=False, compare=False)

    def sparse_output(
        self, output_component: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return sparse CG entries contributing to one output component."""
        selected = self.cg_output == output_component
        return (
            self.cg_input[selected],
            self.cg_edge[selected],
            self.cg_value[selected],
        )


@dataclass(frozen=True, slots=True, eq=False)
class FactorizedConvPlan:
    """Static sparse schedule for a generated receiver-owned convolution.

    ``paths`` preserve the flat external feature and weight layout of the
    source :class:`~openequivariance.core.e3nn_lite.TPProblem`.  The generated
    kernels use receiver-sorted topology at runtime. Graph size and edge count
    are intentionally absent from this static plan.
    """

    paths: tuple[FactorizedConvPath, ...]
    input_dim: int
    edge_dim: int
    output_dim: int
    weight_numel: int
    channels: int
    layout: str = "mul_ir"


def factorized_plan_from_problem(problem: TPProblem) -> FactorizedConvPlan:
    """Lower a supported tensor-product problem to sparse generated paths.

    Each weighted ``uvu`` instruction becomes one path containing the nonzero
    real Clebsch--Gordan entries and its original flat weight offset.  The
    resulting plan covers the same complete external weight vector as
    ``problem``. It does not create learnable values or select graph-dependent
    launch dimensions.

    :param problem: External-weight, unshared ``TPProblem`` in ``"mul_ir"``
        layout with weighted ``"uvu"`` instructions and edge multiplicity one.
    :return: Sparse plan preserving the problem's flattened input, edge,
        output, and weight layouts.
    :raises ValueError: If the layout, instruction modes, channel structure, or
        flat-weight coverage is unsupported.
    """

    # Validate the weight ownership and feature layout used by the kernel.
    if problem.layout != "mul_ir":
        raise ValueError("generated factorized convolution requires mul_ir layout")
    if problem.shared_weights or problem.internal_weights:
        raise ValueError(
            "generated factorized convolution requires external unshared weights"
        )

    # Resolve each irrep block to its offset in the flattened feature arrays.
    input_slices = problem.irreps_in1.slices()
    edge_slices = problem.irreps_in2.slices()
    output_slices = problem.irreps_out.slices()
    paths = []
    referenced_outputs = set()
    weight_start = 0
    channel_count = None

    # Lower weighted uvu instructions to sparse coupling paths.
    for index, instruction in enumerate(problem.instructions):
        if not instruction.has_weight:
            raise ValueError("generated factorized convolution requires weighted paths")
        if instruction.connection_mode != "uvu":
            raise ValueError(
                "generated factorized convolution supports uvu paths only"
            )
        input_mul_ir = problem.irreps_in1[instruction.i_in1]
        edge_mul_ir = problem.irreps_in2[instruction.i_in2]
        output_mul_ir = problem.irreps_out[instruction.i_out]
        referenced_outputs.add(instruction.i_out)
        if edge_mul_ir.mul != 1:
            raise ValueError(
                "generated factorized convolution requires edge multiplicity one"
            )
        if input_mul_ir.mul != output_mul_ir.mul:
            raise ValueError("uvu input and output channel multiplicities must match")
        if channel_count is None:
            channel_count = input_mul_ir.mul
        elif input_mul_ir.mul != channel_count:
            raise ValueError(
                "generated factorized convolution requires uniform channels"
            )

        # Prune zero entries from the Clebsch–Gordan tensor. Forbidden
        # component couplings do not need source expressions.
        cg = np.asarray(
            wigner_3j(
                input_mul_ir.ir.l,
                edge_mul_ir.ir.l,
                output_mul_ir.ir.l,
            ),
            dtype=np.float64,
        ) * float(instruction.path_weight)
        nonzero = np.nonzero(cg)
        values = np.ascontiguousarray(cg[nonzero], dtype=np.float64)
        if values.size == 0:
            raise ValueError(f"instruction {index} has an empty coupling tensor")
        paths.append(
            FactorizedConvPath(
                input_start=input_slices[instruction.i_in1].start,
                edge_start=edge_slices[instruction.i_in2].start,
                output_start=output_slices[instruction.i_out].start,
                weight_start=weight_start,
                input_irrep_dim=input_mul_ir.ir.dim,
                output_irrep_dim=output_mul_ir.ir.dim,
                cg_input=np.ascontiguousarray(nonzero[0], dtype=np.int32),
                cg_edge=np.ascontiguousarray(nonzero[1], dtype=np.int32),
                cg_output=np.ascontiguousarray(nonzero[2], dtype=np.int32),
                cg_value=values,
            )
        )
        weight_start += prod(instruction.path_shape)

    # Verify that the paths cover the original flat weight vector exactly.
    if not paths:
        raise ValueError("generated factorized convolution requires at least one path")
    missing_outputs = [
        output
        for output, mul_ir in enumerate(problem.irreps_out)
        if mul_ir.mul * mul_ir.ir.dim > 0 and output not in referenced_outputs
    ]
    if missing_outputs:
        raise ValueError(
            "factorized plan does not reference positive-dimensional output "
            f"irrep blocks {missing_outputs}"
        )
    if weight_start != problem.weight_numel:
        raise ValueError("factorized plan does not cover the complete weight vector")
    assert channel_count is not None
    return FactorizedConvPlan(
        paths=tuple(paths),
        input_dim=problem.irreps_in1.dim,
        edge_dim=problem.irreps_in2.dim,
        output_dim=problem.irreps_out.dim,
        weight_numel=problem.weight_numel,
        channels=channel_count,
    )


__all__ = [
    "FactorizedConvPath",
    "FactorizedConvPlan",
    "factorized_plan_from_problem",
]
