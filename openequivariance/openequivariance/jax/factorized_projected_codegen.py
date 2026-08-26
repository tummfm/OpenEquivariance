"""Render the receiver-owned factorized convolution kernels."""

from dataclasses import dataclass

import numpy as np

from openequivariance.core.FactorizedConvPlan import FactorizedConvPlan
from openequivariance.core.utils import hash_str_64
from openequivariance.templates.jinja_utils import get_jinja_environment


@dataclass(frozen=True, slots=True)
class GeneratedProjectedSource:
    """Rendered factorized-kernel source and deterministic 64-bit cache hash.

    The source includes the static sparse plan and AD-activity specialisation.
    graph topology and feature-array leading dimensions remain runtime
    operands.
    """

    source: str
    source_hash: int


@dataclass(frozen=True, slots=True)
class _TangentContext:
    x: str
    sh: str
    weights: str
    output: str = "scalar_t(0)"


@dataclass(frozen=True, slots=True)
class _TermContext:
    input_index: int
    edge_index: int
    coefficient: str
    input_gradient_name: str


@dataclass(frozen=True, slots=True)
class _OutputContext:
    output_index: int
    output_irrep_dim: int
    input_irrep_dim: int
    value_name: str
    terms: tuple[_TermContext, ...]


@dataclass(frozen=True, slots=True)
class _PathContext:
    weight_start: int
    outputs: tuple[_OutputContext, ...]


@dataclass(frozen=True, slots=True)
class _OutputSlot:
    name: str
    output_index: int
    output_irrep_dim: int


@dataclass(frozen=True, slots=True)
class _InputGradientSlot:
    name: str
    input_index: int
    input_irrep_dim: int


def _literal(value: float, scalar: str) -> str:
    suffix = "f" if scalar == "float" else ""
    return f"static_cast<scalar_t>({float(value).hex()}{suffix})"


def _tangent_context(active, *, include_output=False):
    """Map an activity tuple to generated tangent-load expressions."""
    names = ("tx[(index)]", "tsh[(index)]", "tweights[(index)]")
    values = tuple(
        name if enabled else "scalar_t(0)" for name, enabled in zip(names, active[:3])
    )
    output = "(value)" if include_output and active[3] else "scalar_t(0)"
    return _TangentContext(*values, output=output)


def _render_path_context(path, scalar, output_slots, input_slots):
    """Lower one sparse plan path to names and expressions used by Jinja."""
    outputs = []
    for output_component in range(path.output_irrep_dim):
        input_components, edge_components, coefficients = path.sparse_output(
            output_component
        )

        # Lower nonzero Clebsch--Gordan entries and intern their input-gradient
        # accumulators across output components.
        terms = []
        for input_component, edge_component, coefficient in zip(
            input_components.tolist(),
            edge_components.tolist(),
            coefficients.tolist(),
        ):
            input_component = int(input_component)
            input_index = path.input_start + input_component
            edge_index = path.edge_start + int(edge_component)
            input_slot_key = (
                path.input_start,
                path.input_irrep_dim,
                input_component,
            )
            input_slot = input_slots.setdefault(
                input_slot_key,
                _InputGradientSlot(
                    name=f"input_gradient_{path.input_start}_{input_component}",
                    input_index=input_index,
                    input_irrep_dim=path.input_irrep_dim,
                ),
            )
            terms.append(
                _TermContext(
                    input_index=input_index,
                    edge_index=edge_index,
                    coefficient=_literal(coefficient, scalar),
                    input_gradient_name=input_slot.name,
                )
            )

        # Output accumulators are shared by every plan path which contributes
        # to the same flattened output component.
        output_index = path.output_start + output_component
        output_slot_key = (
            path.output_start,
            output_component,
            path.output_irrep_dim,
        )
        output_slot = output_slots.setdefault(
            output_slot_key,
            _OutputSlot(
                name=f"output_{path.output_start}_{output_component}",
                output_index=output_index,
                output_irrep_dim=path.output_irrep_dim,
            ),
        )
        outputs.append(
            _OutputContext(
                output_index=output_index,
                output_irrep_dim=path.output_irrep_dim,
                input_irrep_dim=path.input_irrep_dim,
                value_name=output_slot.name,
                terms=tuple(terms),
            )
        )

    return _PathContext(weight_start=path.weight_start, outputs=tuple(outputs))


def generate_projected_source(
    plan: FactorizedConvPlan,
    *,
    dtype: type[np.generic],
    forward_jvp_active: tuple[bool, bool, bool] = (True, True, True),
    backward_jvp_active: tuple[bool, bool, bool, bool] = (
        True,
        True,
        True,
        True,
    ),
    is_hip: bool = False,
) -> GeneratedProjectedSource:
    """Render forward, reverse, and mixed-derivative factorized kernels.

    ``is_hip`` selects HIP-compatible intrinsic spellings in the rendered
    source. HIP shuffles retain the kernels' logical 32-lane ownership groups,
    including on hardware with 64-lane wavefronts.

    ``forward_jvp_active`` corresponds, in order, to tangents of the sender
    features ``x``, edge features ``sh``, and projected edge ``weights``.
    ``backward_jvp_active`` uses the same first three entries and adds the
    tangent of ``dout``, the output cotangent entering the reverse pass.

    An inactive entry is rendered as ``scalar_t(0)`` instead of a buffer read.
    The tuple is therefore a compile-time derivative specialisation selected by
    JAX's active tangent directions.

    :param plan: Static sparse coupling paths and buffer dimensions.
    :param dtype: Generated scalar dtype, either float32 or float64.
    :param forward_jvp_active: Activity of ``(x, sh, weights)`` tangents.
    :param backward_jvp_active: Activity of ``(x, sh, weights, dout)`` tangents.
    :param is_hip: Render HIP-compatible rather than CUDA intrinsic spellings.
    :return: Rendered source and its stable cache hash.
    """
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("projected factorized convolution supports f32 and f64")

    scalar = "float" if dtype == np.dtype(np.float32) else "double"

    # Lower the plan to Jinja-ready records. This resolves flat indices and
    # exact C++ coefficient literals, while interning accumulator names shared
    # by several sparse coupling terms.
    output_slots = {}
    input_gradient_slots = {}
    paths = tuple(
        _render_path_context(
            path,
            scalar,
            output_slots,
            input_gradient_slots,
        )
        for path in plan.paths
    )

    environment = get_jinja_environment(is_hip=is_hip)
    template = environment.get_template("factorized_projected.cuh")
    source = template.render(
        scalar=scalar,
        plan=plan,
        paths=paths,
        output_slots=tuple(output_slots.values()),
        input_gradient_slots=tuple(input_gradient_slots.values()),
        forward_tangent=_tangent_context(forward_jvp_active),
        tangent=_tangent_context(backward_jvp_active, include_output=True),
    )
    return GeneratedProjectedSource(
        source=source,
        source_hash=hash_str_64(source),
    )


__all__ = ["GeneratedProjectedSource", "generate_projected_source"]
