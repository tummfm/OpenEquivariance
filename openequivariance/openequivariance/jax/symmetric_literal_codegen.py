"""Render literal runtime-JIT sources for symmetric contractions."""

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from openequivariance.core.SymmetricContractionPlan import SymmetricContractionPlan
from openequivariance.templates.jinja_utils import get_jinja_environment


SYMMETRIC_LITERAL_PATH_LIMIT = 1024


@dataclass(frozen=True, slots=True)
class SymmetricLiteralEligibility:
    """Generated literal-kernel eligibility and its diagnostic.

    ``reason`` is ``None`` exactly when ``eligible`` is true.  ``max_degree``
    is reported even for an ineligible plan so callers can distinguish a
    degree limit from a layout or path-validation failure.
    """

    eligible: bool
    reason: str | None
    max_degree: int


@dataclass(frozen=True, slots=True)
class GeneratedSymmetricLiteralSource:
    """Rendered literal kernel source and deterministic 64-bit cache hash.

    The source embeds only a static sparse plan. Node and species dimensions
    remain runtime operands, which preserves dynamic leading dimensions for
    export.
    """

    source: str
    source_hash: int


@dataclass(frozen=True, slots=True)
class _LiteralPath:
    index: int
    output_base: int
    output_irrep_dim: int
    output_component: int
    weight_base: int
    couplings_per_channel: int
    coupling_index: int
    feature_components: tuple[int, ...]
    coefficient: str


@dataclass(frozen=True, slots=True)
class _LiteralOutputGroup:
    output_base: int
    output_irrep_dim: int
    output_component: int
    paths: tuple[_LiteralPath, ...]


@dataclass(frozen=True, slots=True)
class _SymmetricLiteralRenderContext:
    scalar: str
    channels: int
    feature_dim: int
    output_dim: int
    weight_dim: int
    max_degree: int
    feature_layout: str
    paths: tuple[_LiteralPath, ...]
    output_groups: tuple[_LiteralOutputGroup, ...]
    out_starts: tuple[int, ...]
    degrees: tuple[int, ...]
    feature_components: tuple[int, ...]


def _output_key(path) -> tuple[int, int, int]:
    return (
        int(path.output_base),
        int(path.output_irrep_dim),
        int(path.output_component),
    )


def _canonical_key(path):
    return (
        *_output_key(path),
        int(path.weight_base),
        int(path.couplings_per_channel),
        int(path.coupling_index),
        tuple(int(value) for value in path.feature_components),
    )


def symmetric_literal_eligibility(
    plan: SymmetricContractionPlan,
) -> SymmetricLiteralEligibility:
    """Check whether a plan can use the bounded literal species kernel.

    The generated kernel is limited to canonical, duplicate-free monomials of
    degree one through three and to ``SYMMETRIC_LITERAL_PATH_LIMIT`` paths. It
    also requires every flat output entry to have exactly one channel owner.
    These checks keep generated reductions explicit. An ineligible plan uses
    the ordinary exportable JAX evaluator instead.

    :param plan: Canonical sparse symmetric-contraction plan to validate.
    :return: Eligibility, explanatory reason when ineligible, and maximum
        observed polynomial degree.
    """
    paths = tuple(plan.paths)
    path_count = len(paths)
    max_degree = max((len(path.feature_components) for path in paths), default=0)
    def result(reason: str | None) -> SymmetricLiteralEligibility:
        return SymmetricLiteralEligibility(
            eligible=reason is None,
            reason=reason,
            max_degree=max_degree,
        )

    if plan.channels <= 0:
        return result("channel count must be positive")
    if plan.feature_dim <= 0 or plan.output_dim <= 0:
        return result("feature and output dimensions must be positive")
    if plan.num_elements <= 0 or plan.weight_dim <= 0:
        return result("element and weight dimensions must be positive")
    if not paths:
        return result("the literal kernel requires at least one path")
    if path_count > SYMMETRIC_LITERAL_PATH_LIMIT:
        return result(
            f"canonical path count {path_count} exceeds literal limit "
            f"{SYMMETRIC_LITERAL_PATH_LIMIT}"
        )
    if max_degree > 3 or any(not path.feature_components for path in paths):
        return result("the literal kernel supports degrees one through three")

    seen = set()
    output_owners: dict[int, int] = {}
    for path in paths:
        components = tuple(int(value) for value in path.feature_components)
        if components != tuple(sorted(components)):
            return result("feature factors must use canonical sorted order")
        key = _canonical_key(path)
        if key in seen:
            return result(
                "duplicate monomials must be combined before literal generation"
            )
        seen.add(key)
        if not math.isfinite(float(path.coefficient)):
            return result("path coefficients must be finite")
        if any(
            component < 0 or component >= plan.feature_dim for component in components
        ):
            return result("path feature component is outside the feature dimension")
        if (
            path.weight_base < 0
            or path.couplings_per_channel <= 0
            or path.coupling_index < 0
            or path.coupling_index >= path.couplings_per_channel
        ):
            return result("path weight metadata is invalid")
        last_weight = (
            path.weight_base
            + (plan.channels - 1) * path.couplings_per_channel
            + path.coupling_index
        )
        if last_weight >= plan.weight_dim:
            return result("path weight index is outside the original flat weight leaf")
        if (
            path.output_base < 0
            or path.output_irrep_dim <= 0
            or path.output_component < 0
            or path.output_component >= path.output_irrep_dim
        ):
            return result("path output metadata is invalid")
        for channel in range(plan.channels):
            output = (
                path.output_base
                + channel * path.output_irrep_dim
                + path.output_component
            )
            if output < 0 or output >= plan.output_dim:
                return result("path output index is outside the output dimension")
            owner = output_owners.setdefault(output, channel)
            if owner != channel:
                return result("output entries are not owned by a unique channel")

    if set(output_owners) != set(range(plan.output_dim)):
        return result("literal output groups do not cover the complete output buffer")
    return result(None)


def _literal(value: float, scalar: str) -> str:
    suffix = "f" if scalar == "float" else ""
    return f"static_cast<{scalar}>({float(value).hex()}{suffix})"


def _render_context(
    plan: SymmetricContractionPlan,
    scalar: str,
    eligibility: SymmetricLiteralEligibility,
) -> _SymmetricLiteralRenderContext:
    paths = tuple(sorted(plan.paths, key=_output_key))
    output_keys = tuple(dict.fromkeys(_output_key(path) for path in paths))
    path_contexts = tuple(
        _LiteralPath(
            index=index,
            output_base=int(path.output_base),
            output_irrep_dim=int(path.output_irrep_dim),
            output_component=int(path.output_component),
            weight_base=int(path.weight_base),
            couplings_per_channel=int(path.couplings_per_channel),
            coupling_index=int(path.coupling_index),
            feature_components=tuple(int(value) for value in path.feature_components),
            coefficient=_literal(path.coefficient, scalar),
        )
        for index, path in enumerate(paths)
    )
    output_groups = tuple(
        _LiteralOutputGroup(
            output_base=key[0],
            output_irrep_dim=key[1],
            output_component=key[2],
            paths=tuple(
                path
                for path in path_contexts
                if (path.output_base, path.output_irrep_dim, path.output_component)
                == key
            ),
        )
        for key in output_keys
    )
    out_starts = (0,)
    for group in output_groups:
        out_starts += (out_starts[-1] + len(group.paths),)
    max_degree = eligibility.max_degree
    return _SymmetricLiteralRenderContext(
        scalar=scalar,
        channels=int(plan.channels),
        feature_dim=int(plan.feature_dim),
        output_dim=int(plan.output_dim),
        weight_dim=int(plan.weight_dim),
        max_degree=max_degree,
        feature_layout=plan.feature_layout,
        paths=path_contexts,
        output_groups=output_groups,
        out_starts=out_starts,
        degrees=tuple(len(path.feature_components) for path in path_contexts),
        feature_components=tuple(
            component
            for path in path_contexts
            for component in path.feature_components
            + (0,) * (max_degree - len(path.feature_components))
        ),
    )


def generate_symmetric_literal_source(
    plan: SymmetricContractionPlan,
    *,
    dtype: type[np.generic] | np.dtype,
    is_hip: bool = False,
) -> GeneratedSymmetricLiteralSource:
    """Render literal symmetric-contraction source for CUDA or HIP.

    The source implements a species-selected polynomial over canonical
    features ``[nodes, channels, feature_dim]`` and a packed table
    ``[species, weight_dim]``.  It contains static plan data only: node count,
    species count, and all other leading runtime dimensions remain operands of
    the generated call.

    :param plan: Eligible canonical sparse symmetric-contraction plan.
    :param dtype: Generated scalar dtype, float32 or float64.
    :param is_hip: Render HIP-safe intrinsic spellings instead of CUDA
        spellings. This changes source text only. Compilation is selected by
        the native extension backend.
    :return: Rendered native source and its deterministic 64-bit cache hash.
    :raises ValueError: If the plan is ineligible or ``dtype`` is unsupported.
    """
    eligibility = symmetric_literal_eligibility(plan)
    if not eligibility.eligible:
        raise ValueError(
            "symmetric literal kernel is ineligible; use JAX fallback: "
            f"{eligibility.reason}"
        )
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("symmetric literal kernel supports f32 and f64")
    scalar = "float" if dtype == np.dtype(np.float32) else "double"
    context = _render_context(plan, scalar, eligibility)
    source = (
        get_jinja_environment(is_hip=is_hip)
        .get_template("symmetric_literal.cuh")
        .render(context=context)
    )
    digest = hashlib.sha256(source.encode()).digest()
    return GeneratedSymmetricLiteralSource(
        source=source,
        source_hash=int.from_bytes(digest[:8], "little", signed=True),
    )


__all__ = [
    "GeneratedSymmetricLiteralSource",
    "SYMMETRIC_LITERAL_PATH_LIMIT",
    "SymmetricLiteralEligibility",
    "generate_symmetric_literal_source",
    "symmetric_literal_eligibility",
]
