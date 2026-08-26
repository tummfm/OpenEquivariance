"""Static plans for generic symmetric contractions."""

# The recursive higher-order real Clebsch--Gordan construction is adapted from
# MACE ``mace.tools.cg`` (Ilyes Batatia; based on e3nn work by Mario Geiger):
# https://github.com/ACEsuit/mace/blob/b5faaa076c49778fc17493edfecebcabeb960155/mace/tools/cg.py
# See the MACE MIT notice in THIRD_PARTY_NOTICES.

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from openequivariance.core.e3nn_lite import Irrep, Irreps, wigner_3j


def _couple_coefficients(
    left_coefficient,
    left_irreps,
    left_irrep,
    right_irrep,
    output_irrep,
    normalization,
    dtype,
):
    """Couple an accumulated left product to one right-hand irrep."""
    coupling = wigner_3j(
        output_irrep.l,
        left_irrep.l,
        right_irrep.l,
        dtype=dtype,
    )
    if normalization == "component":
        coupling *= output_irrep.dim**0.5
    elif normalization == "norm":
        coupling *= left_irrep.dim**0.5 * right_irrep.dim**0.5

    # j is the intermediate component, q flattens the previous input axes,
    # i is the new output component, and l is the new right-hand component.
    flattened_left = left_coefficient.reshape(left_coefficient.shape[0], -1)
    return np.einsum("jq,ijl->iql", flattened_left, coupling).reshape(
        output_irrep.dim,
        *(irreps.dim for irreps in left_irreps),
        right_irrep.dim,
    )


def _enumerate_recursive_couplings(
    irrepss: Sequence[Irreps],
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
):
    """Enumerate coefficient tensors for recursively coupled input factors.

    Each result is an ``(output_irrep, coefficient)`` pair.  The coefficient
    has one leading output-component axis followed by one flattened input axis
    per factor.  Multiplicity copies are embedded in their own input slices.
    """
    irrepss = [Irreps(irreps) for irreps in irrepss]
    if filter_ir_mid is not None:
        filter_ir_mid = [Irrep(ir) for ir in filter_ir_mid]

    # Degree one is the identity map for every irrep copy.
    if len(irrepss) == 1:
        (irreps,) = irrepss
        result = []
        eye = np.eye(irreps.dim, dtype=dtype)
        offset = 0
        for mul, irrep in irreps:
            for _ in range(mul):
                sl = slice(offset, offset + irrep.dim)
                result.append((irrep, eye[sl]))
                offset += irrep.dim
        return result

    # Recursively couple the left product, then append one right-hand factor.
    *left_irreps, right_irreps = irrepss
    result = []
    for left_irrep, left_coefficient in _enumerate_recursive_couplings(
        left_irreps,
        normalization=normalization,
        filter_ir_mid=filter_ir_mid,
        dtype=dtype,
    ):
        offset = 0
        for mul, right_irrep in right_irreps:
            allowed_outputs = tuple(left_irrep * right_irrep)
            for output_irrep in allowed_outputs:
                if filter_ir_mid is not None and output_irrep not in filter_ir_mid:
                    continue
                coefficient = _couple_coefficients(
                    left_coefficient,
                    left_irreps,
                    left_irrep,
                    right_irrep,
                    output_irrep,
                    normalization,
                    dtype,
                )

                # Each multiplicity copy occupies a separate slice of the
                # flattened right-hand feature axis.
                for multiplicity in range(mul):
                    right_slice = slice(
                        offset + multiplicity * right_irrep.dim,
                        offset + (multiplicity + 1) * right_irrep.dim,
                    )
                    embedded = np.zeros(
                        (
                            output_irrep.dim,
                            *(irreps.dim for irreps in left_irreps),
                            right_irreps.dim,
                        ),
                        dtype=dtype,
                    )
                    embedded[..., right_slice] = coefficient
                    result.append((output_irrep, embedded))
            offset += mul * right_irrep.dim
    return sorted(result, key=lambda item: item[0])


def _normalize_dtype(dtype) -> np.dtype:
    """Normalize a supported real coupling dtype."""
    dtype = np.dtype(np.float64 if dtype is None else dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("symmetric couplings support float32 and float64")
    return dtype


def _coupling_basis_for_output(
    irreps_in,
    output_irrep,
    correlation: int,
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
):
    r"""Construct the real O(3) coupling basis for one final irrep.

    The repeated input representation is coupled ``correlation`` times.  The
    final axis enumerates retained coupling tensors. It is not an
    input-feature axis.  For a coupling of degree :math:`d` to an output irrep
    of dimension :math:`D_{\rm out}`, the tensor has shape
    :math:`(D_{\rm out}, D_{\rm in}, \ldots, D_{\rm in}, P)`, with ``d``
    copies of :math:`D_{\rm in}` and ``P`` retained coupling routes.  With no
    intermediate filter, every irrep which can contribute to a requested final
    irrep is retained.

    :param irreps_in: Irreps of one repeated input factor. Multiplicities are
        part of the flattened input dimension.
    :param output_irrep: Final irrep to retain after the complete product.
    :param correlation: Number of repeated input factors. Must be positive.
    :param normalization: Clebsch--Gordan normalisation, one of
        ``"component"``, ``"norm"``, or ``"none"``.
    :param filter_ir_mid: Optional allowed intermediate irreps. ``None`` keeps
        every representation-theoretically allowed intermediate irrep.
    :param dtype: Coefficient dtype, float32 or float64. ``None`` selects
        float64.
    :return: Dense coefficient tensor for the requested output irrep.
    :raises ValueError: If the correlation or normalisation is invalid, or no
        requested final coupling exists.
    :raises TypeError: If ``dtype`` is not float32 or float64.
    """
    if correlation < 1:
        raise ValueError("correlation must be positive")
    if normalization not in ("component", "norm", "none"):
        raise ValueError("normalization must be 'component', 'norm', or 'none'")
    dtype = _normalize_dtype(dtype)
    output_irrep = Irrep(output_irrep)
    irrepss = [Irreps(irreps_in)] * correlation

    stack = []
    for irrep, coefficient in _enumerate_recursive_couplings(
        irrepss, normalization, filter_ir_mid, dtype
    ):
        if irrep != output_irrep:
            continue
        # A fixed [out_component, input_component..., path] convention keeps
        # the tensor rank independent of scalar input or output spaces.
        stack.append(coefficient)
    if not stack:
        raise ValueError(
            f"no coupling paths from {irreps_in} to {output_irrep} "
            f"at correlation {correlation}"
        )
    return np.stack(stack, axis=-1)


@dataclass(frozen=True, slots=True)
class SymmetricPath:
    """One fixed nonzero monomial and its learned-weight location.

    ``feature_components`` selects one component from every repeated input
    factor. ``coupling_index`` selects its fixed coupling basis and therefore
    the matching learned coefficient within each species/channel weight group.
    """

    output_base: int
    output_irrep_dim: int
    output_component: int
    weight_base: int
    couplings_per_channel: int
    coupling_index: int
    feature_components: tuple[int, ...]
    coefficient: float


@dataclass(frozen=True, slots=True)
class SymmetricContractionPlan:
    """Static sparse polynomial paths and packed feature/weight layouts.

    ``paths`` index canonical rank-three features with one node axis followed
    by the layout named by ``feature_layout``.  A packed weight table has shape
    ``[species, weight_dim]``. ``weight_shapes`` records the corresponding
    external ``[species, coupling_basis, channel]`` groups.
    """

    channels: int
    feature_dim: int
    output_dim: int
    num_elements: int
    weight_dim: int
    weight_shapes: tuple[tuple[tuple[int, int, int], ...], ...]
    paths: tuple[SymmetricPath, ...]
    feature_layout: str = "channel_feature"

    def __post_init__(self):
        """Validate the canonical rank-three feature layout."""
        if self.feature_layout not in ("channel_feature", "feature_channel"):
            raise ValueError(
                "feature_layout must be 'channel_feature' or 'feature_channel'"
            )


def _correlation_for(correlation, mul_ir) -> int:
    """Resolve a global or output-specific correlation order."""

    if not isinstance(correlation, Mapping):
        return int(correlation)
    candidates = dict.fromkeys((mul_ir, mul_ir.ir, str(mul_ir), str(mul_ir.ir)))
    matches = [int(correlation[key]) for key in candidates if key in correlation]
    if len(set(matches)) > 1:
        raise ValueError(f"conflicting correlation aliases for output irrep {mul_ir}")
    if matches:
        return matches[0]
    raise ValueError(f"missing correlation for output irrep {mul_ir}")


def _nonzero_paths(tensor, output_base, weight_base):
    """Convert a dense coupling tensor to sparse monomial paths."""
    coupling_count = int(tensor.shape[-1])
    paths = []
    for coordinate in zip(*np.nonzero(tensor), strict=True):
        output_component, *feature_components, coupling_index = coordinate
        paths.append(
            SymmetricPath(
                output_base=output_base,
                output_irrep_dim=int(tensor.shape[0]),
                output_component=int(output_component),
                weight_base=weight_base,
                couplings_per_channel=coupling_count,
                coupling_index=int(coupling_index),
                feature_components=tuple(int(value) for value in feature_components),
                coefficient=float(tensor[coordinate]),
            )
        )
    return paths


def build_symmetric_contraction_plan(
    irreps_in,
    irreps_out,
    correlation,
    *,
    num_elements: int,
    irrep_normalization: str = "component",
    dtype=np.float64,
) -> SymmetricContractionPlan:
    r"""Build a sparse symmetric-contraction plan from irreps.

    One output block receives the sum of every degree from its configured
    maximum down to one.  At degree :math:`d`, the fixed angular tensor
    couples :math:`A^{\otimes d}` to that final output block. Each retained
    coupling route receives a distinct learned weight for every species and
    channel.  Channel multiplicities are intentionally removed from this
    angular construction and restored in the packed layout.

    :param irreps_in: Channel-uniform input irreps. Every block must have the
        same multiplicity.
    :param irreps_out: Channel-uniform requested output irreps, with the same
        multiplicity as ``irreps_in``.
    :param correlation: Positive global degree or mapping from an output irrep
        (or one of its aliases) to its positive maximum degree.
    :param num_elements: Positive number of species rows in the weight table.
    :param irrep_normalization: Clebsch--Gordan normalisation, one of
        ``"component"``, ``"norm"``, or ``"none"``.
    :param dtype: Fixed coupling-coefficient dtype, float32 or float64.
    :return: Canonical paths, output layout, and external weight shapes.
    :raises ValueError: If layouts, multiplicities, correlation, or species
        count are incompatible.
    :raises TypeError: If ``dtype`` is not float32 or float64.
    """

    # Validate the channel-uniform representation layout.
    irreps_in = Irreps(irreps_in)
    irreps_out = Irreps(irreps_out)
    if num_elements <= 0:
        raise ValueError("num_elements must be positive")
    if irrep_normalization not in ("component", "norm", "none"):
        raise ValueError(
            "irrep_normalization must be 'component', 'norm', or 'none'"
        )
    np_dtype = _normalize_dtype(dtype)

    if not irreps_in:
        raise ValueError("irreps_in must not be empty")
    channels = irreps_in[0].mul
    if channels <= 0:
        raise ValueError("the first input irrep must have positive multiplicity")
    if any(mul_ir.mul != channels for mul_ir in irreps_in):
        raise ValueError("all input irreps must have the same channel multiplicity")
    if any(mul_ir.mul != channels for mul_ir in irreps_out):
        raise ValueError("all output irreps must match the input channel count")

    # Remove channel multiplicities from the angular coupling problem.
    coupling_irreps = Irreps([mul_ir.ir for mul_ir in irreps_in])
    feature_dim = coupling_irreps.dim
    weight_shapes = []
    paths = []
    weight_cursor = 0
    output_cursor = 0
    for mul_ir in irreps_out:
        max_correlation = _correlation_for(correlation, mul_ir)
        if max_correlation < 1:
            raise ValueError("correlation must be positive")
        contraction_shapes = []
        for degree in range(max_correlation, 0, -1):
            tensor = _coupling_basis_for_output(
                coupling_irreps,
                mul_ir.ir,
                degree,
                normalization=irrep_normalization,
                dtype=np_dtype,
            )
            expected = (mul_ir.ir.dim,) + (feature_dim,) * degree
            if tensor.shape[:-1] != expected:
                raise ValueError(
                    f"unexpected coupling shape {tensor.shape}; expected "
                    f"{expected + (tensor.shape[-1],)}"
                )
            coupling_count = int(tensor.shape[-1])
            contraction_shapes.append(
                (int(num_elements), coupling_count, channels)
            )
            # Store only nonzero component couplings.
            paths.extend(_nonzero_paths(tensor, output_cursor, weight_cursor))
            weight_cursor += channels * coupling_count
        weight_shapes.append(tuple(contraction_shapes))
        output_cursor += channels * mul_ir.ir.dim

    return SymmetricContractionPlan(
        channels=channels,
        feature_dim=feature_dim,
        output_dim=irreps_out.dim,
        num_elements=int(num_elements),
        weight_dim=weight_cursor,
        weight_shapes=tuple(weight_shapes),
        paths=tuple(paths),
    )


__all__ = [
    "SymmetricContractionPlan",
    "SymmetricPath",
    "build_symmetric_contraction_plan",
]
