"""Public plan and source-rendering coverage for symmetric contractions."""

from pathlib import Path
import runpy

import numpy as np
import pytest

def _plan_api():
    from openequivariance.core.SymmetricContractionPlan import (
        SymmetricContractionPlan,
        build_symmetric_contraction_plan,
    )
    from openequivariance.core.e3nn_lite import Irrep

    return SymmetricContractionPlan, build_symmetric_contraction_plan, Irrep


def test_plan_builder_defines_canonical_channels_layout_and_weights():
    """Build a channel-uniform mixed-irrep plan with its public layout."""
    _, build_symmetric_contraction_plan, _ = _plan_api()
    plan = build_symmetric_contraction_plan(
        "2x0e + 2x1o", "2x0e + 2x1o", 3, num_elements=4, dtype=np.float64
    )

    assert plan.channels == 2
    assert plan.feature_dim == 4
    assert plan.output_dim == 8
    assert plan.num_elements == 4
    assert plan.feature_layout == "channel_feature"
    assert plan.weight_dim == sum(
        paths * channels
        for group in plan.weight_shapes
        for _, paths, channels in group
    )
    assert plan.paths


@pytest.mark.parametrize("normalization", ("component", "norm", "none"))
@pytest.mark.parametrize("dtype", (np.float32, np.float64), ids=("f32", "f64"))
def test_plan_builder_accepts_normalization_and_real_dtypes(normalization, dtype):
    """Preserve requested normalization and real coefficient precision."""
    _, build_symmetric_contraction_plan, _ = _plan_api()
    plan = build_symmetric_contraction_plan(
        "1x1o",
        "1x1o",
        1,
        num_elements=2,
        irrep_normalization=normalization,
        dtype=dtype,
    )
    assert plan.weight_shapes == (((2, 1, 1),),)
    assert plan.paths


def test_plan_builder_accepts_output_specific_correlation():
    """Resolve equivalent public irrep aliases to one degree."""
    _, build_symmetric_contraction_plan, Irrep = _plan_api()
    plan = build_symmetric_contraction_plan(
        "1x0e + 1x1o",
        "1x0e + 1x1o",
        {Irrep("0e"): 2, "1o": 1},
        num_elements=2,
    )
    assert len(plan.weight_shapes[0]) == 2
    assert len(plan.weight_shapes[1]) == 1


@pytest.mark.parametrize(
    ("kwargs", "error"),
    (
        ({"num_elements": 0}, "num_elements must be positive"),
        ({"irrep_normalization": "invalid"}, "irrep_normalization must"),
        ({"dtype": np.int32}, "float32 and float64"),
        ({"dtype": np.complex64}, "float32 and float64"),
    ),
)
def test_plan_builder_rejects_invalid_public_configuration(kwargs, error):
    """Reject unsupported species, normalization, and coefficient settings."""
    _, build_symmetric_contraction_plan, _ = _plan_api()
    with pytest.raises((TypeError, ValueError), match=error):
        build_symmetric_contraction_plan(
            "1x0e", "1x0e", 1, **{"num_elements": 1, **kwargs}
        )


def test_plan_builder_rejects_incompatible_irrep_layouts_and_aliases():
    """Reject empty, non-uniform, and contradictory public layouts."""
    _, build_symmetric_contraction_plan, Irrep = _plan_api()
    with pytest.raises(ValueError, match="irreps_in must not be empty"):
        build_symmetric_contraction_plan("", "1x0e", 1, num_elements=1)
    with pytest.raises(ValueError, match="same channel multiplicity"):
        build_symmetric_contraction_plan("1x0e + 2x1o", "1x0e", 1, num_elements=1)
    with pytest.raises(ValueError, match="conflicting correlation aliases"):
        build_symmetric_contraction_plan(
            "1x0e", "1x0e", {Irrep("0e"): 1, "0e": 2}, num_elements=1
        )


def test_plan_layout_validation_is_public_and_strict():
    """Reject an unsupported canonical feature layout."""
    SymmetricContractionPlan, _, _ = _plan_api()
    with pytest.raises(ValueError, match="feature_layout must"):
        SymmetricContractionPlan(1, 1, 1, 1, 1, (), (), "invalid")


@pytest.mark.parametrize(
    ("feature_layout", "expected_abi_value"),
    (("channel_feature", 0), ("feature_channel", 1)),
)
def test_symmetric_feature_layout_is_an_explicit_stable_ffi_attribute(
    feature_layout, expected_abi_value
):
    """Encode each public [N,C,F]/[N,F,C] layout for native validation."""
    ffi_targets = runpy.run_path(
        Path(__file__).parents[1]
        / "openequivariance"
        / "openequivariance"
        / "jax"
        / "ffi_targets.py"
    )

    assert (
        ffi_targets["symmetric_feature_layout_abi_value"](feature_layout)
        == expected_abi_value
    )
    assert ffi_targets["SYMMETRIC_LAYOUT_CHANNEL_FEATURE"] == 0
    assert ffi_targets["SYMMETRIC_LAYOUT_FEATURE_CHANNEL"] == 1
    with pytest.raises(ValueError, match="unsupported symmetric feature layout"):
        ffi_targets["symmetric_feature_layout_abi_value"]("invalid")


def test_literal_source_renders_cuda_and_hip_runtime_dimensions(with_jax):
    """Render minimal CUDA and HIP literal kernels with dynamic nodes."""
    if not with_jax:
        pytest.skip("requires --jax")
    SymmetricContractionPlan, _, _ = _plan_api()
    from openequivariance.core.SymmetricContractionPlan import SymmetricPath
    from openequivariance.jax.symmetric_literal_codegen import (
        generate_symmetric_literal_source,
    )

    plan = SymmetricContractionPlan(
        channels=1,
        feature_dim=1,
        output_dim=1,
        num_elements=2,
        weight_dim=1,
        weight_shapes=(),
        paths=(SymmetricPath(0, 1, 0, 0, 1, 0, (0,), 1.0),),
    )
    cuda = generate_symmetric_literal_source(plan, dtype=np.float32).source
    hip = generate_symmetric_literal_source(plan, dtype=np.float32, is_hip=True).source

    assert "oeq_symmetric_literal_forward_species" in cuda
    assert "int64_t nodes," in cuda
    assert "atomicAdd(" in cuda
    assert "oeq_symmetric_literal_forward_species" in hip
    assert "int64_t nodes," in hip
    assert "unsafeAtomicAdd(" in hip
