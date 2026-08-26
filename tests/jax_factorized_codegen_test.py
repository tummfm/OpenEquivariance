"""Source coverage for projected factorized convolution kernels."""

import numpy as np
import pytest


def _codegen(with_jax):
    if not with_jax:
        pytest.skip("requires --jax")
    from openequivariance.benchmark.problems import mace_problems
    from openequivariance.core.FactorizedConvPlan import factorized_plan_from_problem
    from openequivariance.jax.factorized_projected_codegen import (
        generate_projected_source,
    )

    return mace_problems, factorized_plan_from_problem, generate_projected_source


@pytest.mark.parametrize("dtype", (np.float32, np.float64), ids=("f32", "f64"))
def test_projected_source_has_fixed_arity_targets_and_dynamic_graph_sizes(
    with_jax, dtype
):
    """Emit all projected kernels with runtime node and edge dimensions."""
    mace_problems, factorized_plan_from_problem, generate_projected_source = _codegen(
        with_jax
    )
    plan = factorized_plan_from_problem(mace_problems()[0].clone())
    source = generate_projected_source(plan, dtype=dtype).source

    for target in (
        "oeq_projected_forward",
        "oeq_projected_forward_jvp",
        "oeq_projected_spatial_backward",
        "oeq_projected_weight_backward",
        "oeq_projected_spatial_backward_jvp",
    ):
        assert target in source
    assert source.count('extern "C" __global__') == 5
    assert "int64_t nodes," in source
    assert "int64_t edges," in source
    assert "projection" not in source


def test_projected_double_backward_masks_inactive_tangents(with_jax):
    """Render symbolic-zero tangent loads for inactive double-backward inputs."""
    mace_problems, factorized_plan_from_problem, generate_projected_source = _codegen(
        with_jax
    )
    source = generate_projected_source(
        factorized_plan_from_problem(mace_problems()[1].clone()),
        dtype=np.float64,
        backward_jvp_active=(True, False, True, False),
    ).source

    assert "#define TX(index) (tx[(index)])" in source
    assert "#define TW(index) (tweights[(index)])" in source
    assert "#define TSH(index) (scalar_t(0))" in source
    assert "#define TDOUT(value) (scalar_t(0))" in source


def test_projected_source_uses_cuda_and_hip_32_lane_intrinsics(with_jax):
    """Render CUDA and HIP reductions in logical 32-lane groups."""
    mace_problems, factorized_plan_from_problem, generate_projected_source = _codegen(
        with_jax
    )
    plan = factorized_plan_from_problem(mace_problems()[1].clone())
    cuda = generate_projected_source(plan, dtype=np.float32).source
    hip = generate_projected_source(plan, dtype=np.float32, is_hip=True).source

    assert "atomicAdd(dx +" in cuda
    assert "__shfl_down_sync(FULL_MASK" in cuda
    assert "unsafeAtomicAdd(dx +" in hip
    assert "__shfl_down_sync" not in hip
    for token in ("threadIdx.x & 31", "threadIdx.x >> 5", "__shfl_down(", ", 32)"):
        assert token in hip
