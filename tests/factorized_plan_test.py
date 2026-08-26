import numpy as np
import pytest

from openequivariance.core.e3nn_lite import Irreps, TPProblem, wigner_3j
from openequivariance.core.FactorizedConvPlan import factorized_plan_from_problem


def _problem(mode="uvu", edge_mul=1, channels=4):
    return TPProblem(
        Irreps(f"{channels}x1e"),
        Irreps(f"{edge_mul}x1e"),
        Irreps(f"{channels}x1e"),
        [(0, 0, 0, mode, True)],
        shared_weights=False,
        internal_weights=False,
        irrep_dtype=np.float64,
        weight_dtype=np.float64,
    )


def test_factorized_plan_preserves_problem_layout_and_normalization():
    """Match a small uvu plan's layout and normalized CG tensor."""
    problem = _problem()
    plan = factorized_plan_from_problem(problem)
    path = plan.paths[0]
    assert plan.input_dim == problem.irreps_in1.dim
    assert plan.edge_dim == problem.irreps_in2.dim
    assert plan.output_dim == problem.irreps_out.dim
    assert plan.weight_numel == problem.weight_numel
    assert path.weight_start == 0

    expected = wigner_3j(1, 1, 1) * problem.instructions[0].path_weight
    actual = np.zeros_like(expected)
    actual[path.cg_input, path.cg_edge, path.cg_output] = path.cg_value
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "problem, message",
    [
        (_problem(mode="uvw"), "uvu paths only"),
        (_problem(edge_mul=2), "edge multiplicity one"),
    ],
)
def test_factorized_plan_rejects_unsupported_problem(problem, message):
    """Reject unsupported path modes and edge multiplicities."""
    with pytest.raises(ValueError, match=message):
        factorized_plan_from_problem(problem)


def test_factorized_plan_rejects_unreferenced_output_block():
    """Reject an output irrep block with no producing instruction."""
    problem = TPProblem(
        Irreps("4x1e"),
        Irreps("1x0e"),
        Irreps("4x1e + 4x0e"),
        [(0, 0, 0, "uvu", True)],
        shared_weights=False,
        internal_weights=False,
        irrep_dtype=np.float64,
        weight_dtype=np.float64,
    )
    with pytest.raises(ValueError, match="output irrep blocks \\[1\\]"):
        factorized_plan_from_problem(problem)
