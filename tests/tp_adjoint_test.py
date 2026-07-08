import os

import numpy as np
import pytest


# Regression coverage for JAX FFI backward outputs. The JAX custom-call result
# buffers are not guaranteed to be zero-initialized, while the OEQ backward
# kernels accumulate into those buffers.
CASES = {
    "shared_uvu_first": {
        "irreps": (
            "128x0e+128x1o+128x2e+128x3o",
            "89x0e",
            "128x0e+128x1o+128x2e+128x3o",
        ),
        "mode": "uvu",
        "shared_weights": True,
    },
    "shared_uvu_second": {
        "irreps": (
            "128x0e+128x1o",
            "89x0e",
            "128x0e",
        ),
        "mode": "uvu",
        "shared_weights": True,
    },
    "unshared_uvw": {
        "irreps": (
            "8x0e+8x1o",
            "3x0e",
            "8x0e+8x1o",
        ),
        "mode": "uvw",
        "shared_weights": False,
    },
}

ORDER_CASES = [
    (
        "shared-uvu-first-then-second",
        ("shared_uvu_first", "shared_uvu_second"),
    ),
    (
        "shared-uvu-second-then-first",
        ("shared_uvu_second", "shared_uvu_first"),
    ),
    (
        "shared-uvu-then-unshared-uvw",
        ("shared_uvu_first", "unshared_uvw"),
    ),
    (
        "unshared-uvw-then-shared-uvu",
        ("unshared_uvw", "shared_uvu_first"),
    ),
]


@pytest.fixture(scope="module")
def ctx(with_jax):
    if not with_jax:
        pytest.skip("Skipping JAX tests")

    os.environ["OEQ_NOTORCH"] = "1"

    import jax
    import jax.numpy as jnp
    import openequivariance as oeq

    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("JAX GPU device is required")

    return {"jax": jax, "jnp": jnp, "oeq": oeq}


@pytest.fixture(
    params=ORDER_CASES,
    ids=lambda case: case[0],
    scope="module",
)
def operator_order(request):
    return request.param[1]


def make_problem(oeq, irreps_in1, irreps_in2, irreps_out, mode, shared_weights):
    irreps1 = oeq.Irreps(irreps_in1)
    irreps2 = oeq.Irreps(irreps_in2)
    requested_out = oeq.Irreps(irreps_out)

    generated_out = []
    instructions = []
    for i_in1, (mul, ir_in1) in enumerate(irreps1):
        for i_in2, (_, ir_in2) in enumerate(irreps2):
            for ir_out in ir_in1 * ir_in2:
                if ir_out in requested_out:
                    i_out = len(generated_out)
                    generated_out.append((mul, ir_out))
                    instructions.append((i_in1, i_in2, i_out, mode, True))

    generated_out = oeq.Irreps(generated_out)
    generated_out, perm, _ = generated_out.sort()
    instructions = [
        (i_in1, i_in2, perm[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]
    instructions = sorted(instructions, key=lambda x: x[2])

    return oeq.TPProblem(
        irreps1,
        irreps2,
        generated_out,
        instructions,
        shared_weights=shared_weights,
        internal_weights=False,
        irrep_dtype=np.float32,
        weight_dtype=np.float32,
    )


def make_tensor_product(oeq, case):
    spec = CASES[case]
    return oeq.jax.TensorProduct(
        make_problem(oeq, *spec["irreps"], spec["mode"], spec["shared_weights"])
    )


def assert_forward_adjoint(jax, jnp, tp, seed):
    keys = jax.random.split(jax.random.PRNGKey(seed), 8)
    batch = 17

    x1 = jax.random.normal(
        keys[0], (batch, tp.config.irreps_in1.dim), dtype=jnp.float32
    )
    species = jax.random.randint(keys[1], (batch,), 0, tp.config.irreps_in2.dim)
    x2 = jax.nn.one_hot(species, tp.config.irreps_in2.dim, dtype=jnp.float32)
    weight_shape = (
        (tp.weight_numel,) if tp.config.shared_weights else (batch, tp.weight_numel)
    )
    weights = jax.random.normal(keys[2], weight_shape, dtype=jnp.float32)

    dx1 = jax.random.normal(keys[3], x1.shape, dtype=jnp.float32)
    dspecies = jax.random.randint(keys[4], (batch,), 0, tp.config.irreps_in2.dim)
    dx2 = jax.nn.one_hot(dspecies, tp.config.irreps_in2.dim, dtype=jnp.float32)
    dweights = jax.random.normal(keys[5], weights.shape, dtype=jnp.float32)
    cotangent = jax.random.normal(keys[6], (batch, tp.L3_dim), dtype=jnp.float32)

    def fn(a, b, c):
        return tp(a, b, c)

    _, jvp_out = jax.jvp(fn, (x1, x2, weights), (dx1, dx2, dweights))
    _, vjp_fn = jax.vjp(fn, x1, x2, weights)
    grad1, grad2, gradw = vjp_fn(cotangent)

    lhs = jnp.vdot(cotangent, jvp_out)
    rhs = jnp.vdot(grad1, dx1) + jnp.vdot(grad2, dx2) + jnp.vdot(gradw, dweights)
    err = jnp.abs(lhs - rhs)
    scale = jnp.maximum(jnp.maximum(jnp.abs(lhs), jnp.abs(rhs)), jnp.array(1.0))

    relative_error = float(np.asarray(err / scale))
    assert relative_error <= 5e-4, f"relative adjoint error={relative_error:.5f}"


def assert_backward_adjoint(jax, jnp, tp, seed):
    keys = jax.random.split(jax.random.PRNGKey(seed), 11)
    batch = 17

    x1 = jax.random.normal(
        keys[0], (batch, tp.config.irreps_in1.dim), dtype=jnp.float32
    )
    species = jax.random.randint(keys[1], (batch,), 0, tp.config.irreps_in2.dim)
    x2 = jax.nn.one_hot(species, tp.config.irreps_in2.dim, dtype=jnp.float32)
    weight_shape = (
        (tp.weight_numel,) if tp.config.shared_weights else (batch, tp.weight_numel)
    )
    weights = jax.random.normal(keys[2], weight_shape, dtype=jnp.float32)
    cotangent = jax.random.normal(keys[3], (batch, tp.L3_dim), dtype=jnp.float32)

    dx1 = jax.random.normal(keys[4], x1.shape, dtype=jnp.float32)
    dspecies = jax.random.randint(keys[5], (batch,), 0, tp.config.irreps_in2.dim)
    dx2 = jax.nn.one_hot(dspecies, tp.config.irreps_in2.dim, dtype=jnp.float32)
    dweights = jax.random.normal(keys[6], weights.shape, dtype=jnp.float32)
    dcotangent = jax.random.normal(keys[7], cotangent.shape, dtype=jnp.float32)

    cgrad1 = jax.random.normal(keys[8], x1.shape, dtype=jnp.float32)
    cgrad2 = jax.random.normal(keys[9], x2.shape, dtype=jnp.float32)
    cgradw = jax.random.normal(keys[10], weights.shape, dtype=jnp.float32)

    def backward_fn(a, b, c, d):
        _, vjp_fn = jax.vjp(lambda x, y, w: tp(x, y, w), a, b, c)
        return vjp_fn(d)

    _, jvp_out = jax.jvp(
        backward_fn,
        (x1, x2, weights, cotangent),
        (dx1, dx2, dweights, dcotangent),
    )
    _, vjp_fn = jax.vjp(backward_fn, x1, x2, weights, cotangent)
    input_grads = vjp_fn((cgrad1, cgrad2, cgradw))

    lhs = (
        jnp.vdot(cgrad1, jvp_out[0])
        + jnp.vdot(cgrad2, jvp_out[1])
        + jnp.vdot(cgradw, jvp_out[2])
    )
    rhs = (
        jnp.vdot(input_grads[0], dx1)
        + jnp.vdot(input_grads[1], dx2)
        + jnp.vdot(input_grads[2], dweights)
        + jnp.vdot(input_grads[3], dcotangent)
    )
    err = jnp.abs(lhs - rhs)
    scale = jnp.maximum(jnp.maximum(jnp.abs(lhs), jnp.abs(rhs)), jnp.array(1.0))

    relative_error = float(np.asarray(err / scale))
    assert relative_error <= 5e-4, f"relative adjoint error={relative_error:.5f}"


def test_tensor_product_adjoint_after_multi_operator_construction(ctx, operator_order):
    jax, jnp, oeq = ctx["jax"], ctx["jnp"], ctx["oeq"]
    operators = {case: make_tensor_product(oeq, case) for case in operator_order}
    for i, case in enumerate(operator_order):
        assert_forward_adjoint(jax, jnp, operators[case], seed=1234 + i)
        assert_backward_adjoint(jax, jnp, operators[case], seed=4321 + i)
