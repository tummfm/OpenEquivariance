"""AD primitives for generated preprojected factorized kernels."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend import core
from jax.interpreters import ad, mlir

from openequivariance.jax.factorized_projected_codegen import generate_projected_source
from openequivariance.jax.ffi_targets import (
    FACTORIZED_FORWARD,
    FACTORIZED_FORWARD_JVP,
    FACTORIZED_SPATIAL_BACKWARD,
    FACTORIZED_SPATIAL_BACKWARD_JVP,
    FACTORIZED_WEIGHT_BACKWARD,
)
from openequivariance.jax.extlib import IS_HIP


def _materialize_symbolic_zero(value):
    """Return an array for an AD value which may be a symbolic zero."""
    if ad.is_undefined_primal(value) or type(value) is ad.Zero:
        return jnp.zeros(value.aval.shape, value.aval.dtype)
    return value


def _generate_source(
    plan,
    x,
    *,
    forward_jvp_active=(True, True, True),
    backward_jvp_active=(True, True, True, True),
):
    """Generate source specialized to the active derivative operands.

    Forward activity is ordered as ``(x, sh, weights)``. Backward-JVP activity
    adds ``dout`` as its fourth entry. A true entry retains the corresponding
    tangent buffer read, while a false entry renders a compile-time zero.
    """
    return generate_projected_source(
        plan,
        dtype=np.dtype(x.dtype).type,
        forward_jvp_active=forward_jvp_active,
        backward_jvp_active=backward_jvp_active,
        is_hip=IS_HIP,
    )


def _plan_attributes(plan):
    """Return static dimensions passed to a generated FFI handler."""
    return dict(
        input_dim=plan.input_dim,
        edge_dim=plan.edge_dim,
        weight_dim=plan.weight_numel,
        output_dim=plan.output_dim,
        channels=plan.channels,
    )


def _generated_attrs(plan, dtype, **source_options):
    generated = generate_projected_source(
        plan, dtype=np.dtype(dtype).type, is_hip=IS_HIP, **source_options
    )
    return {
        "source": generated.source,
        "hash": generated.source_hash,
        **_plan_attributes(plan),
    }


def _ffi_lowering(target, operand_indices, *, source_options=None):
    """Lower directly to FFI without recursively tracing the Python impl.

    ``mlir.lower_fun`` is unsuitable here: the implementation constructs an
    ``ffi_call`` whose internal lowering cache can retain the surrounding
    primitive tracer during the nested trace.  Building the FFI custom call
    directly is both simpler and safe for repeated JIT/export traces.
    """
    source_options = source_options or (lambda params: {})

    def lowering(ctx, *operands, plan, **params):
        attributes = _generated_attrs(
            plan,
            ctx.avals_in[0].dtype,
            **source_options(params),
        )
        rule = jax.ffi.ffi_lowering(target)
        selected = tuple(operands[index] for index in operand_indices)
        return rule(
            ctx.replace(
                avals_in=tuple(ctx.avals_in[index] for index in operand_indices)
            ),
            *selected,
            **attributes,
        )

    return lowering


def _validate(plan, x, sh, weights, senders, receivers, row_ptr, dout=None):
    e = senders.shape[0]
    if x.ndim != 2 or x.shape[1] != plan.input_dim:
        raise ValueError(f"x must have shape [N, {plan.input_dim}]")
    if sh.shape != (e, plan.edge_dim) or weights.shape != (e, plan.weight_numel):
        raise ValueError("sh/weights must have E rows and plan feature widths")
    if receivers.shape != senders.shape or senders.dtype != np.dtype(np.int32):
        raise ValueError("senders/receivers must be equal int32 vectors")
    if receivers.dtype != np.dtype(np.int32) or row_ptr.dtype != np.dtype(np.int32):
        raise ValueError("topology must use int32")
    if row_ptr.shape != (x.shape[0] + 1,):
        raise ValueError("receiver row pointer must have shape [N+1]")
    if sh.dtype != x.dtype or weights.dtype != x.dtype:
        raise ValueError("floating operand dtypes must match")
    if dout is not None and (
        dout.shape != (x.shape[0], plan.output_dim) or dout.dtype != x.dtype
    ):
        raise ValueError("dout shape/dtype mismatch")


fwd_p = core.Primitive("factorized_projected_fwd")
# B_q returns the complete gradient with q=(x, sh, preprojected weights).
# The native target's "spatial" name distinguishes it from the weight-only
# target. This primitive returns all three gradients.
bwd_p = core.Primitive("factorized_projected_q_bwd")
bwd_p.multiple_results = True
weight_bwd_p = core.Primitive("factorized_projected_weight_bwd")
dbwd_p = core.Primitive("factorized_projected_q_dbwd")
dbwd_p.multiple_results = True
fwd_jvp_p = core.Primitive("factorized_projected_fwd_jvp")


def _fwd_impl(x, sh, weights, senders, receivers, row_ptr, *, plan):
    _validate(plan, x, sh, weights, senders, receivers, row_ptr)
    generated = _generate_source(plan, x)
    shape = jax.ShapeDtypeStruct((x.shape[0], plan.output_dim), x.dtype)
    return jax.ffi.ffi_call(FACTORIZED_FORWARD, shape)(
        x,
        sh,
        weights,
        senders,
        row_ptr,
        source=generated.source,
        hash=generated.source_hash,
        **_plan_attributes(plan),
    )


def _fwd_abstract(x, sh, weights, senders, receivers, row_ptr, *, plan):
    _validate(plan, x, sh, weights, senders, receivers, row_ptr)
    return jax.core.ShapedArray((x.shape[0], plan.output_dim), x.dtype)


fwd_p.def_impl(_fwd_impl)
fwd_p.def_abstract_eval(_fwd_abstract)
mlir.register_lowering(
    fwd_p,
    _ffi_lowering(FACTORIZED_FORWARD, (0, 1, 2, 3, 5)),
    platform="cuda",
)
mlir.register_lowering(
    fwd_p,
    _ffi_lowering(FACTORIZED_FORWARD, (0, 1, 2, 3, 5)),
    platform="rocm",
)


def _bwd_impl(x, sh, weights, dout, senders, receivers, row_ptr, *, plan):
    _validate(plan, x, sh, weights, senders, receivers, row_ptr, dout)
    generated = _generate_source(plan, x)
    shapes = tuple(jax.ShapeDtypeStruct(a.shape, a.dtype) for a in (x, sh, weights))
    return jax.ffi.ffi_call(FACTORIZED_SPATIAL_BACKWARD, shapes)(
        x,
        sh,
        weights,
        senders,
        receivers,
        dout,
        source=generated.source,
        hash=generated.source_hash,
        **_plan_attributes(plan),
    )


def _bwd_abstract(x, sh, weights, dout, senders, receivers, row_ptr, *, plan):
    _validate(plan, x, sh, weights, senders, receivers, row_ptr, dout)
    return tuple(jax.core.ShapedArray(a.shape, a.dtype) for a in (x, sh, weights))


bwd_p.def_impl(_bwd_impl)
bwd_p.def_abstract_eval(_bwd_abstract)
mlir.register_lowering(
    bwd_p,
    _ffi_lowering(FACTORIZED_SPATIAL_BACKWARD, (0, 1, 2, 4, 5, 3)),
    platform="cuda",
)
mlir.register_lowering(
    bwd_p,
    _ffi_lowering(FACTORIZED_SPATIAL_BACKWARD, (0, 1, 2, 4, 5, 3)),
    platform="rocm",
)


def _weight_bwd_impl(x, sh, weights, dout, senders, receivers, row_ptr, *, plan):
    _validate(plan, x, sh, weights, senders, receivers, row_ptr, dout)
    generated = _generate_source(plan, x)
    shape = jax.ShapeDtypeStruct(weights.shape, weights.dtype)
    return jax.ffi.ffi_call(FACTORIZED_WEIGHT_BACKWARD, shape)(
        x,
        sh,
        senders,
        receivers,
        dout,
        source=generated.source,
        hash=generated.source_hash,
        **_plan_attributes(plan),
    )


def _weight_bwd_abstract(x, sh, weights, dout, senders, receivers, row_ptr, *, plan):
    _validate(plan, x, sh, weights, senders, receivers, row_ptr, dout)
    return jax.core.ShapedArray(weights.shape, weights.dtype)


weight_bwd_p.def_impl(_weight_bwd_impl)
weight_bwd_p.def_abstract_eval(_weight_bwd_abstract)
mlir.register_lowering(
    weight_bwd_p,
    _ffi_lowering(FACTORIZED_WEIGHT_BACKWARD, (0, 1, 4, 5, 3)),
    platform="cuda",
)
mlir.register_lowering(
    weight_bwd_p,
    _ffi_lowering(FACTORIZED_WEIGHT_BACKWARD, (0, 1, 4, 5, 3)),
    platform="rocm",
)


def _dbwd_impl(
    x,
    sh,
    weights,
    dout,
    tx,
    tsh,
    tweights,
    tdout,
    senders,
    receivers,
    row_ptr,
    *,
    plan,
    active,
):
    _validate(plan, x, sh, weights, senders, receivers, row_ptr, dout)
    generated = _generate_source(plan, x, backward_jvp_active=active)
    shapes = tuple(jax.ShapeDtypeStruct(a.shape, a.dtype) for a in (x, sh, weights))
    return jax.ffi.ffi_call(FACTORIZED_SPATIAL_BACKWARD_JVP, shapes)(
        x,
        sh,
        weights,
        senders,
        receivers,
        dout,
        tx,
        tsh,
        tweights,
        tdout,
        source=generated.source,
        hash=generated.source_hash,
        **_plan_attributes(plan),
    )


def _dbwd_abstract(
    x,
    sh,
    weights,
    dout,
    tx,
    tsh,
    tweights,
    tdout,
    senders,
    receivers,
    row_ptr,
    *,
    plan,
    active,
):
    _validate(plan, x, sh, weights, senders, receivers, row_ptr, dout)
    for a, b, requested in zip(
        (x, sh, weights, dout), (tx, tsh, tweights, tdout), active
    ):
        if requested and (a.shape != b.shape or a.dtype != b.dtype):
            raise ValueError("tangent shape/dtype mismatch")
    return tuple(jax.core.ShapedArray(a.shape, a.dtype) for a in (x, sh, weights))


dbwd_p.def_impl(_dbwd_impl)
dbwd_p.def_abstract_eval(_dbwd_abstract)
mlir.register_lowering(
    dbwd_p,
    _ffi_lowering(
        FACTORIZED_SPATIAL_BACKWARD_JVP,
        (0, 1, 2, 8, 9, 3, 4, 5, 6, 7),
        source_options=lambda params: {"backward_jvp_active": params["active"]},
    ),
    platform="cuda",
)
mlir.register_lowering(
    dbwd_p,
    _ffi_lowering(
        FACTORIZED_SPATIAL_BACKWARD_JVP,
        (0, 1, 2, 8, 9, 3, 4, 5, 6, 7),
        source_options=lambda params: {"backward_jvp_active": params["active"]},
    ),
    platform="rocm",
)


def _fwd_jvp_impl(
    x, sh, weights, tx, tsh, tweights, senders, receivers, row_ptr, *, plan, active
):
    if not any(active):
        return jnp.zeros((x.shape[0], plan.output_dim), dtype=x.dtype)
    generated = _generate_source(plan, x, forward_jvp_active=active)
    shape = jax.ShapeDtypeStruct((x.shape[0], plan.output_dim), x.dtype)
    return jax.ffi.ffi_call(FACTORIZED_FORWARD_JVP, shape)(
        x,
        sh,
        weights,
        senders,
        row_ptr,
        tx,
        tsh,
        tweights,
        source=generated.source,
        hash=generated.source_hash,
        **_plan_attributes(plan),
    )


def _fwd_jvp_abstract(
    x, sh, weights, tx, tsh, tweights, senders, receivers, row_ptr, *, plan, active
):
    return _fwd_abstract(x, sh, weights, senders, receivers, row_ptr, plan=plan)


fwd_jvp_p.def_impl(_fwd_jvp_impl)
fwd_jvp_p.def_abstract_eval(_fwd_jvp_abstract)
mlir.register_lowering(
    fwd_jvp_p,
    _ffi_lowering(
        FACTORIZED_FORWARD_JVP,
        (0, 1, 2, 6, 8, 3, 4, 5),
        source_options=lambda params: {"forward_jvp_active": params["active"]},
    ),
    platform="cuda",
)
mlir.register_lowering(
    fwd_jvp_p,
    _ffi_lowering(
        FACTORIZED_FORWARD_JVP,
        (0, 1, 2, 6, 8, 3, 4, 5),
        source_options=lambda params: {"forward_jvp_active": params["active"]},
    ),
    platform="rocm",
)


def _fwd_jvp_rule(primals, tangents, *, plan):
    x, sh, weights, senders, receivers, row_ptr = primals
    active = tuple(type(tangent) is not ad.Zero for tangent in tangents[:3])
    tx, tsh, tw = (
        _materialize_symbolic_zero(tangent)
        if is_active
        else jnp.zeros((), dtype=primal.dtype)
        for primal, tangent, is_active in zip((x, sh, weights), tangents[:3], active)
    )
    return (
        fwd_p.bind(*primals, plan=plan),
        fwd_jvp_p.bind(
            x,
            sh,
            weights,
            tx,
            tsh,
            tw,
            senders,
            receivers,
            row_ptr,
            plan=plan,
            active=active,
        ),
    )


ad.primitive_jvps[fwd_p] = _fwd_jvp_rule


def _fwd_jvp_jvp_rule(primals, tangents, *, plan, active):
    """Differentiate the trilinear projected forward-JVP exactly.

    For trilinear ``F``, the first JVP is the sum of ``F(tx, sh, w)``,
    ``F(x, tsh, w)``, and ``F(x, sh, tw)``. Differentiating each of its three
    operands gives a 3-by-3 product rule of generated forward-JVP calls.
    """
    x, sh, weights, tx, tsh, tw, senders, receivers, row_ptr = primals
    dx, dsh, dw, dtx, dtsh, dtw, _, _, _ = tangents

    # ``active`` describes (tx, tsh, tw) in the inner JVP. ``q_active`` marks
    # outer derivatives of (x, sh, weights), while ``t_active`` marks outer
    # derivatives of those inner tangents. Classify them before materialising
    # zeros: inactive scalar placeholders must never reach a generated kernel.
    q_active = tuple(type(tangent) is not ad.Zero for tangent in (dx, dsh, dw))
    t_active = tuple(
        requested and type(tangent) is not ad.Zero
        for requested, tangent in zip(active, (dtx, dtsh, dtw))
    )
    zero_x, zero_sh, zero_w = (
        jnp.zeros_like(x),
        jnp.zeros_like(sh),
        jnp.zeros_like(weights),
    )
    tx = _materialize_symbolic_zero(tx) if active[0] else zero_x
    tsh = _materialize_symbolic_zero(tsh) if active[1] else zero_sh
    tw = _materialize_symbolic_zero(tw) if active[2] else zero_w
    dx = _materialize_symbolic_zero(dx) if q_active[0] else zero_x
    dsh = _materialize_symbolic_zero(dsh) if q_active[1] else zero_sh
    dw = _materialize_symbolic_zero(dw) if q_active[2] else zero_w
    dtx = _materialize_symbolic_zero(dtx) if t_active[0] else zero_x
    dtsh = _materialize_symbolic_zero(dtsh) if t_active[1] else zero_sh
    dtw = _materialize_symbolic_zero(dtw) if t_active[2] else zero_w

    # Rows select the tangent operand in the first JVP. Columns differentiate
    # one operand of that trilinear summand. Row-major order defines the
    # generated product-rule summand order.
    base_operands = (x, sh, weights)
    tangent_operands = (tx, tsh, tw)
    primal_directions = (dx, dsh, dw)
    tangent_directions = (dtx, dtsh, dtw)
    zero_directions = (zero_x, zero_sh, zero_w)
    product_rule = []
    for tangent_axis in range(3):
        values = list(base_operands)
        values[tangent_axis] = tangent_operands[tangent_axis]
        for derivative_axis in range(3):
            directions = list(zero_directions)
            if derivative_axis == tangent_axis:
                enabled = t_active[tangent_axis]
                directions[derivative_axis] = tangent_directions[derivative_axis]
            else:
                enabled = active[tangent_axis] and q_active[derivative_axis]
                directions[derivative_axis] = primal_directions[derivative_axis]
            term_active = tuple(axis == derivative_axis for axis in range(3))
            product_rule.append(
                (enabled, tuple(values), tuple(directions), term_active)
            )
    terms = [
        fwd_jvp_p.bind(
            *values,
            *directions,
            senders,
            receivers,
            row_ptr,
            plan=plan,
            active=term_active,
        )
        for enabled, values, directions, term_active in product_rule
        if enabled
    ]
    tangent = (
        sum(terms[1:], terms[0])
        if terms
        else jnp.zeros((x.shape[0], plan.output_dim), dtype=x.dtype)
    )
    return (
        fwd_jvp_p.bind(
            x,
            sh,
            weights,
            tx,
            tsh,
            tw,
            senders,
            receivers,
            row_ptr,
            plan=plan,
            active=active,
        ),
        tangent,
    )


ad.primitive_jvps[fwd_jvp_p] = _fwd_jvp_jvp_rule


def _fwd_jvp_transpose(
    ct, x, sh, weights, tx, tsh, tw, senders, receivers, row_ptr, *, plan, active
):
    if any(ad.is_undefined_primal(value) for value in (x, sh, weights)):
        raise NotImplementedError(
            "factorized_projected_fwd_jvp transpose requires defined coefficients"
        )
    undefined = tuple(ad.is_undefined_primal(value) for value in (tx, tsh, tw))
    if any(
        is_undefined and not requested
        for is_undefined, requested in zip(undefined, active)
    ):
        raise NotImplementedError(
            "inactive factorized_projected_fwd_jvp directions must be defined"
        )
    x, sh, weights, ct = map(_materialize_symbolic_zero, (x, sh, weights, ct))
    if active == (False, False, True):
        grads = (
            None,
            None,
            weight_bwd_p.bind(
                x, sh, weights, ct, senders, receivers, row_ptr, plan=plan
            ),
        )
    else:
        grads = bwd_p.bind(x, sh, weights, ct, senders, receivers, row_ptr, plan=plan)
    tangent_grads = tuple(
        grad if is_undefined else None for grad, is_undefined in zip(grads, undefined)
    )
    return (None, None, None, *tangent_grads, None, None, None)


ad.primitive_transposes[fwd_jvp_p] = _fwd_jvp_transpose


def _bwd_jvp_rule(primals, tangents, *, plan):
    x, sh, weights, dout, senders, receivers, row_ptr = primals
    active = tuple(type(tangent) is not ad.Zero for tangent in tangents[:4])
    tx, tsh, tw, tdout = (
        _materialize_symbolic_zero(tangent)
        if is_active
        else jnp.zeros((), dtype=primal.dtype)
        for primal, tangent, is_active in zip(
            (x, sh, weights, dout), tangents[:4], active
        )
    )
    return (
        bwd_p.bind(*primals, plan=plan),
        dbwd_p.bind(
            x,
            sh,
            weights,
            dout,
            tx,
            tsh,
            tw,
            tdout,
            senders,
            receivers,
            row_ptr,
            plan=plan,
            active=active,
        ),
    )


ad.primitive_jvps[bwd_p] = _bwd_jvp_rule


def _weight_bwd_jvp_rule(primals, tangents, *, plan):
    x, sh, weights, dout, senders, receivers, row_ptr = primals
    active = tuple(type(tangent) is not ad.Zero for tangent in tangents[:4])
    tx, tsh, tw, tdout = (
        _materialize_symbolic_zero(tangent)
        if requested
        else jnp.zeros((), dtype=primal.dtype)
        for primal, tangent, requested in zip(
            (x, sh, weights, dout), tangents[:4], active
        )
    )
    primal = weight_bwd_p.bind(*primals, plan=plan)
    tangent = dbwd_p.bind(
        x,
        sh,
        weights,
        dout,
        tx,
        tsh,
        tw,
        tdout,
        senders,
        receivers,
        row_ptr,
        plan=plan,
        active=active,
    )[2]
    return primal, tangent


ad.primitive_jvps[weight_bwd_p] = _weight_bwd_jvp_rule


def _bwd_transpose(
    cotangents, x, sh, weights, dout, senders, receivers, row_ptr, *, plan
):
    """Transpose one differentiated input of the projected backward.

    This is the Hessian symmetry identity for ``B_q = grad_q <F, dout>``.
    Higher-order transforms only need one unknown primal at a time. Rejecting
    wider requests avoids pretending that an incomplete custom rule handles a
    general reverse sweep.
    """
    primals = (x, sh, weights, dout)
    undefined = tuple(ad.is_undefined_primal(value) for value in primals)
    if sum(undefined) != 1:
        raise NotImplementedError(
            "factorized_projected_q_bwd transpose requires one undefined primal"
        )
    x, sh, weights, dout = map(_materialize_symbolic_zero, primals)
    cx, csh, cw = map(_materialize_symbolic_zero, cotangents)
    q_grads = dbwd_p.bind(
        x,
        sh,
        weights,
        dout,
        cx,
        csh,
        cw,
        jnp.zeros_like(dout),
        senders,
        receivers,
        row_ptr,
        plan=plan,
        active=(True, True, True, False),
    )
    dout_grad = fwd_jvp_p.bind(
        x,
        sh,
        weights,
        cx,
        csh,
        cw,
        senders,
        receivers,
        row_ptr,
        plan=plan,
        active=(True, True, True),
    )
    grads = (*q_grads, dout_grad)
    return (
        *(grad if requested else None for grad, requested in zip(grads, undefined)),
        None,
        None,
        None,
    )


ad.primitive_transposes[bwd_p] = _bwd_transpose


def _dbwd_jvp_rule(primals, tangents, *, plan, active):
    """Differentiate ``DB_q(z)[v]`` using its multilinear cross terms."""
    x, sh, weights, dout, tx, tsh, tw, tdout, senders, receivers, row_ptr = primals
    dx, dsh, dw, ddout, dtx, dtsh, dtw, dtdout, _, _, _ = tangents
    z = (x, sh, weights, dout)
    dz = (dx, dsh, dw, ddout)
    v = (tx, tsh, tw, tdout)
    dv = (dtx, dtsh, dtw, dtdout)
    z_active = tuple(type(tangent) is not ad.Zero for tangent in dz)
    dv_active = tuple(
        requested and type(tangent) is not ad.Zero
        for requested, tangent in zip(active, dv)
    )
    zero_v = tuple(jnp.zeros((), dtype=value.dtype) for value in z)
    primal = dbwd_p.bind(*primals, plan=plan, active=active)
    if any(dv_active):
        dv_term = dbwd_p.bind(
            x,
            sh,
            weights,
            dout,
            *(
                _materialize_symbolic_zero(tangent) if requested else zero
                for tangent, requested, zero in zip(dv, dv_active, zero_v)
            ),
            senders,
            receivers,
            row_ptr,
            plan=plan,
            active=dv_active,
        )
    else:
        dv_term = tuple(jnp.zeros_like(value) for value in primal)

    tangent = list(dv_term)
    for i, (zi, dzi) in enumerate(zip(z, dz)):
        if not z_active[i]:
            continue
        for j, vj in enumerate(v):
            if i == j or not active[j]:
                continue
            cross_inputs = list(z)
            cross_inputs[i] = _materialize_symbolic_zero(dzi)
            cross_inputs[j] = _materialize_symbolic_zero(vj)
            cross = bwd_p.bind(*cross_inputs, senders, receivers, row_ptr, plan=plan)
            for k, value in enumerate(cross):
                if k != i and k != j:
                    tangent[k] = tangent[k] + value
    return primal, tuple(tangent)


ad.primitive_jvps[dbwd_p] = _dbwd_jvp_rule


def _dbwd_transpose(
    cotangents,
    x,
    sh,
    weights,
    dout,
    tx,
    tsh,
    tweights,
    tdout,
    senders,
    receivers,
    row_ptr,
    *,
    plan,
    active,
):
    """Transpose the mixed derivative using Hessian symmetry.

    The spatial backward is the gradient of ``vdot(F, dout)`` with respect
    to ``(x, sh, weights)``.  Its derivative is therefore symmetric in those
    three tangent slots.  The remaining ``dout`` cotangent is the forward JVP
    in the received cotangent direction.  This closes reverse-over-forward
    force-loss differentiation without a materialized reference expansion.
    """
    if any(ad.is_undefined_primal(value) for value in (x, sh, weights, dout)):
        raise NotImplementedError(
            "factorized_projected_q_dbwd transpose requires defined coefficients"
        )
    undefined = tuple(
        ad.is_undefined_primal(value) for value in (tx, tsh, tweights, tdout)
    )
    if any(
        is_undefined and not requested
        for is_undefined, requested in zip(undefined, active)
    ):
        raise NotImplementedError(
            "inactive factorized_projected_q_dbwd directions must be defined"
        )
    x, sh, weights, dout = map(_materialize_symbolic_zero, (x, sh, weights, dout))
    cx, csh, cw = map(_materialize_symbolic_zero, cotangents)
    zero_dout = jnp.zeros_like(dout)
    q_grads = dbwd_p.bind(
        x,
        sh,
        weights,
        dout,
        cx,
        csh,
        cw,
        zero_dout,
        senders,
        receivers,
        row_ptr,
        plan=plan,
        active=(True, True, True, False),
    )
    dout_grad = fwd_jvp_p.bind(
        x,
        sh,
        weights,
        cx,
        csh,
        cw,
        senders,
        receivers,
        row_ptr,
        plan=plan,
        active=(True, True, True),
    )
    tangent_grads = (q_grads[0], q_grads[1], q_grads[2], dout_grad)
    tangent_grads = tuple(
        value if is_undefined else None
        for value, is_undefined in zip(tangent_grads, undefined)
    )
    return (
        None,
        None,
        None,
        None,
        *tangent_grads,
        None,
        None,
        None,
    )


ad.primitive_transposes[dbwd_p] = _dbwd_transpose


def factorized_projected(plan, x, sh, weights, senders, receivers, row_ptr):
    return fwd_p.bind(x, sh, weights, senders, receivers, row_ptr, plan=plan)


__all__ = ["factorized_projected"]
