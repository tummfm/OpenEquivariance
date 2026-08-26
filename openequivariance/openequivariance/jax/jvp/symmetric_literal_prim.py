"""JAX transformations for literal symmetric-contraction kernels.

The primal is always the lean literal energy kernel. JAX tangent activity then
selects feature-only kernels when weights are constant, mixed literal kernels
when supported, or ordinary JAX reference pullbacks for the remaining cases.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend import core
from jax.interpreters import ad, mlir

from openequivariance.jax import extlib
from openequivariance.jax.ffi_targets import (
    SYMMETRIC_BACKWARD,
    SYMMETRIC_BACKWARD_HVP,
    SYMMETRIC_BACKWARD_JVP,
    SYMMETRIC_FORWARD,
    SYMMETRIC_FORWARD_JVP,
    SYMMETRIC_MIXED_JVP,
    SYMMETRIC_MIXED_TRANSPOSE,
)
from openequivariance.jax.symmetric_literal_codegen import (
    generate_symmetric_literal_source,
)
from openequivariance.jax.symmetric_literal import (
    evaluate_symmetric_plan,
    _materialize_symbolic_zero,
    _plan_attributes,
    _reference_backward,
    _validate,
)


def _source(plan, x):
    return generate_symmetric_literal_source(
        plan, dtype=np.dtype(x.dtype), is_hip=extlib.IS_HIP
    )


def _is_zero(tangent):
    return type(tangent) is ad.Zero


def _reference_dx(plan, x, species, weights, dout):
    return _reference_backward(plan, x, species, weights, dout, attributes=False)[0]


literal_fwd_species_p = core.Primitive("symmetric_literal_fwd_species")
literal_fwd_jvp_x_species_p = core.Primitive("symmetric_literal_fwd_jvp_x_species")
literal_bwd_x_species_p = core.Primitive("symmetric_literal_bwd_x_species")
literal_bwd_jvp_x_species_p = core.Primitive("symmetric_literal_bwd_jvp_x_species")
literal_bwd_hvp_x_species_p = core.Primitive("symmetric_literal_bwd_hvp_x_species")
literal_bwd_jvp_xw_species_p = core.Primitive("symmetric_literal_bwd_jvp_xw_species")
literal_bwd_jvp_xw_transpose_species_p = core.Primitive(
    "symmetric_literal_bwd_jvp_xw_transpose_species"
)
literal_bwd_jvp_xw_transpose_species_p.multiple_results = True


def _fwd_impl(x, species, weights, *, plan):
    _validate(plan, x, species, weights, attributes=False)
    generated = _source(plan, x)
    shape = jax.ShapeDtypeStruct((x.shape[0], plan.output_dim), x.dtype)
    return jax.ffi.ffi_call(SYMMETRIC_FORWARD, shape)(
        x,
        species,
        weights,
        source=generated.source,
        hash=generated.source_hash,
        **_plan_attributes(plan),
    )


def _fwd_abstract(x, species, weights, *, plan):
    _validate(plan, x, species, weights, attributes=False)
    return jax.core.ShapedArray((x.shape[0], plan.output_dim), x.dtype)


literal_fwd_species_p.def_impl(_fwd_impl)
literal_fwd_species_p.def_abstract_eval(_fwd_abstract)
mlir.register_lowering(
    literal_fwd_species_p,
    mlir.lower_fun(_fwd_impl, multiple_results=False),
    platform="cuda",
)
mlir.register_lowering(
    literal_fwd_species_p,
    mlir.lower_fun(_fwd_impl, multiple_results=False),
    platform="rocm",
)


def _fwd_jvp_x_impl(x, species, weights, tx, *, plan):
    _validate(plan, x, species, weights, attributes=False)
    generated = _source(plan, x)
    shape = jax.ShapeDtypeStruct((x.shape[0], plan.output_dim), x.dtype)
    return jax.ffi.ffi_call(SYMMETRIC_FORWARD_JVP, shape)(
        x,
        species,
        weights,
        tx,
        source=generated.source,
        hash=generated.source_hash,
        **_plan_attributes(plan),
    )


def _fwd_jvp_x_abstract(x, species, weights, tx, *, plan):
    _validate(plan, x, species, weights, attributes=False)
    if tx.shape != x.shape or tx.dtype != x.dtype:
        raise ValueError("tx must match x")
    return jax.core.ShapedArray((x.shape[0], plan.output_dim), x.dtype)


literal_fwd_jvp_x_species_p.def_impl(_fwd_jvp_x_impl)
literal_fwd_jvp_x_species_p.def_abstract_eval(_fwd_jvp_x_abstract)
mlir.register_lowering(
    literal_fwd_jvp_x_species_p,
    mlir.lower_fun(_fwd_jvp_x_impl, multiple_results=False),
    platform="cuda",
)
mlir.register_lowering(
    literal_fwd_jvp_x_species_p,
    mlir.lower_fun(_fwd_jvp_x_impl, multiple_results=False),
    platform="rocm",
)


def _bwd_x_impl(x, species, weights, dout, *, plan):
    _validate(plan, x, species, weights, attributes=False, dout=dout)
    generated = _source(plan, x)
    shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return jax.ffi.ffi_call(SYMMETRIC_BACKWARD, shape)(
        x,
        species,
        weights,
        dout,
        source=generated.source,
        hash=generated.source_hash,
        **_plan_attributes(plan),
    )


def _bwd_x_abstract(x, species, weights, dout, *, plan):
    _validate(plan, x, species, weights, attributes=False, dout=dout)
    return jax.core.ShapedArray(x.shape, x.dtype)


literal_bwd_x_species_p.def_impl(_bwd_x_impl)
literal_bwd_x_species_p.def_abstract_eval(_bwd_x_abstract)
mlir.register_lowering(
    literal_bwd_x_species_p,
    mlir.lower_fun(_bwd_x_impl, multiple_results=False),
    platform="cuda",
)
mlir.register_lowering(
    literal_bwd_x_species_p,
    mlir.lower_fun(_bwd_x_impl, multiple_results=False),
    platform="rocm",
)


def _bwd_jvp_x_impl(x, species, weights, dout, tx, tdout, *, plan):
    _validate(plan, x, species, weights, attributes=False, dout=dout)
    generated = _source(plan, x)
    shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return jax.ffi.ffi_call(SYMMETRIC_BACKWARD_JVP, shape)(
        x,
        species,
        weights,
        dout,
        tx,
        tdout,
        source=generated.source,
        hash=generated.source_hash,
        **_plan_attributes(plan),
    )


def _bwd_jvp_x_abstract(x, species, weights, dout, tx, tdout, *, plan):
    _validate(plan, x, species, weights, attributes=False, dout=dout)
    if tx.shape != x.shape or tx.dtype != x.dtype:
        raise ValueError("tx must match x")
    if tdout.shape != dout.shape or tdout.dtype != dout.dtype:
        raise ValueError("tdout must match dout")
    return jax.core.ShapedArray(x.shape, x.dtype)


literal_bwd_jvp_x_species_p.def_impl(_bwd_jvp_x_impl)
literal_bwd_jvp_x_species_p.def_abstract_eval(_bwd_jvp_x_abstract)
mlir.register_lowering(
    literal_bwd_jvp_x_species_p,
    mlir.lower_fun(_bwd_jvp_x_impl, multiple_results=False),
    platform="cuda",
)
mlir.register_lowering(
    literal_bwd_jvp_x_species_p,
    mlir.lower_fun(_bwd_jvp_x_impl, multiple_results=False),
    platform="rocm",
)


def _bwd_hvp_x_impl(x, species, weights, dout, tx, *, plan):
    _validate(plan, x, species, weights, attributes=False, dout=dout)
    generated = _source(plan, x)
    shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return jax.ffi.ffi_call(SYMMETRIC_BACKWARD_HVP, shape)(
        x,
        species,
        weights,
        dout,
        tx,
        source=generated.source,
        hash=generated.source_hash,
        **_plan_attributes(plan),
    )


def _bwd_hvp_x_abstract(x, species, weights, dout, tx, *, plan):
    _validate(plan, x, species, weights, attributes=False, dout=dout)
    if tx.shape != x.shape or tx.dtype != x.dtype:
        raise ValueError("tx must match x")
    return jax.core.ShapedArray(x.shape, x.dtype)


literal_bwd_hvp_x_species_p.def_impl(_bwd_hvp_x_impl)
literal_bwd_hvp_x_species_p.def_abstract_eval(_bwd_hvp_x_abstract)
mlir.register_lowering(
    literal_bwd_hvp_x_species_p,
    mlir.lower_fun(_bwd_hvp_x_impl, multiple_results=False),
    platform="cuda",
)
mlir.register_lowering(
    literal_bwd_hvp_x_species_p,
    mlir.lower_fun(_bwd_hvp_x_impl, multiple_results=False),
    platform="rocm",
)


def _bwd_jvp_xw_impl(x, species, weights, dout, tx, tweights, tdout, *, plan):
    _validate(plan, x, species, weights, attributes=False, dout=dout)
    generated = _source(plan, x)
    shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return jax.ffi.ffi_call(SYMMETRIC_MIXED_JVP, shape)(
        x,
        species,
        weights,
        dout,
        tx,
        tweights,
        tdout,
        source=generated.source,
        hash=generated.source_hash,
        **_plan_attributes(plan),
    )


def _bwd_jvp_xw_abstract(x, species, weights, dout, tx, tweights, tdout, *, plan):
    _validate(plan, x, species, weights, attributes=False, dout=dout)
    if tx.shape != x.shape or tx.dtype != x.dtype:
        raise ValueError("tx must match x")
    if tweights.shape != weights.shape or tweights.dtype != weights.dtype:
        raise ValueError("tweights must match weights")
    if tdout.shape != dout.shape or tdout.dtype != dout.dtype:
        raise ValueError("tdout must match dout")
    return jax.core.ShapedArray(x.shape, x.dtype)


literal_bwd_jvp_xw_species_p.def_impl(_bwd_jvp_xw_impl)
literal_bwd_jvp_xw_species_p.def_abstract_eval(_bwd_jvp_xw_abstract)
mlir.register_lowering(
    literal_bwd_jvp_xw_species_p,
    mlir.lower_fun(_bwd_jvp_xw_impl, multiple_results=False),
    platform="cuda",
)
mlir.register_lowering(
    literal_bwd_jvp_xw_species_p,
    mlir.lower_fun(_bwd_jvp_xw_impl, multiple_results=False),
    platform="rocm",
)


def _bwd_jvp_xw_transpose_impl(x, species, weights, dout, ctdx, *, plan):
    _validate(plan, x, species, weights, attributes=False, dout=dout)
    if ctdx.shape != x.shape or ctdx.dtype != x.dtype:
        raise ValueError("ctdx must match x")
    generated = _source(plan, x)
    shapes = tuple(
        jax.ShapeDtypeStruct(value.shape, value.dtype) for value in (x, weights, dout)
    )
    return jax.ffi.ffi_call(SYMMETRIC_MIXED_TRANSPOSE, shapes)(
        x,
        species,
        weights,
        dout,
        ctdx,
        source=generated.source,
        hash=generated.source_hash,
        **_plan_attributes(plan),
    )


def _bwd_jvp_xw_transpose_abstract(x, species, weights, dout, ctdx, *, plan):
    _validate(plan, x, species, weights, attributes=False, dout=dout)
    if ctdx.shape != x.shape or ctdx.dtype != x.dtype:
        raise ValueError("ctdx must match x")
    return tuple(
        jax.core.ShapedArray(value.shape, value.dtype) for value in (x, weights, dout)
    )


literal_bwd_jvp_xw_transpose_species_p.def_impl(_bwd_jvp_xw_transpose_impl)
literal_bwd_jvp_xw_transpose_species_p.def_abstract_eval(_bwd_jvp_xw_transpose_abstract)
mlir.register_lowering(
    literal_bwd_jvp_xw_transpose_species_p,
    mlir.lower_fun(_bwd_jvp_xw_transpose_impl, multiple_results=True),
    platform="cuda",
)
mlir.register_lowering(
    literal_bwd_jvp_xw_transpose_species_p,
    mlir.lower_fun(_bwd_jvp_xw_transpose_impl, multiple_results=True),
    platform="rocm",
)


def _fwd_jvp_rule(primals, tangents, *, plan):
    x, species, weights = primals
    tx, _, tweights = tangents
    primal = literal_fwd_species_p.bind(*primals, plan=plan)
    if _is_zero(tweights):
        tangent = literal_fwd_jvp_x_species_p.bind(
            x, species, weights, _materialize_symbolic_zero(tx), plan=plan
        )
    else:
        _, tangent = jax.jvp(
            lambda a, w: evaluate_symmetric_plan(plan, a, species, w),
            (x, weights),
            (
                _materialize_symbolic_zero(tx),
                _materialize_symbolic_zero(tweights),
            ),
        )
    return primal, tangent


ad.primitive_jvps[literal_fwd_species_p] = _fwd_jvp_rule


def _fwd_jvp_x_transpose(ct, x, species, weights, tx, *, plan):
    x, weights, ct = map(_materialize_symbolic_zero, (x, weights, ct))
    dx = literal_bwd_x_species_p.bind(x, species, weights, ct, plan=plan)
    return None, None, None, dx


ad.primitive_transposes[literal_fwd_jvp_x_species_p] = _fwd_jvp_x_transpose


def _bwd_x_jvp_rule(primals, tangents, *, plan):
    x, species, weights, dout = primals
    tx, _, tweights, tdout = tangents
    primal = literal_bwd_x_species_p.bind(*primals, plan=plan)
    if _is_zero(tweights):
        if _is_zero(tdout):
            tangent = literal_bwd_hvp_x_species_p.bind(
                x,
                species,
                weights,
                dout,
                _materialize_symbolic_zero(tx),
                plan=plan,
            )
        else:
            tangent = literal_bwd_jvp_x_species_p.bind(
                x,
                species,
                weights,
                dout,
                _materialize_symbolic_zero(tx),
                _materialize_symbolic_zero(tdout),
                plan=plan,
            )
    else:
        tangent = literal_bwd_jvp_xw_species_p.bind(
            x,
            species,
            weights,
            dout,
            _materialize_symbolic_zero(tx),
            _materialize_symbolic_zero(tweights),
            _materialize_symbolic_zero(tdout),
            plan=plan,
        )
    return primal, tangent


ad.primitive_jvps[literal_bwd_x_species_p] = _bwd_x_jvp_rule


def _higher_fwd_jvp_x(primals, tangents, *, plan):
    x, species, weights, tx = primals
    values = (x, weights, tx)
    tangent_values = tuple(
        map(
            _materialize_symbolic_zero,
            (tangents[0], tangents[2], tangents[3]),
        )
    )

    def function(a, w, direction):
        return jax.jvp(
            lambda value: evaluate_symmetric_plan(plan, value, species, w),
            (a,),
            (direction,),
        )[1]

    return jax.jvp(function, values, tangent_values)


ad.primitive_jvps[literal_fwd_jvp_x_species_p] = _higher_fwd_jvp_x


def _higher_bwd_jvp_x(primals, tangents, *, plan):
    x, species, weights, dout, tx, tdout = primals
    values = (x, weights, dout, tx, tdout)
    tangent_values = tuple(
        map(
            _materialize_symbolic_zero,
            (
                tangents[0],
                tangents[2],
                tangents[3],
                tangents[4],
                tangents[5],
            ),
        )
    )

    def function(a, w, d, direction, tdirection):
        return jax.jvp(
            lambda value, cotangent: _reference_dx(plan, value, species, w, cotangent),
            (a, d),
            (direction, tdirection),
        )[1]

    return jax.jvp(function, values, tangent_values)


ad.primitive_jvps[literal_bwd_jvp_x_species_p] = _higher_bwd_jvp_x


def _higher_bwd_hvp_x(primals, tangents, *, plan):
    x, species, weights, dout, tx = primals
    values = (x, weights, dout, tx)
    tangent_values = tuple(
        map(
            _materialize_symbolic_zero,
            (tangents[0], tangents[2], tangents[3], tangents[4]),
        )
    )

    def function(a, w, d, direction):
        return jax.jvp(
            lambda value: _reference_dx(plan, value, species, w, d),
            (a,),
            (direction,),
        )[1]

    return jax.jvp(function, values, tangent_values)


ad.primitive_jvps[literal_bwd_hvp_x_species_p] = _higher_bwd_hvp_x


def _higher_bwd_jvp_xw(primals, tangents, *, plan):
    x, species, weights, dout, tx, tweights, tdout = primals
    primal = literal_bwd_jvp_xw_species_p.bind(*primals, plan=plan)
    if all(_is_zero(tangent) for tangent in tangents[:4]):
        # This operator is linear in its three direction slots.  Keeping that
        # linearization as the same primitive lets reverse mode use the custom
        # transpose below instead of materializing the polynomial in HLO.
        tangent = literal_bwd_jvp_xw_species_p.bind(
            x,
            species,
            weights,
            dout,
            _materialize_symbolic_zero(tangents[4]),
            _materialize_symbolic_zero(tangents[5]),
            _materialize_symbolic_zero(tangents[6]),
            plan=plan,
        )
    else:
        values = (x, weights, dout, tx, tweights, tdout)
        tangent_values = tuple(
            map(
                _materialize_symbolic_zero,
                (
                    tangents[0],
                    tangents[2],
                    tangents[3],
                    tangents[4],
                    tangents[5],
                    tangents[6],
                ),
            )
        )

        def function(a, w, d, direction, weight_direction, dout_direction):
            return jax.jvp(
                lambda value, table, cotangent: _reference_dx(
                    plan, value, species, table, cotangent
                ),
                (a, w, d),
                (direction, weight_direction, dout_direction),
            )[1]

        tangent = jax.jvp(function, values, tangent_values)[1]
    return primal, tangent


ad.primitive_jvps[literal_bwd_jvp_xw_species_p] = _higher_bwd_jvp_xw


def _bwd_jvp_xw_transpose(ct, x, species, weights, dout, tx, tweights, tdout, *, plan):
    if any(ad.is_undefined_primal(value) for value in (x, weights, dout)):
        raise NotImplementedError(
            "the literal mixed transpose only differentiates direction slots"
        )
    x, weights, dout, ct = map(_materialize_symbolic_zero, (x, weights, dout, ct))
    ctx, ctweights, ctdout = literal_bwd_jvp_xw_transpose_species_p.bind(
        x, species, weights, dout, ct, plan=plan
    )
    return None, None, None, None, ctx, ctweights, ctdout


ad.primitive_transposes[literal_bwd_jvp_xw_species_p] = _bwd_jvp_xw_transpose


def _reference_bwd_jvp_xw(plan, x, species, weights, dout, tx, tweights, tdout):
    return jax.jvp(
        lambda value, table, cotangent: _reference_dx(
            plan, value, species, table, cotangent
        ),
        (x, weights, dout),
        (tx, tweights, tdout),
    )[1]


def _reference_bwd_jvp_xw_transpose(plan, x, species, weights, dout, ctdx):
    zeros = (jnp.zeros_like(x), jnp.zeros_like(weights), jnp.zeros_like(dout))
    _, pullback = jax.vjp(
        lambda tx, tw, td: _reference_bwd_jvp_xw(
            plan, x, species, weights, dout, tx, tw, td
        ),
        *zeros,
    )
    return pullback(ctdx)


def _higher_bwd_jvp_xw_transpose(primals, tangents, *, plan):
    x, species, weights, dout, ctdx = primals
    primal = literal_bwd_jvp_xw_transpose_species_p.bind(*primals, plan=plan)
    if all(_is_zero(tangent) for tangent in tangents[:4]):
        tangent = literal_bwd_jvp_xw_transpose_species_p.bind(
            x,
            species,
            weights,
            dout,
            _materialize_symbolic_zero(tangents[4]),
            plan=plan,
        )
    else:
        values = (x, weights, dout, ctdx)
        tangent_values = tuple(
            map(
                _materialize_symbolic_zero,
                (tangents[0], tangents[2], tangents[3], tangents[4]),
            )
        )
        tangent = jax.jvp(
            lambda a, w, d, ct: _reference_bwd_jvp_xw_transpose(
                plan, a, species, w, d, ct
            ),
            values,
            tangent_values,
        )[1]
    return primal, tangent


ad.primitive_jvps[literal_bwd_jvp_xw_transpose_species_p] = _higher_bwd_jvp_xw_transpose


def _bwd_jvp_xw_transpose_transpose(
    cotangents, x, species, weights, dout, ctdx, *, plan
):
    if any(ad.is_undefined_primal(value) for value in (x, weights, dout)):
        raise NotImplementedError(
            "the literal transpose adjoint only differentiates its cotangent slot"
        )
    x, weights, dout = map(_materialize_symbolic_zero, (x, weights, dout))
    tx, tweights, tdout = map(_materialize_symbolic_zero, cotangents)
    return (
        None,
        None,
        None,
        None,
        literal_bwd_jvp_xw_species_p.bind(
            x,
            species,
            weights,
            dout,
            tx,
            tweights,
            tdout,
            plan=plan,
        ),
    )


ad.primitive_transposes[literal_bwd_jvp_xw_transpose_species_p] = (
    _bwd_jvp_xw_transpose_transpose
)


def _bwd_jvp_x_transpose(ct, x, species, weights, dout, tx, tdout, *, plan):
    x, weights, dout = map(_materialize_symbolic_zero, (x, weights, dout))
    zeros = (jnp.zeros_like(x), jnp.zeros_like(dout))

    def linear(directions):
        return jax.jvp(
            lambda value, cotangent: _reference_dx(
                plan, value, species, weights, cotangent
            ),
            (x, dout),
            directions,
        )[1]

    _, pullback = jax.vjp(linear, zeros)
    dtx, dtdout = pullback(_materialize_symbolic_zero(ct))[0]
    return None, None, None, None, dtx, dtdout


ad.primitive_transposes[literal_bwd_jvp_x_species_p] = _bwd_jvp_x_transpose


def _bwd_hvp_x_transpose(ct, x, species, weights, dout, tx, *, plan):
    x, weights, dout = map(_materialize_symbolic_zero, (x, weights, dout))

    def linear(direction):
        return jax.jvp(
            lambda value: _reference_dx(plan, value, species, weights, dout),
            (x,),
            (direction,),
        )[1]

    return (
        None,
        None,
        None,
        None,
        jax.vjp(linear, jnp.zeros_like(x))[1](_materialize_symbolic_zero(ct))[0],
    )


ad.primitive_transposes[literal_bwd_hvp_x_species_p] = _bwd_hvp_x_transpose


def symmetric_literal_generated(plan, x, species, weights):
    """Bind the lean primal. JAX transforms choose the required AD rule."""
    return literal_fwd_species_p.bind(x, species, weights, plan=plan)


__all__ = ["symmetric_literal_generated"]
