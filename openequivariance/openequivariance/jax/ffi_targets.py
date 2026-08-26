"""Names of JAX FFI handlers provided by the native extension.

Target names follow ``<family>_<operation>[_<transform>][_<operands>]``:

* ``family`` identifies the generated kernel family.
* ``operation`` is ``forward`` or ``backward``.
* ``transform`` records an additional AD operation such as ``jvp``, ``hvp``,
  or ``transpose``.
* ``operands`` identifies differentiated values: ``x`` for features and ``w``
  for weights.

The ``species`` suffix marks symmetric kernels whose weight block is selected
at run time from a per-node species index.  The string value is the exact
symbol registered by the native extension. The upper-case Python name groups
that symbol with the primitive which calls it.
"""

TENSOR_PRODUCT_TARGETS = (
    "tp_forward",
    "tp_backward",
    "tp_double_backward",
)

CONVOLUTION_TARGETS = (
    "conv_forward",
    "conv_backward",
    "conv_double_backward",
)

FACTORIZED_FORWARD = "factorized_projected_forward"
FACTORIZED_FORWARD_JVP = "factorized_projected_forward_jvp"
FACTORIZED_SPATIAL_BACKWARD = "factorized_projected_spatial_backward"
FACTORIZED_WEIGHT_BACKWARD = "factorized_projected_weight_backward"
FACTORIZED_SPATIAL_BACKWARD_JVP = "factorized_projected_spatial_backward_jvp"

FACTORIZED_TARGETS = (
    FACTORIZED_FORWARD,
    FACTORIZED_FORWARD_JVP,
    FACTORIZED_SPATIAL_BACKWARD,
    FACTORIZED_WEIGHT_BACKWARD,
    FACTORIZED_SPATIAL_BACKWARD_JVP,
)

SYMMETRIC_FORWARD = "symmetric_literal_forward_species"
SYMMETRIC_FORWARD_JVP = "symmetric_literal_forward_jvp_x_species"
SYMMETRIC_BACKWARD = "symmetric_literal_backward_x_species"
SYMMETRIC_BACKWARD_JVP = "symmetric_literal_backward_jvp_x_species"
SYMMETRIC_BACKWARD_HVP = "symmetric_literal_backward_hvp_x_species"
SYMMETRIC_MIXED_JVP = "symmetric_literal_backward_jvp_xw_species"
SYMMETRIC_MIXED_TRANSPOSE = "symmetric_literal_backward_jvp_xw_transpose_species"

# Stable integer ABI for the two canonical rank-three symmetric feature
# layouts. Native handlers use it to validate the trailing dimensions selected
# by the generated source whenever the channel and feature dimensions differ.
SYMMETRIC_LAYOUT_CHANNEL_FEATURE = 0  # [nodes, channels, feature_dim]
SYMMETRIC_LAYOUT_FEATURE_CHANNEL = 1  # [nodes, feature_dim, channels]


def symmetric_feature_layout_abi_value(feature_layout: str) -> int:
    """Encode a public symmetric plan layout for the native FFI ABI."""
    if feature_layout == "channel_feature":
        return SYMMETRIC_LAYOUT_CHANNEL_FEATURE
    if feature_layout == "feature_channel":
        return SYMMETRIC_LAYOUT_FEATURE_CHANNEL
    raise ValueError(f"unsupported symmetric feature layout: {feature_layout!r}")

SYMMETRIC_TARGETS = (
    SYMMETRIC_FORWARD,
    SYMMETRIC_FORWARD_JVP,
    SYMMETRIC_BACKWARD,
    SYMMETRIC_BACKWARD_JVP,
    SYMMETRIC_BACKWARD_HVP,
    SYMMETRIC_MIXED_JVP,
    SYMMETRIC_MIXED_TRANSPOSE,
)

FFI_TARGETS = (
    TENSOR_PRODUCT_TARGETS
    + CONVOLUTION_TARGETS
    + FACTORIZED_TARGETS
    + SYMMETRIC_TARGETS
)
