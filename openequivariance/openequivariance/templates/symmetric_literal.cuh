{# These macros expand during rendering and add no device-call overhead. #}
{% macro feature_value(component, tangent=False) -%}
{{ "tx" if tangent else "x" }}[feature_index(node, channel, {{ component }})]
{%- endmacro %}
{% macro path_weight(path) -%}
weights[weight_base + {{ path.weight_base }} + channel * {{ path.couplings_per_channel }} + {{ path.coupling_index }}]
{%- endmacro %}
{% macro output_adjoint(path, name="dout") -%}
{{ name }}[node * OUTPUT_DIM + {{ path.output_base }} + channel * {{ path.output_irrep_dim }} + {{ path.output_component }}]
{%- endmacro %}
{# A logical 32-thread group is partitioned into channel cohorts. path_lane is
   a lane's position within its cohort. Each cohort owns one (node, channel).
   This mapping is independent of the native HIP wavefront width. #}
{% macro locate_node_channel() -%}
  int lane = threadIdx.x & 31;
  int path_lane = lane / CHANNELS_PER_COHORT;
  int64_t logical_group =
      (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  int channel_groups =
      (CHANNELS + CHANNELS_PER_COHORT - 1) / CHANNELS_PER_COHORT;
  int64_t node = logical_group / channel_groups;
  int channel = int(logical_group % channel_groups) * CHANNELS_PER_COHORT
      + lane % CHANNELS_PER_COHORT;
  if (node >= nodes || channel >= CHANNELS) return;
  int32_t species_id = species[node];
  if (species_id < 0 || species_id >= num_elements) return;
  int64_t weight_base = int64_t(species_id) * WEIGHT_DIM;
{%- endmacro %}
{% macro path_lane_sum(value) -%}
{{ shfl_xor_32(value, "offset * CHANNELS_PER_COHORT") }}
{%- endmacro %}
using int64_t = signed long long;
using int32_t = int;
using scalar_t = {{ context.scalar }};
constexpr int CHANNELS = {{ context.channels }};
constexpr int FEATURE_DIM = {{ context.feature_dim }};
constexpr int OUTPUT_DIM = {{ context.output_dim }};
constexpr int WEIGHT_DIM = {{ context.weight_dim }};
constexpr int OUTPUT_GROUPS = {{ context.output_groups | length }};
constexpr int MAX_DEGREE = {{ context.max_degree }};
constexpr int CANONICAL_PATHS = {{ context.paths | length }};
// PATH_LANES lanes cooperate on paths for one channel. This kernel assigns one
// logical 32-thread group to 32 channels, so no path-lane reduction is needed.
#define PATH_LANES 1
#define CHANNELS_PER_COHORT (32 / PATH_LANES)

__device__ __forceinline__ int64_t feature_index(
    int64_t node, int channel, int component) {
{% if context.feature_layout == "feature_channel" %}
  return node * FEATURE_DIM * CHANNELS + component * CHANNELS + channel;
{% else %}
  return node * CHANNELS * FEATURE_DIM + channel * FEATURE_DIM + component;
{% endif %}
}

// Rendered path metadata. OUT_START partitions canonical paths by output group.
// FEATURE_COMPONENT is padded to MAX_DEGREE entries per path.
__device__ int OUT_START[] = { {% for value in context.out_starts %}{{ value }}{{ ", " if not loop.last else "" }}{% endfor %} };
__device__ int OUT_BASE[] = { {% for group in context.output_groups %}{{ group.output_base }}{{ ", " if not loop.last else "" }}{% endfor %} };
__device__ int OUT_IRREP_DIM[] = { {% for group in context.output_groups %}{{ group.output_irrep_dim }}{{ ", " if not loop.last else "" }}{% endfor %} };
__device__ int OUT_COMPONENT[] = { {% for group in context.output_groups %}{{ group.output_component }}{{ ", " if not loop.last else "" }}{% endfor %} };
__device__ int DEGREE[] = { {% for value in context.degrees %}{{ value }}{{ ", " if not loop.last else "" }}{% endfor %} };
__device__ int WEIGHT_BASE[] = { {% for path in context.paths %}{{ path.weight_base }}{{ ", " if not loop.last else "" }}{% endfor %} };
__device__ int COUPLINGS_PER_CHANNEL[] = { {% for path in context.paths %}{{ path.couplings_per_channel }}{{ ", " if not loop.last else "" }}{% endfor %} };
__device__ int COUPLING_INDEX[] = { {% for path in context.paths %}{{ path.coupling_index }}{{ ", " if not loop.last else "" }}{% endfor %} };
__device__ int FEATURE_COMPONENT[] = { {% for value in context.feature_components %}{{ value }}{{ ", " if not loop.last else "" }}{% endfor %} };
__device__ scalar_t COEFFICIENT[] = { {% for path in context.paths %}{{ path.coefficient }}{{ ", " if not loop.last else "" }}{% endfor %} };

// Forward: a channel cohort owns (node, channel), and path_lane 0 uniquely
// writes each output group after the cohort reduction.
extern "C" __global__ void oeq_symmetric_literal_forward_species(
    int64_t nodes, int64_t num_elements, const scalar_t* x, const int32_t* species,
    const scalar_t* weights, scalar_t* out) {
{{ locate_node_channel() }}
{% for group in context.output_groups %}
  {
    scalar_t sum = scalar_t(0);
{% for path in group.paths %}
    if ({{ path.index }} % PATH_LANES == path_lane) sum +=
        {{ path.coefficient }} * {{ path_weight(path) }}{% for component in path.feature_components %} * {{ feature_value(component) }}{% endfor %};
{% endfor %}
    for (int offset = 1; offset < PATH_LANES; offset <<= 1)
      sum += {{ path_lane_sum("sum") }};
    if (path_lane == 0)
      out[node * OUTPUT_DIM + {{ group.output_base }}
          + channel * {{ group.output_irrep_dim }} + {{ group.output_component }}] = sum;
  }
{% endfor %}
}

#undef PATH_LANES
// Two lanes split alternate paths for each channel and XOR-reduce their sums.
#define PATH_LANES 2
extern "C" __global__ void oeq_symmetric_literal_forward_jvp_x_species(
    int64_t nodes, int64_t num_elements, const scalar_t* x, const int32_t* species,
    const scalar_t* weights, const scalar_t* tx, scalar_t* tout) {
{{ locate_node_channel() }}
  for (int output_group = 0; output_group < OUTPUT_GROUPS; ++output_group) {
    scalar_t sum = scalar_t(0);
    int path_begin = OUT_START[output_group];
    int path_end = OUT_START[output_group + 1];
    // Product rule: differentiate one feature factor at a time.
    for (int path = path_begin + path_lane; path < path_end; path += PATH_LANES) {
      scalar_t path_weight = COEFFICIENT[path] * weights[
          weight_base + WEIGHT_BASE[path]
          + channel * COUPLINGS_PER_CHANNEL[path] + COUPLING_INDEX[path]];
      int degree = DEGREE[path];
      for (int active = 0; active < degree; ++active) {
        int active_component = FEATURE_COMPONENT[path * MAX_DEGREE + active];
        scalar_t value = path_weight * tx[feature_index(node, channel, active_component)];
        for (int factor = 0; factor < degree; ++factor) {
          if (factor == active) continue;
          int component = FEATURE_COMPONENT[path * MAX_DEGREE + factor];
          value *= x[feature_index(node, channel, component)];
        }
        sum += value;
      }
    }
    for (int offset = 1; offset < PATH_LANES; offset <<= 1)
      sum += {{ path_lane_sum("sum") }};
    if (path_lane == 0)
      tout[node * OUTPUT_DIM + OUT_BASE[output_group]
          + channel * OUT_IRREP_DIM[output_group]
          + OUT_COMPONENT[output_group]] = sum;
  }
}

#undef PATH_LANES
// One lane owns each channel and accumulates every path locally.
#define PATH_LANES 1
// Input reverse pass: each cohort uniquely owns every (node, channel, feature).
extern "C" __global__ void oeq_symmetric_literal_backward_x_species(
    int64_t nodes, int64_t num_elements, const scalar_t* x, const int32_t* species,
    const scalar_t* weights, const scalar_t* dout, scalar_t* dx) {
{{ locate_node_channel() }}
  scalar_t grad[FEATURE_DIM];
#pragma unroll
  for (int component = 0; component < FEATURE_DIM; ++component)
    grad[component] = scalar_t(0);
{% for path in context.paths %}
  if ({{ path.index }} % PATH_LANES == path_lane) {
    scalar_t common = {{ path.coefficient }} * {{ path_weight(path) }} * {{ output_adjoint(path) }};
{% for component in path.feature_components %}
    scalar_t x{{ loop.index0 }} = {{ feature_value(component) }};
{% endfor %}
{% for component in path.feature_components %}
{% set active = loop.index0 %}
    grad[{{ component }}] += common{% for factor in range(path.feature_components | length) if factor != active %} * x{{ factor }}{% endfor %};
{% endfor %}
  }
{% endfor %}
  for (int component = 0; component < FEATURE_DIM; ++component) {
    scalar_t sum = grad[component];
    for (int offset = 1; offset < PATH_LANES; offset <<= 1)
      sum += {{ path_lane_sum("sum") }};
    if (path_lane == 0)
      dx[feature_index(node, channel, component)] = sum;
  }
}

#undef PATH_LANES
// Two lanes split canonical paths for each channel and reduce before storing.
#define PATH_LANES 2
extern "C" __global__ void oeq_symmetric_literal_backward_jvp_x_species(
    int64_t nodes, int64_t num_elements, const scalar_t* x, const int32_t* species,
    const scalar_t* weights, const scalar_t* dout,
    const scalar_t* tx, const scalar_t* tdout, scalar_t* tdx) {
{{ locate_node_channel() }}
  scalar_t grad[FEATURE_DIM];
#pragma unroll
  for (int component = 0; component < FEATURE_DIM; ++component)
    grad[component] = scalar_t(0);
  int output_group = 0;
  for (int path = path_lane; path < CANONICAL_PATHS; path += PATH_LANES) {
    while (path >= OUT_START[output_group + 1]) ++output_group;
    scalar_t path_weight = COEFFICIENT[path] * weights[
        weight_base + WEIGHT_BASE[path]
        + channel * COUPLINGS_PER_CHANNEL[path] + COUPLING_INDEX[path]];
    int output_index = node * OUTPUT_DIM + OUT_BASE[output_group]
        + channel * OUT_IRREP_DIM[output_group] + OUT_COMPONENT[output_group];
    // Output adjoint and its tangent for the explicit degree-one/two/three rule.
    scalar_t d = dout[output_index];
    scalar_t td = tdout[output_index];
    int degree = DEGREE[path];
    int component0 = FEATURE_COMPONENT[path * MAX_DEGREE];
    if (degree == 1) {
      grad[component0] += path_weight * td;
    } else {
      int component1 = FEATURE_COMPONENT[path * MAX_DEGREE + 1];
      scalar_t x0 = x[feature_index(node, channel, component0)];
      scalar_t x1 = x[feature_index(node, channel, component1)];
      scalar_t tx0 = tx[feature_index(node, channel, component0)];
      scalar_t tx1 = tx[feature_index(node, channel, component1)];
      if (degree == 2) {
        grad[component0] += path_weight * (td * x1 + d * tx1);
        grad[component1] += path_weight * (td * x0 + d * tx0);
      } else {
        int component2 = FEATURE_COMPONENT[path * MAX_DEGREE + 2];
        scalar_t x2 = x[feature_index(node, channel, component2)];
        scalar_t tx2 = tx[feature_index(node, channel, component2)];
        grad[component0] += path_weight *
            (td * x1 * x2 + d * (tx1 * x2 + x1 * tx2));
        grad[component1] += path_weight *
            (td * x0 * x2 + d * (tx0 * x2 + x0 * tx2));
        grad[component2] += path_weight *
            (td * x0 * x1 + d * (tx0 * x1 + x0 * tx1));
      }
    }
  }
  for (int component = 0; component < FEATURE_DIM; ++component) {
    scalar_t sum = grad[component];
    for (int offset = 1; offset < PATH_LANES; offset <<= 1)
      sum += {{ path_lane_sum("sum") }};
    if (path_lane == 0) tdx[feature_index(node, channel, component)] = sum;
  }
}

#undef PATH_LANES
// One lane owns each channel. Explicit degree-two/three expressions form the
// feature Hessian-vector product without a path-lane reduction.
#define PATH_LANES 1
extern "C" __global__ void oeq_symmetric_literal_backward_hvp_x_species(
    int64_t nodes, int64_t num_elements, const scalar_t* x, const int32_t* species,
    const scalar_t* weights, const scalar_t* dout,
    const scalar_t* tx, scalar_t* tdx) {
{{ locate_node_channel() }}
  scalar_t grad[FEATURE_DIM];
#pragma unroll
  for (int component = 0; component < FEATURE_DIM; ++component)
    grad[component] = scalar_t(0);
{% for path in context.paths if path.feature_components | length > 1 %}
  if ({{ path.index }} % PATH_LANES == path_lane) {
    scalar_t common = {{ path.coefficient }} * {{ path_weight(path) }} * {{ output_adjoint(path) }};
{% if path.feature_components | length == 2 %}
    grad[{{ path.feature_components[0] }}] += common * {{ feature_value(path.feature_components[1], True) }};
    grad[{{ path.feature_components[1] }}] += common * {{ feature_value(path.feature_components[0], True) }};
{% else %}
    grad[{{ path.feature_components[0] }}] += common *
        ({{ feature_value(path.feature_components[1], True) }} * {{ feature_value(path.feature_components[2]) }}
         + {{ feature_value(path.feature_components[1]) }} * {{ feature_value(path.feature_components[2], True) }});
    grad[{{ path.feature_components[1] }}] += common *
        ({{ feature_value(path.feature_components[0], True) }} * {{ feature_value(path.feature_components[2]) }}
         + {{ feature_value(path.feature_components[0]) }} * {{ feature_value(path.feature_components[2], True) }});
    grad[{{ path.feature_components[2] }}] += common *
        ({{ feature_value(path.feature_components[0], True) }} * {{ feature_value(path.feature_components[1]) }}
         + {{ feature_value(path.feature_components[0]) }} * {{ feature_value(path.feature_components[1], True) }});
{% endif %}
  }
{% endfor %}
  for (int component = 0; component < FEATURE_DIM; ++component) {
    scalar_t sum = grad[component];
    for (int offset = 1; offset < PATH_LANES; offset <<= 1)
      sum += {{ path_lane_sum("sum") }};
    if (path_lane == 0) tdx[feature_index(node, channel, component)] = sum;
  }
}

#undef PATH_LANES
// Two lanes split canonical paths for the full x/weight/output mixed JVP.
#define PATH_LANES 2
extern "C" __global__ void oeq_symmetric_literal_backward_jvp_xw_species(
    int64_t nodes, int64_t num_elements, const scalar_t* x, const int32_t* species,
    const scalar_t* weights, const scalar_t* dout, const scalar_t* tx,
    const scalar_t* tweights, const scalar_t* tdout, scalar_t* tdx) {
{{ locate_node_channel() }}
  scalar_t grad[FEATURE_DIM];
#pragma unroll
  for (int component = 0; component < FEATURE_DIM; ++component)
    grad[component] = scalar_t(0);
  int output_group = 0;
  for (int path = path_lane; path < CANONICAL_PATHS; path += PATH_LANES) {
    while (path >= OUT_START[output_group + 1]) ++output_group;
    int weight_index = weight_base + WEIGHT_BASE[path]
        + channel * COUPLINGS_PER_CHANNEL[path] + COUPLING_INDEX[path];
    scalar_t path_weight = COEFFICIENT[path] * weights[weight_index];
    scalar_t tangent_weight = COEFFICIENT[path] * tweights[weight_index];
    int output_index = node * OUTPUT_DIM + OUT_BASE[output_group]
        + channel * OUT_IRREP_DIM[output_group] + OUT_COMPONENT[output_group];
    scalar_t output_adjoint = dout[output_index];
    scalar_t tangent_output_adjoint = tdout[output_index];
    int degree = DEGREE[path];
    for (int active = 0; active < degree; ++active) {
      int active_component = FEATURE_COMPONENT[path * MAX_DEGREE + active];
      scalar_t product = scalar_t(1);
      scalar_t tproduct = scalar_t(0);
      for (int factor = 0; factor < degree; ++factor) {
        if (factor == active) continue;
        int component = FEATURE_COMPONENT[path * MAX_DEGREE + factor];
        scalar_t product_before_factor = product;
        product *= x[feature_index(node, channel, component)];
        tproduct = tproduct * x[feature_index(node, channel, component)]
            + product_before_factor * tx[feature_index(node, channel, component)];
      }
      grad[active_component] += tangent_weight * output_adjoint * product
          + path_weight * tangent_output_adjoint * product
          + path_weight * output_adjoint * tproduct;
    }
  }
  for (int component = 0; component < FEATURE_DIM; ++component) {
    scalar_t sum = grad[component];
    for (int offset = 1; offset < PATH_LANES; offset <<= 1)
      sum += {{ path_lane_sum("sum") }};
    if (path_lane == 0) tdx[feature_index(node, channel, component)] = sum;
  }
}

// Reset and declare this kernel's path-lane schedule locally.
#undef PATH_LANES
#define PATH_LANES 2
// Transpose of the mixed JVP. Node-local feature/output cotangents reduce within
// a channel cohort. Species-shared weight cotangents require global atomics.
extern "C" __global__ void oeq_symmetric_literal_backward_jvp_xw_transpose_species(
    int64_t nodes, int64_t num_elements, const scalar_t* x, const int32_t* species,
    const scalar_t* weights, const scalar_t* dout, const scalar_t* ctdx,
    scalar_t* ctx, scalar_t* ctweights, scalar_t* ctdout) {
{{ locate_node_channel() }}
  scalar_t grad_x[FEATURE_DIM];
  scalar_t grad_out[OUTPUT_GROUPS];
#pragma unroll
  for (int component = 0; component < FEATURE_DIM; ++component)
    grad_x[component] = scalar_t(0);
#pragma unroll
  for (int group = 0; group < OUTPUT_GROUPS; ++group)
    grad_out[group] = scalar_t(0);
  int output_group = 0;
  for (int path = path_lane; path < CANONICAL_PATHS; path += PATH_LANES) {
    while (path >= OUT_START[output_group + 1]) ++output_group;
    int weight_index = weight_base + WEIGHT_BASE[path]
        + channel * COUPLINGS_PER_CHANNEL[path] + COUPLING_INDEX[path];
    scalar_t coefficient = COEFFICIENT[path];
    scalar_t path_weight = coefficient * weights[weight_index];
    int output_index = node * OUTPUT_DIM + OUT_BASE[output_group]
        + channel * OUT_IRREP_DIM[output_group] + OUT_COMPONENT[output_group];
    scalar_t output_adjoint = dout[output_index];
    int degree = DEGREE[path];
    scalar_t weight_cotangent = scalar_t(0);
    for (int active = 0; active < degree; ++active) {
      int active_component = FEATURE_COMPONENT[path * MAX_DEGREE + active];
      scalar_t input_cotangent =
          ctdx[feature_index(node, channel, active_component)];
      scalar_t product = scalar_t(1);
      for (int factor = 0; factor < degree; ++factor) {
        if (factor == active) continue;
        int component = FEATURE_COMPONENT[path * MAX_DEGREE + factor];
        product *= x[feature_index(node, channel, component)];
      }
      weight_cotangent +=
          coefficient * output_adjoint * input_cotangent * product;
      grad_out[output_group] += path_weight * input_cotangent * product;
      for (int tangent = 0; tangent < degree; ++tangent) {
        if (tangent == active) continue;
        scalar_t mixed_product = scalar_t(1);
        for (int factor = 0; factor < degree; ++factor) {
          if (factor == active || factor == tangent) continue;
          int component = FEATURE_COMPONENT[path * MAX_DEGREE + factor];
          mixed_product *= x[feature_index(node, channel, component)];
        }
        int tangent_component = FEATURE_COMPONENT[path * MAX_DEGREE + tangent];
        grad_x[tangent_component] += path_weight * output_adjoint
            * input_cotangent * mixed_product;
      }
    }
    // Nodes of the same species, and paths sharing a parameter, collide here.
    {{ atomic_add }}(ctweights + weight_index, weight_cotangent);
  }
  for (int component = 0; component < FEATURE_DIM; ++component) {
    scalar_t sum = grad_x[component];
    for (int offset = 1; offset < PATH_LANES; offset <<= 1)
      sum += {{ path_lane_sum("sum") }};
    if (path_lane == 0) ctx[feature_index(node, channel, component)] = sum;
  }
  for (int group = 0; group < OUTPUT_GROUPS; ++group) {
    scalar_t sum = grad_out[group];
    for (int offset = 1; offset < PATH_LANES; offset <<= 1)
      sum += {{ path_lane_sum("sum") }};
    if (path_lane == 0)
      ctdout[node * OUTPUT_DIM + OUT_BASE[group]
          + channel * OUT_IRREP_DIM[group] + OUT_COMPONENT[group]] = sum;
  }
}

#undef PATH_LANES
