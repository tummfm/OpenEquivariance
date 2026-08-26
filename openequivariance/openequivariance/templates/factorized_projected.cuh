// Receiver-owned forward kernels and edge-owned reverse kernels generated from
// sparse coupling paths. Jinja macros below expand into expressions at render
// time. They do not introduce device helper calls.
using int64_t = signed long long;
using int32_t = int;
using scalar_t = {{ scalar }};
using acc_t = {{ scalar }};

constexpr int INPUT_DIM = {{ plan.input_dim }};
constexpr int EDGE_DIM = {{ plan.edge_dim }};
constexpr int OUTPUT_DIM = {{ plan.output_dim }};
constexpr int WEIGHT_DIM = {{ plan.weight_numel }};
constexpr int CHANNEL_DIM = {{ plan.channels }};
constexpr unsigned int FULL_MASK = 0xffffffffu;

// Selectively active tangent reads. Inactive operands render as scalar_t(0).
#define TX(index) ({{ tangent.x }})
#define TSH(index) ({{ tangent.sh }})
#define TW(index) ({{ tangent.weights }})
#define TDOUT(value) ({{ tangent.output }})
#define FTX(index) ({{ forward_tangent.x }})
#define FTSH(index) ({{ forward_tangent.sh }})
#define FTW(index) ({{ forward_tangent.weights }})

{% macro primal_angular(output, channel) -%}
{%- if output.terms -%}
  {%- for term in output.terms -%}
x[sender * INPUT_DIM + {{ term.input_index }}
  + ({{ channel }}) * {{ output.input_irrep_dim }}]
* sh[e * EDGE_DIM + {{ term.edge_index }}]
* {{ term.coefficient }}{{ "+" if not loop.last else "" }}
  {%- endfor -%}
{%- else -%}
scalar_t(0)
{%- endif -%}
{%- endmacro %}

{% macro angular_jvp(output, channel) -%}
{%- if output.terms %}
{%- for term in output.terms %}
FTX(sender * INPUT_DIM + {{ term.input_index }}
    + {{ channel }} * {{ output.input_irrep_dim }})
* sh[e * EDGE_DIM + {{ term.edge_index }}]
* {{ term.coefficient }}{{ "+" if not loop.last else "" }}
{%- endfor %}
+
{%- for term in output.terms %}
x[sender * INPUT_DIM + {{ term.input_index }}
  + {{ channel }} * {{ output.input_irrep_dim }}]
* FTSH(e * EDGE_DIM + {{ term.edge_index }})
* {{ term.coefficient }}{{ "+" if not loop.last else "" }}
{%- endfor %}
{%- else %}
scalar_t(0)
{%- endif %}
{%- endmacro %}

{% macro output_adjoint(output, channel) -%}
dout[receiver * OUTPUT_DIM + {{ output.output_index }}
  + {{ channel }} * {{ output.output_irrep_dim }}]
{%- endmacro %}

// One thread owns (receiver node, channel), scans that receiver's CSR edges,
// and directly writes all of its output components.
extern "C" __global__ void oeq_projected_forward(
    int64_t nodes,
    int64_t edges,
    const scalar_t* x,
    const scalar_t* sh,
    const scalar_t* weights,
    const int32_t* senders,
    const int32_t* row_ptr,
    scalar_t* out) {
  int64_t work = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = nodes * CHANNEL_DIM;
  if (work >= total) return;

  int64_t node = work / CHANNEL_DIM;
  int channel = int(work - node * CHANNEL_DIM);
{% for slot in output_slots %}
  scalar_t {{ slot.name }} = scalar_t(0);
{% endfor %}
  int64_t edge_begin = row_ptr[node];
  int64_t edge_end = row_ptr[node + 1];
  edge_begin = edge_begin < 0 ? 0 : edge_begin;
  edge_end = edge_end < edge_begin ? edge_begin : edge_end;
  edge_end = edge_end > edges ? edges : edge_end;
  for (int64_t e = edge_begin; e < edge_end; ++e) {
    int32_t sender = senders[e];
    if (sender < 0 || sender >= nodes) continue;
{% for path in paths %}
    {
      int weight = {{ path.weight_start }} + channel;
      scalar_t radial_weight = weights[e * WEIGHT_DIM + weight];
{% for output in path.outputs %}
      {{ output.value_name }} += radial_weight * ({{ primal_angular(output, "channel") }});
{% endfor %}
    }
{% endfor %}
  }
{% for slot in output_slots %}
  out[node * OUTPUT_DIM + {{ slot.output_index }}
      + channel * {{ slot.output_irrep_dim }}] = {{ slot.name }};
{% endfor %}
}

// The same (receiver node, channel) ownership computes the forward tangent.
extern "C" __global__ void oeq_projected_forward_jvp(
    int64_t nodes,
    int64_t edges,
    const scalar_t* x,
    const scalar_t* sh,
    const scalar_t* weights,
    const int32_t* senders,
    const int32_t* row_ptr,
    const scalar_t* tx,
    const scalar_t* tsh,
    const scalar_t* tweights,
    scalar_t* out) {
  int64_t work = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = nodes * CHANNEL_DIM;
  if (work >= total) return;

  int64_t node = work / CHANNEL_DIM;
  int channel = int(work - node * CHANNEL_DIM);
{% for slot in output_slots %}
  scalar_t tangent_{{ slot.name }} = scalar_t(0);
{% endfor %}
  int64_t edge_begin = row_ptr[node];
  int64_t edge_end = row_ptr[node + 1];
  edge_begin = edge_begin < 0 ? 0 : edge_begin;
  edge_end = edge_end < edge_begin ? edge_begin : edge_end;
  edge_end = edge_end > edges ? edges : edge_end;
  for (int64_t e = edge_begin; e < edge_end; ++e) {
    int32_t sender = senders[e];
    if (sender < 0 || sender >= nodes) continue;
{% for path in paths %}
    {
      int weight = {{ path.weight_start }} + channel;
      scalar_t radial_weight = weights[e * WEIGHT_DIM + weight];
      scalar_t tangent_weight = FTW(e * WEIGHT_DIM + weight);
{% for output in path.outputs %}
      tangent_{{ output.value_name }} +=
          tangent_weight * ({{ primal_angular(output, "channel") }})
          + radial_weight * ({{ angular_jvp(output, "channel") }});
{% endfor %}
    }
{% endfor %}
  }
{% for slot in output_slots %}
  out[node * OUTPUT_DIM + {{ slot.output_index }}
      + channel * {{ slot.output_irrep_dim }}] = tangent_{{ slot.name }};
{% endfor %}
}

// One logical 32-thread group owns an edge, including on HIP wavefront-64.
// Lane `lane` owns channels lane, lane + 32, ... .
extern "C" __global__ void oeq_projected_spatial_backward(
    int64_t nodes,
    int64_t edges,
    const scalar_t* x,
    const scalar_t* sh,
    const scalar_t* weights,
    const int32_t* senders,
    const int32_t* receivers,
    const scalar_t* dout,
    scalar_t* dx,
    scalar_t* dsh,
    scalar_t* dweights) {
  int lane = threadIdx.x & 31;
  int logical_group = threadIdx.x >> 5;
  int64_t e = int64_t(blockIdx.x) * (blockDim.x >> 5) + logical_group;
  if (e >= edges) return;

  int32_t sender = senders[e];
  int32_t receiver = receivers[e];
  if (sender < 0 || sender >= nodes || receiver < 0 || receiver >= nodes)
    return;

  acc_t edge_gradient[EDGE_DIM] = {acc_t(0)};
  for (int channel = lane; channel < CHANNEL_DIM; channel += 32) {
{% for slot in input_gradient_slots %}
    acc_t {{ slot.name }} = acc_t(0);
{% endfor %}
    // Path weight intervals are disjoint. This lane directly owns each
    // dweights[e, weight] element generated below.
{% for path in paths %}
    {
      int weight = {{ path.weight_start }} + channel;
      scalar_t radial_weight = weights[e * WEIGHT_DIM + weight];
      acc_t weight_gradient = acc_t(0);
{% for output in path.outputs %}
      {
      scalar_t output_gradient = {{ output_adjoint(output, "channel") }};
      // Accumulate weight, sender-feature, and edge-feature adjoints.
{% for term in output.terms %}
      weight_gradient += output_gradient
          * x[sender * INPUT_DIM + {{ term.input_index }}
              + channel * {{ output.input_irrep_dim }}]
          * sh[e * EDGE_DIM + {{ term.edge_index }}]
          * {{ term.coefficient }};
      {{ term.input_gradient_name }} += output_gradient * radial_weight
          * sh[e * EDGE_DIM + {{ term.edge_index }}]
          * {{ term.coefficient }};
      edge_gradient[{{ term.edge_index }}] += output_gradient * radial_weight
          * x[sender * INPUT_DIM + {{ term.input_index }}
              + channel * {{ output.input_irrep_dim }}]
          * {{ term.coefficient }};
{% endfor %}
      }
{% endfor %}
      dweights[e * WEIGHT_DIM + weight] = scalar_t(weight_gradient);
    }
{% endfor %}
    // Paths are combined locally for this (edge, channel, input component),
    // but different receiver-owned edges can contribute to the same sender.
{% for slot in input_gradient_slots %}
    {{ atomic_add }}(dx + sender * INPUT_DIM + {{ slot.input_index }}
        + channel * {{ slot.input_irrep_dim }}, scalar_t({{ slot.name }}));
{% endfor %}
  }

  // Reduce within the logical 32-thread edge owner. Lane 0 writes dsh[e, :].
  for (int offset = 16; offset > 0; offset >>= 1)
    for (int component = 0; component < EDGE_DIM; ++component)
      edge_gradient[component] +=
          {{ shfl_down_32("edge_gradient[component]", "offset") }};
  if (lane == 0)
    for (int component = 0; component < EDGE_DIM; ++component)
      dsh[e * EDGE_DIM + component] = scalar_t(edge_gradient[component]);
}

// One thread owns (edge, channel). Path weight intervals are disjoint, so each
// dweights[e, weight] element has one direct writer.
extern "C" __global__ void oeq_projected_weight_backward(
    int64_t nodes,
    int64_t edges,
    const scalar_t* x,
    const scalar_t* sh,
    const int32_t* senders,
    const int32_t* receivers,
    const scalar_t* dout,
    scalar_t* dweights) {
  int64_t work = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (work >= edges * CHANNEL_DIM) return;
  int64_t e = work / CHANNEL_DIM;
  int channel = int(work - e * CHANNEL_DIM);
  int32_t sender = senders[e];
  int32_t receiver = receivers[e];
  if (sender < 0 || sender >= nodes || receiver < 0 || receiver >= nodes)
    return;
{% for path in paths %}
  {
    int weight = {{ path.weight_start }} + channel;
    acc_t weight_gradient = acc_t(0);
{% for output in path.outputs %}
{% for term in output.terms %}
    weight_gradient += {{ output_adjoint(output, "channel") }}
        * x[sender * INPUT_DIM + {{ term.input_index }}
            + channel * {{ output.input_irrep_dim }}]
        * sh[e * EDGE_DIM + {{ term.edge_index }}]
        * {{ term.coefficient }};
{% endfor %}
{% endfor %}
    dweights[e * WEIGHT_DIM + weight] = scalar_t(weight_gradient);
  }
{% endfor %}
}

// Mixed derivative of the edge-owned reverse pass, with the same logical
// 32-thread group and channel ownership as oeq_projected_spatial_backward.
extern "C" __global__ void oeq_projected_spatial_backward_jvp(
    int64_t nodes,
    int64_t edges,
    const scalar_t* x,
    const scalar_t* sh,
    const scalar_t* weights,
    const int32_t* senders,
    const int32_t* receivers,
    const scalar_t* dout,
    const scalar_t* tx,
    const scalar_t* tsh,
    const scalar_t* tweights,
    const scalar_t* tdout,
    scalar_t* tdx,
    scalar_t* tdsh,
    scalar_t* tdweights) {
  int lane = threadIdx.x & 31;
  int logical_group = threadIdx.x >> 5;
  int64_t e = int64_t(blockIdx.x) * (blockDim.x >> 5) + logical_group;
  if (e >= edges) return;

  int32_t sender = senders[e];
  int32_t receiver = receivers[e];
  if (sender < 0 || sender >= nodes || receiver < 0 || receiver >= nodes)
    return;

  acc_t tangent_edge_gradient[EDGE_DIM] = {acc_t(0)};
  for (int channel = lane; channel < CHANNEL_DIM; channel += 32) {
{% for slot in input_gradient_slots %}
    acc_t {{ slot.name }} = acc_t(0);
{% endfor %}
    // Path weight intervals are disjoint. This lane directly owns each
    // tdweights[e, weight] element generated below.
{% for path in paths %}
    {
      int weight_index = {{ path.weight_start }} + channel;
      scalar_t radial_weight = weights[e * WEIGHT_DIM + weight_index];
      acc_t tangent_weight_gradient = acc_t(0);
{% for output in path.outputs %}
      {
      scalar_t output_gradient = {{ output_adjoint(output, "channel") }};
      scalar_t tangent_output_gradient = TDOUT(
          tdout[receiver * OUTPUT_DIM + {{ output.output_index }}
              + channel * {{ output.output_irrep_dim }}]);
{% for term in output.terms %}
      {{ term.input_gradient_name }} +=
          (tangent_output_gradient * radial_weight
              * sh[e * EDGE_DIM + {{ term.edge_index }}]
           + output_gradient * TW(e * WEIGHT_DIM + weight_index)
              * sh[e * EDGE_DIM + {{ term.edge_index }}]
           + output_gradient * radial_weight
              * TSH(e * EDGE_DIM + {{ term.edge_index }}))
          * {{ term.coefficient }};
      tangent_edge_gradient[{{ term.edge_index }}] +=
          (tangent_output_gradient * radial_weight
              * x[sender * INPUT_DIM + {{ term.input_index }}
                  + channel * {{ output.input_irrep_dim }}]
           + output_gradient * TW(e * WEIGHT_DIM + weight_index)
              * x[sender * INPUT_DIM + {{ term.input_index }}
                  + channel * {{ output.input_irrep_dim }}]
           + output_gradient * radial_weight
              * TX(sender * INPUT_DIM + {{ term.input_index }}
                  + channel * {{ output.input_irrep_dim }}))
          * {{ term.coefficient }};
      tangent_weight_gradient +=
          (tangent_output_gradient
              * x[sender * INPUT_DIM + {{ term.input_index }}
                  + channel * {{ output.input_irrep_dim }}]
              * sh[e * EDGE_DIM + {{ term.edge_index }}]
           + output_gradient
              * TX(sender * INPUT_DIM + {{ term.input_index }}
                  + channel * {{ output.input_irrep_dim }})
              * sh[e * EDGE_DIM + {{ term.edge_index }}]
           + output_gradient
              * x[sender * INPUT_DIM + {{ term.input_index }}
                  + channel * {{ output.input_irrep_dim }}]
              * TSH(e * EDGE_DIM + {{ term.edge_index }}))
          * {{ term.coefficient }};
{% endfor %}
      }
{% endfor %}
      tdweights[e * WEIGHT_DIM + weight_index] = scalar_t(tangent_weight_gradient);
    }
{% endfor %}
    // Paths are combined locally for this (edge, channel, input component),
    // but different receiver-owned edges can contribute to the same sender.
{% for slot in input_gradient_slots %}
    {{ atomic_add }}(tdx + sender * INPUT_DIM + {{ slot.input_index }}
        + channel * {{ slot.input_irrep_dim }}, scalar_t({{ slot.name }}));
{% endfor %}
  }

  // Reduce within the logical 32-thread edge owner. Lane 0 writes tdsh[e, :].
  for (int offset = 16; offset > 0; offset >>= 1)
    for (int component = 0; component < EDGE_DIM; ++component)
      tangent_edge_gradient[component] +=
          {{ shfl_down_32("tangent_edge_gradient[component]", "offset") }};
  if (lane == 0)
    for (int component = 0; component < EDGE_DIM; ++component)
      tdsh[e * EDGE_DIM + component] =
          scalar_t(tangent_edge_gradient[component]);
}
