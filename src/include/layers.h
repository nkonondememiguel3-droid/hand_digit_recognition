#ifndef hand_digit_recognition_layers_h
#define hand_digit_recognition_layers_h

#include "ds_arena.h"
#include "tensor.h"
#include <stdint.h>

typedef struct _layer_ _layer_t;

typedef enum
{
  LAYER_DENSE,
  LAYER_CONV2D,
  LAYER_MAXPOOL2D,
  LAYER_BATCHNORM,
  LAYER_GAP,
  LAYER_RELU,
  LAYER_SIGMOID,
  LAYER_FLATTEN,
  LAYER_DROPOUT,
} _layer_type_t;

/* ── Shape descriptor ─────────────────────────────────────────────────────
 *
 * Generic replacement for in_dimension/out_dimension. A layer declares
 * the shape it expects on input and the shape it produces on output,
 * using the same TENSOR_MAX_DIMS-rank convention as _tensor_t itself.
 *
 * dims[0] is always treated as the batch dimension and is allowed to be
 * 0 (meaning "any batch size") — every other entry must be a concrete
 * positive size, since that's what determines weight tensor shapes.
 *
 * Example -- dense(784 -> 128):
 *   in_shape  = { ndim=2, dims={0, 784} }
 *   out_shape = { ndim=2, dims={0, 128} }
 *
 * Example -- conv2d(in_c=1, out_c=32, 28x28 -> 28x28, k=3, pad=1):
 *   in_shape  = { ndim=4, dims={0, 1,  28, 28} }
 *   out_shape = { ndim=4, dims={0, 32, 28, 28} }
 */
typedef struct
{
  int ndim;
  int dims[TENSOR_MAX_DIMS];
} _layer_shape_t;

static inline _layer_shape_t layer_shape_1d( int batch_agnostic, int d0 )
{
  _layer_shape_t s = { .ndim = 2 };
  s.dims[0] = batch_agnostic ? 0 : 1;
  s.dims[1] = d0;
  return s;
}

static inline _layer_shape_t layer_shape_3d( int c, int h, int w )
{
  _layer_shape_t s = { .ndim = 4 };
  s.dims[0] = 0; /* batch-agnostic */
  s.dims[1] = c;
  s.dims[2] = h;
  s.dims[3] = w;
  return s;
}

/* Does `actual` satisfy `expected`? dims[0]=0 in expected matches any
   batch size in actual; every other dim must match exactly.            */
static inline bool layer_shape_matches( const _layer_shape_t *expected, const _tensor_t *actual )
{
  if ( expected->ndim != actual->dimension ) return false;
  for ( int i = 0; i < expected->ndim; i++ )
  {
    if ( i == 0 && expected->dims[0] == 0 ) continue; /* batch-agnostic */
    if ( expected->dims[i] != actual->shape[i] ) return false;
  }
  return true;
}

/* ── The generalized layer ───────────────────────────────────────────────
 *
 * `params` is an opaque pointer to a layer-type-specific struct (e.g.
 * _dense_params_t, _conv2d_params_t) allocated alongside the layer from
 * the same arena. Generic code (network.c, optimizer.c, serialization)
 * never dereferences `params` directly -- it goes through the
 * `get_weights`/`get_bias`/`num_param_tensors` callbacks below, OR for
 * the common case of "one weight tensor + one bias tensor", layers can
 * still populate `weights`/`bias` directly for backward compatibility
 * with the existing optimizer code.
 *
 * For layers with more than 2 parameter tensors (e.g. batchnorm has 4:
 * gamma, beta, running_mean, running_var), extra_params lets the layer
 * expose however many tensors it needs without changing this struct.
 */
typedef struct _layer_
{
  _tensor_t *weights; /* primary weight tensor, or NULL          */
  _tensor_t *bias;    /* primary bias tensor, or NULL            */

  _tensor_t **extra_params; /* additional learnable tensors, or NULL   */
  int num_extra_params;

  _layer_shape_t in_shape;  /* expected input shape                    */
  _layer_shape_t out_shape; /* shape this layer produces               */

  void *params; /* layer-type-specific config/state blob   */

  /* Cache for backward pass. Most layers need exactly one cached tensor
     (last_input or last_output) -- kept as a single slot for the common
     case. Layers needing more (e.g. maxpool's argmax indices) store
     them inside `params` instead.                                      */
  _tensor_t *cache;

  _layer_type_t layer_type;
  const char *layer_name;

  _tensor_t *( *forward )( _ds_arena_t_ *arena, _layer_t *self, _tensor_t *input );
  _tensor_t *( *backward )( _ds_arena_t_ *arena, _layer_t *self, _tensor_t *grad_output );
} _layer_t;

/* ── Existing layer constructors (signatures unchanged) ──────────────── */
extern _layer_t *layer_create_dense( _ds_arena_t_ *arena, int in_features, int out_features );
extern _layer_t *layer_create_sigmoid( _ds_arena_t_ *arena );
extern _layer_t *layer_create_relu( _ds_arena_t_ *arena );

#endif /* hand_digit_recognition_layers_h */
