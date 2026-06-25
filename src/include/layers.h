#ifndef hand_digit_recognition_layers_h
#define hand_digit_recognition_layers_h

#include "ds_arena.h"
#include "tensor.h"

typedef struct _layer_ _layer_t;

typedef enum
{
  LAYER_DENSE,
  LAYER_CONV2D,

  // optimization layers
  LAYER_BATCHNORM,
  LAYER_GAP,

  // activation layers
  LAYER_RELU,
  LAYER_SIGMOID,
} _layer_type_t;

typedef struct _layer_
{
  _layer_type_t layer_type;
  const char *layer_name;

  _tensor_t *weights;
  _tensor_t *bias;

  // this is the shape of the matrix representing the layer.
  int in_dimension;
  int out_dimension;

  // cache
  _tensor_t *last_input;

  _tensor_t *( *forward )( _ds_arena_t_ *arena, _layer_t *self, _tensor_t *inut );
  _tensor_t *( *backward )( _ds_arena_t_ *arena, _layer_t *self, _tensor_t *gradients_output );
} _layer_t;

extern _layer_t *layer_create_dense( _ds_arena_t_ *arena, int in_features, int out_features );

#endif // hand_digit_recognition_layers_h
