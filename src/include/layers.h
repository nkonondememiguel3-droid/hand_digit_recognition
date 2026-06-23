#ifndef hand_digit_recognition_layers_h
#define hand_digit_recognition_layers_h

#include "ds_arena.h"
#include "tensor.h"

typedef struct _layer_ _layer_t;

typedef enum
{
  LAYER_DENSE,
  LAYER_CONV2D,
  LAYER_BATCHNORM,
  LAYER_GAP,

  // activation function
  LAYER_RELU,
} _layer_type_t;

typedef struct _layer_
{
  _layer_type_t layer_type;
  const char *layer_name;

  _tensor_t *weights;
  _tensor_t *bias;

  _tensor_t *( *forward )( _ds_arena_t_ *arena, _layer_t *self, _tensor_t *inut );
  _tensor_t *( *backward )( _ds_arena_t_ *arena, _layer_t *self, _tensor_t *gradients_output );
} _layer_t;

#endif // hand_digit_recognition_layers_h
