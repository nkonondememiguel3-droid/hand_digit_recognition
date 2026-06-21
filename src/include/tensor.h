#ifndef hand_digit_recognition_h
#define hand_digit_recognition_h

#include "ds_arena.h"
#include <stdbool.h>
#include <stdint.h>

#define TENSOR_MAX_DIMS 8

typedef struct
{
  float *data;
  float *gradients;
  int dimension;
  int shape[TENSOR_MAX_DIMS];
  int strides[TENSOR_MAX_DIMS];

  int size;
  bool requires_gradients; // does the current tensor has gradients associated with it?
} _tensor_t;

extern _tensor_t *tensor_create( _ds_arena_t_ *a, int dimension, const int shape[], bool requires_gradients );
extern _tensor_t *tensor_zeros( _ds_arena_t_ *a, int dimension, const int shape[] );
extern _tensor_t *tensor_ones( _ds_arena_t_ *a, int dimension, const int shape[] );
extern _tensor_t *tensor_random_normal( _ds_arena_t_ *a, int ndim, const int *shape, float mean, float std );
extern void tensor_zero_gradients( _tensor_t *grad );
extern void tensor_clip_gradient( _tensor_t *grad, float clip_value );

// macros to index into a tensor
#define T4( t, n, c, h, w )                                                                                                                          \
  ( ( t )->data[( n ) * ( t )->strides[0] + ( c ) * ( t )->strides[1] + ( h ) * ( t )->strides[2] + ( w ) * ( t )->strides[3]] )

#define G4( t, n, c, h, w )                                                                                                                          \
  ( ( t )->gradients[( n ) * ( t )->strides[0] + ( c ) * ( t )->strides[1] + ( h ) * ( t )->strides[2] + ( w ) * ( t )->strides[3]] )

#endif // hand_digit_recognition_h
