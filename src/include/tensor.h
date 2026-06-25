#ifndef hand_digit_recognition_tensor_h /* ✅ fixed guard name */
#define hand_digit_recognition_tensor_h

#include "ds_arena.h"
#include <stdbool.h>
#include <stdint.h>

#define TENSOR_MAX_DIMS 8

/*
 * _tensor_t — generic N-dimensional float tensor, arena-allocated.
 *
 * Memory layout: flat row-major buffer.
 * Strides are pre-computed so indexing is a single multiply-add chain.
 *
 * `requires_gradients` controls whether a gradient buffer is allocated
 * alongside the data buffer. Set true for weight/bias tensors, false for
 * intermediate activations that don't need gradient accumulation.
 */
typedef struct
{
  float *data;                  /* flat row-major data buffer                  */
  float *gradients;             /* gradient buffer, NULL if !requires_gradients */
  int dimension;                /* number of active dimensions                 */
  int shape[TENSOR_MAX_DIMS];   /* size along each dimension            */
  int strides[TENSOR_MAX_DIMS]; /* pre-computed row-major strides       */
  int size;                     /* total element count (product of shape)      */
  bool requires_gradients;
} _tensor_t;

/*
 * tensor_create — allocate a tensor from `a`.
 * Returns NULL if any shape[i] <= 0 or dimension > TENSOR_MAX_DIMS.
 * Data buffer is zero-initialised (ds_arena_alloc guarantees this).
 * Gradient buffer is zero-initialised too, when allocated.
 */
extern _tensor_t *tensor_create( _ds_arena_t_ *a, int dimension, const int shape[], bool requires_gradients );

/* Convenience: zeroed tensor (no gradient buffer) */
extern _tensor_t *tensor_zeros( _ds_arena_t_ *a, int dimension, const int shape[] );

/* Convenience: tensor filled with 1.0f (no gradient buffer) */
extern _tensor_t *tensor_ones( _ds_arena_t_ *a, int dimension, const int shape[] );

// randomly initiliaze the learnable parmaters of the neuron network.
// NOTE: YOU MUST CALL srand before using these functions.
extern _tensor_t *tensor_random_normal( _ds_arena_t_ *a, int ndim, const int *shape, float mean, float std );
extern _tensor_t *tensor_random_uniform( _ds_arena_t_ *arena, int ndim, const int *shape, float min, float max, bool gradient_required );

/* Gradient helpers */

/* Zero-fill the gradient buffer. No-op if gradients == NULL. */
extern void tensor_zero_gradients( _tensor_t *t );

/* Clip every gradient element to [-clip_value, +clip_value]. */
extern void tensor_clip_gradient( _tensor_t *t, float clip_value );

/* Rank-generic index macros
 *
 * These expand to a single expression so they are safe on the right-hand
 * side of assignments and inside larger expressions.
 *
 * T1..T4  →  data element
 * G1..G4  →  gradient element (only valid when requires_gradients == true)
 *
 * Strides are pre-computed by tensor_create, so each access is:
 *   offset = i0*s0 + i1*s1 + ...   (no division, no modulo)
 */
#define T1( t, i ) ( ( t )->data[( i ) * ( t )->strides[0]] )
#define T2( t, i, j ) ( ( t )->data[( i ) * ( t )->strides[0] + ( j ) * ( t )->strides[1]] )
#define T3( t, i, j, k ) ( ( t )->data[( i ) * ( t )->strides[0] + ( j ) * ( t )->strides[1] + ( k ) * ( t )->strides[2]] )
#define T4( t, n, c, h, w )                                                                                                                          \
  ( ( t )->data[( n ) * ( t )->strides[0] + ( c ) * ( t )->strides[1] + ( h ) * ( t )->strides[2] + ( w ) * ( t )->strides[3]] )

#define G1( t, i ) ( ( t )->gradients[( i ) * ( t )->strides[0]] )
#define G2( t, i, j ) ( ( t )->gradients[( i ) * ( t )->strides[0] + ( j ) * ( t )->strides[1]] )
#define G3( t, i, j, k ) ( ( t )->gradients[( i ) * ( t )->strides[0] + ( j ) * ( t )->strides[1] + ( k ) * ( t )->strides[2]] )
#define G4( t, n, c, h, w )                                                                                                                          \
  ( ( t )->gradients[( n ) * ( t )->strides[0] + ( c ) * ( t )->strides[1] + ( h ) * ( t )->strides[2] + ( w ) * ( t )->strides[3]] )

#endif /* hand_digit_recognition_tensor_h */
