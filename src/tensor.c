#include "tensor.h"
#include "ds_arena.h"

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

static void compute_strides( _tensor_t *t )
{
  t->strides[t->dimension - 1] = 1;
  for ( int i = t->dimension - 2; i >= 0; i-- ) t->strides[i] = t->strides[i + 1] * t->shape[i + 1];
}

_tensor_t *tensor_create( _ds_arena_t_ *a, int dimension, const int shape[], bool requires_gradients )
{
  _tensor_t *t = ARENA_NEW( a, _tensor_t );
  t->dimension = dimension;
  t->size = 1;

  for ( int i = 0; i < dimension; ++i )
  {
    t->shape[i] = shape[i];
    t->size *= shape[i];
  }
  compute_strides( t );

  t->data = ARENA_ARRAY( a, float, t->size );

  t->requires_gradients = requires_gradients;
  t->gradients = requires_gradients ? ARENA_ARRAY( a, float, t->size ) : NULL;

  return t;
}

_tensor_t *tensor_zeros( _ds_arena_t_ *a, int dimension, const int shape[] )
{
  return tensor_create( a, dimension, shape, false );
}

_tensor_t *tensor_ones( _ds_arena_t_ *a, int dimension, const int shape[] )
{
  _tensor_t *t = tensor_create( a, dimension, shape, false );
  for ( int i = 0; i < t->size; i++ ) t->data[i] = 1.0f;

  return t;
}

_tensor_t *tensor_random_normal( _ds_arena_t_ *a, int ndim, const int *shape, float mean, float std )
{
  _tensor_t *t = tensor_create( a, ndim, shape, false );

  for ( int i = 0; i < t->size; i += 2 )
  {
    float u1 = (float)rand() / (float)RAND_MAX;
    float u2 = (float)rand() / (float)RAND_MAX;
    if ( u1 < 1e-7f ) u1 = 1e-7f;

    float mag = std * sqrtf( -2.0f * logf( u1 ) );
    float z0 = mag * cosf( 2.0f * (float)M_PI * u2 ) + mean;
    float z1 = mag * sinf( 2.0f * (float)M_PI * u2 ) + mean;

    t->data[i] = z0;
    if ( i + 1 < t->size ) t->data[i + 1] = z1;
  }

  return t;
}

void tensor_zero_gradients( _tensor_t *t )
{
  if ( !t->gradients ) return;
  for ( int i = 0; i < t->size; i++ ) t->gradients[i] = 0.0f;
}

void tensor_clip_gradient( _tensor_t *t, float clip_value )
{
  if ( !t->gradients ) return;
  for ( int i = 0; i < t->size; ++i )
  {
    if ( t->gradients[i] > clip_value ) t->gradients[i] = clip_value;
    if ( t->gradients[i] < -clip_value ) t->gradients[i] = -clip_value;
  }
}
