#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"

#include "ds_arena.h"

static void compute_strides( _tensor_t *t )
{
  t->strides[t->dimension - 1] = 1;
  for ( int i = t->dimension - 2; i >= 0; i-- ) t->strides[i] = t->strides[i + 1] * t->shape[i + 1];
}

/* tensor_create
 * Returns NULL on invalid arguments rather than silently producing a
 * corrupt tensor (e.g. size == 0 because a dimension was 0).
 * ds_arena_alloc zeroes its memory, so data and gradients start at 0.
 */
_tensor_t *tensor_create( _ds_arena_t_ *a, int dimension, const int shape[], bool requires_gradients )
{
  if ( dimension <= 0 || dimension > TENSOR_MAX_DIMS )
  {
    fprintf( stderr, "tensor_create: invalid dimension %d (max %d)\n", dimension, TENSOR_MAX_DIMS );
    return NULL;
  }

  for ( int i = 0; i < dimension; i++ )
  {
    if ( shape[i] <= 0 )
    {
      fprintf( stderr, "tensor_create: invalid shape[%d] = %d\n", i, shape[i] );
      return NULL;
    }
  }

  _tensor_t *t = ARENA_NEW( a, _tensor_t );

  t->dimension = dimension;
  t->size = 1;

  for ( int i = 0; i < dimension; i++ )
  {
    t->shape[i] = shape[i];
    t->size *= shape[i];
  }

  compute_strides( t );

  /* ds_arena_alloc always zeroes, so no explicit memset needed */
  t->data = ARENA_ARRAY( a, float, t->size );

  t->requires_gradients = requires_gradients;
  t->gradients = requires_gradients ? ARENA_ARRAY( a, float, t->size ) : NULL;

  return t;
}

/* tensor_zeros
 * Thin wrapper: no gradient buffer, data already zeroed by the arena.
 */
_tensor_t *tensor_zeros( _ds_arena_t_ *a, int dimension, const int shape[] )
{
  return tensor_create( a, dimension, shape, false );
}

/* tensor_ones
 * Allocate and fill every element with 1.0f.
 */
_tensor_t *tensor_ones( _ds_arena_t_ *a, int dimension, const int shape[] )
{
  _tensor_t *t = tensor_create( a, dimension, shape, false );
  if ( !t ) return NULL;

  for ( int i = 0; i < t->size; i++ ) t->data[i] = 1.0f;
  return t;
}

/* tensor_random_normal
 * Box-Muller transform: generates pairs of N(0,1) samples, scales by std,
 * then shifts by mean.
 *
 * IMPORTANT: the caller MUST call srand() before this function is first
 * used, otherwise every training run will start with identical weights:
 *
 *   #include <time.h>
 *   srand( (unsigned int)time( NULL ) );
 *
 * Guard against u1 == 0 to prevent log(0) = -inf.
 */
_tensor_t *tensor_random_normal( _ds_arena_t_ *a, int dimension, const int *shape, float mean, float std, bool gradient_required )
{
  _tensor_t *t = tensor_create( a, dimension, shape, gradient_required );
  if ( !t ) return NULL;

  for ( int i = 0; i < t->size; i += 2 )
  {
    float u1 = (float)rand() / (float)RAND_MAX;
    float u2 = (float)rand() / (float)RAND_MAX;

    /* Clamp u1 away from 0 to keep logf defined */
    if ( u1 < 1e-7f ) u1 = 1e-7f;

    float mag = std * sqrtf( -2.0f * logf( u1 ) );
    float z0 = mag * cosf( 2.0f * (float)M_PI * u2 ) + mean;
    float z1 = mag * sinf( 2.0f * (float)M_PI * u2 ) + mean;

    t->data[i] = z0;
    if ( i + 1 < t->size ) t->data[i + 1] = z1;
  }

  return t;
}


_tensor_t *tensor_random_uniform( _ds_arena_t_ *arena, int ndim, const int *shape, float min, float max, bool gradient_required )
{
  _tensor_t *t = tensor_create( arena, ndim, shape, gradient_required );

  float range = max - min;

  for ( int i = 0; i < t->size; i++ )
  {
    float u = (float)rand() / (float)RAND_MAX;
    t->data[i] = min + range * u;
  }

  return t;
}

/* Gradient helpers */

void tensor_zero_gradients( _tensor_t *t )
{
  if ( !t || !t->gradients ) return;
  for ( int i = 0; i < t->size; i++ ) t->gradients[i] = 0.0f;
}

void tensor_clip_gradient( _tensor_t *t, float clip_value )
{
  if ( !t || !t->gradients ) return;

  for ( int i = 0; i < t->size; i++ )
  {
    if ( t->gradients[i] > clip_value ) t->gradients[i] = clip_value;
    if ( t->gradients[i] < -clip_value ) t->gradients[i] = -clip_value;
  }
}
