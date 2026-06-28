#include "loses.h"
#include "tensor.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define LOSS_EPS 1e-7f

static _loss_result_t loss_result_null( void )
{
  return ( _loss_result_t ){ .loss = NULL, .gradients = NULL };
}

static bool shapes_match( const _tensor_t *a, const _tensor_t *b )
{
  if ( a->dimension != b->dimension ) return false;
  for ( int i = 0; i < a->dimension; i++ )
  {
    if ( a->shape[i] != b->shape[i] ) return false;
  }

  return true;
}

_loss_result_t loss_softmax_cross_entropy( _ds_arena_t_ *arena, _tensor_t *logits, _tensor_t *labels )
{
  if ( !logits || !labels )
  {
    fprintf( stderr, "loss_softmax_cross_entropy: NULL input\n" );
    return loss_result_null();
  }
  if ( logits->dimension != 2 )
  {
    fprintf( stderr, "loss_softmax_cross_entropy: logits must be 2D [batch, classes], got %dD\n", logits->dimension );
    return loss_result_null();
  }
  if ( !shapes_match( logits, labels ) )
  {
    fprintf( stderr, "loss_softmax_cross_entropy: logits and labels shape mismatch\n" );
    return loss_result_null();
  }

  int batch_size = logits->shape[0];
  int number_classes = logits->shape[1];
  float inv_batch = 1.0f / (float)batch_size;

  _tensor_t *gradients = tensor_zeros( arena, 2, logits->shape );
  if ( gradients == NULL ) return loss_result_null();

  float total_loss = 0.0f;

  for ( int batch_idx = 0; batch_idx < batch_size; ++batch_idx )
  {
    // find the max probability for the current sample in batch_idx
    float max = T2( logits, batch_idx, 0 );
    for ( int class = 0; class < number_classes; ++class )
      if ( T2( logits, batch_idx, class ) > max ) max = T2( logits, batch_idx, class );

    // compute expf(z - max) sum
    float exp_sum = 0.0f;
    for ( int class = 0; class < number_classes; ++class )
    {
      float e = expf( T2( logits, batch_idx, class ) - max );
      T2( gradients, batch_idx, class ) = e;
      exp_sum += e;
    }

    // compute p[class] = exp(z - max) / sum
    float sample_loss = 0.0f;
    for ( int class = 0; class < number_classes; ++class )
    {
      float p = T2( gradients, batch_idx, class ) / exp_sum;
      float y = T2( labels, batch_idx, class );

      // compute the cross-entropy for the current class
      sample_loss -= y * logf( fmaxf( p, LOSS_EPS ) );

      // compute the gradients
      T2( gradients, batch_idx, class ) = ( p - y ) * inv_batch;
    }

    total_loss += sample_loss;
  }

  // mean loss over batch, stored as a scalar [1] tensor
  int loss_shape[] = { 1 };
  _tensor_t *loss_tensor = tensor_zeros( arena, 1, loss_shape );
  if ( !loss_tensor ) return loss_result_null();

  loss_tensor->data[0] = total_loss * inv_batch;

  return ( _loss_result_t ){ .loss = loss_tensor, .gradients = gradients };
}
