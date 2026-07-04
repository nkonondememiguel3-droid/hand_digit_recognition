#include "layers.h"
#include "ds_arena.h"
#include "tensor.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

static _tensor_t *dense_forward( _ds_arena_t_ *arena, _layer_t *self, _tensor_t *input )
{
  if ( input->dimension != 2 )
  {
    fprintf( stderr, "dense_forward: expected 2D input [batch, features], got %dD\n", input->dimension );
    return NULL;
  }

  /* ✅ shape now comes from in_shape.dims[1], not the removed in_dimension */
  if ( input->shape[1] != self->in_shape.dims[1] )
  {
    fprintf( stderr, "dense_forward: input features %d != layer in_shape %d\n", input->shape[1], self->in_shape.dims[1] );
    return NULL;
  }

  int batch_size = input->shape[0];
  int features = input->shape[1];
  int out_dimension = self->out_shape.dims[1]; /* ✅ was self->out_dimension */

  int out_shape[] = { batch_size, out_dimension };
  _tensor_t *output = tensor_zeros( arena, 2, out_shape );
  if ( !output ) return NULL;

  for ( int sample = 0; sample < batch_size; sample++ )
  {
    for ( int neuron = 0; neuron < out_dimension; neuron++ )
    {
      float sum = self->bias->data[neuron];

      for ( int feature = 0; feature < features; feature++ ) sum += T2( self->weights, neuron, feature ) * T2( input, sample, feature );

      T2( output, sample, neuron ) = sum;
    }
  }

  /* ✅ was self->last_input — now uses the generic cache field */
  self->cache = input;

  return output;
}

static _tensor_t *dense_backward( _ds_arena_t_ *arena, _layer_t *self, _tensor_t *output_gradients )
{
  if ( output_gradients->dimension != 2 )
  {
    fprintf( stderr, "dense_backward: expected 2D input [batch, features], got %dD\n", output_gradients->dimension );
    return NULL;
  }

  if ( output_gradients->shape[1] != self->out_shape.dims[1] ) /* ✅ */
  {
    fprintf( stderr, "dense_backward: gradient features %d != layer out_shape %d\n", output_gradients->shape[1], self->out_shape.dims[1] );
    return NULL;
  }

  _tensor_t *input = self->cache; /* ✅ was self->last_input */
  if ( !input )
  {
    fprintf( stderr, "dense_backward: cache is NULL -- was forward() called?\n" );
    return NULL;
  }

  int batch_size = input->shape[0];
  int features = input->shape[1];
  int out_dimension = self->out_shape.dims[1]; /* ✅ */

  int gradient_shape[] = { batch_size, features };
  _tensor_t *input_gradients = tensor_zeros( arena, 2, gradient_shape );
  if ( !input_gradients ) return NULL;

  for ( int sample = 0; sample < batch_size; sample++ )
  {
    for ( int neuron = 0; neuron < out_dimension; neuron++ )
    {
      float g = T2( output_gradients, sample, neuron );
      self->bias->gradients[neuron] += g;

      for ( int feature = 0; feature < features; feature++ )
      {
        G2( self->weights, neuron, feature ) += g * T2( input, sample, feature );
        T2( input_gradients, sample, feature ) += g * T2( self->weights, neuron, feature );
      }
    }
  }

  return input_gradients;
}

_layer_t *layer_create_dense( _ds_arena_t_ *arena, int in_features, int out_features )
{
  _layer_t *l = ARENA_NEW( arena, _layer_t );
  l->layer_type = LAYER_DENSE;
  l->layer_name = "dense";

  l->in_shape = layer_shape_1d( /*batch_agnostic=*/1, in_features );
  l->out_shape = layer_shape_1d( /*batch_agnostic=*/1, out_features );

  int w_shape[] = { out_features, in_features };
  float limit = sqrtf( 6.0f / ( in_features + out_features ) );
  l->weights = tensor_random_uniform( arena, 2, w_shape, -limit, limit, true );

  int b_shape[] = { out_features };
  l->bias = tensor_create( arena, 1, b_shape, true );

  l->forward = dense_forward;
  l->backward = dense_backward;

  return l;
}

static _tensor_t *sigmoid_forward( _ds_arena_t_ *arena, _layer_t *self, _tensor_t *input )
{
  _tensor_t *output = tensor_zeros( arena, input->dimension, input->shape );
  if ( !output ) return NULL;

  for ( int i = 0; i < input->size; i++ ) output->data[i] = 1.0f / ( 1.0f + expf( -input->data[i] ) );

  self->cache = output; /* ✅ was self->last_input */
  return output;
}

static _tensor_t *sigmoid_backward( _ds_arena_t_ *arena, _layer_t *self, _tensor_t *output_gradient )
{
  _tensor_t *sigmoid_out = self->cache; /* ✅ was self->last_input */
  if ( !sigmoid_out )
  {
    fprintf( stderr, "sigmoid_backward: cache is NULL -- was forward() called?\n" );
    return NULL;
  }

  _tensor_t *input_gradients = tensor_zeros( arena, sigmoid_out->dimension, sigmoid_out->shape );
  if ( !input_gradients ) return NULL;

  for ( int i = 0; i < sigmoid_out->size; ++i )
  {
    float s = sigmoid_out->data[i];
    input_gradients->data[i] = output_gradient->data[i] * s * ( 1.0f - s );
  }

  return input_gradients;
}

_layer_t *layer_create_sigmoid( _ds_arena_t_ *arena )
{
  _layer_t *sigmoid = ARENA_NEW( arena, _layer_t );

  sigmoid->layer_type = LAYER_SIGMOID;
  sigmoid->layer_name = "sigmoid";

  sigmoid->weights = NULL;
  sigmoid->bias = NULL;
  /* in_shape/out_shape default to {.ndim = 0} via ARENA_NEW's zeroing,
     which correctly means "shape-agnostic, accepts/passes through
     whatever it receives" -- no explicit assignment needed.           */

  sigmoid->forward = sigmoid_forward;
  sigmoid->backward = sigmoid_backward;

  return sigmoid;
}

static _tensor_t *relu_forward( _ds_arena_t_ *arena, _layer_t *self, _tensor_t *input )
{
  _tensor_t *output = tensor_zeros( arena, input->dimension, input->shape );
  if ( !output ) return NULL;

  for ( int i = 0; i < input->size; i++ ) output->data[i] = input->data[i] > 0.0f ? input->data[i] : 0.0f;

  self->cache = output; /* ✅ was self->last_input */
  return output;
}

static _tensor_t *relu_backward( _ds_arena_t_ *arena, _layer_t *self, _tensor_t *output_gradient )
{
  _tensor_t *input = self->cache; /* ✅ was self->last_input */
  if ( !input )
  {
    fprintf( stderr, "relu_backward: cache is NULL\n" );
    return NULL;
  }

  _tensor_t *grad = tensor_zeros( arena, input->dimension, input->shape );
  if ( !grad ) return NULL;

  for ( int i = 0; i < input->size; i++ ) grad->data[i] = input->data[i] > 0.0f ? output_gradient->data[i] : 0.0f;

  return grad;
}

_layer_t *layer_create_relu( _ds_arena_t_ *arena )
{
  _layer_t *relu = ARENA_NEW( arena, _layer_t );

  relu->layer_type = LAYER_RELU;
  relu->layer_name = "relu";

  relu->weights = NULL;
  relu->bias = NULL;

  relu->forward = relu_forward;
  relu->backward = relu_backward;

  return relu;
}
