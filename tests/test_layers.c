#include "ds_arena.h"
#include "layers.h"
#include "tensor.h"
#include <criterion/criterion.h>
#include <criterion/internal/assert.h>
#include <criterion/internal/test.h>
#include <math.h>

static _ds_arena_t_ param_arena;
static _ds_arena_t_ batch_arena;

void setup( void )
{
  param_arena = ds_arena_new( 0 );
  batch_arena = ds_arena_new( 0 );
}

void teardown( void )
{
  ds_arena_destroy( &param_arena );
  ds_arena_destroy( &batch_arena );
}

TestSuite( layers, .init = setup, .fini = teardown );

Test( layers, dense_create_not_null )
{
  _layer_t *dense = layer_create_dense( &param_arena, 128, 64 );
  cr_assert_not_null( dense );
}

Test( layers, dense_layer_type )
{
  _layer_t *dense = layer_create_dense( &param_arena, 16, 8 );
  cr_assert_eq( dense->layer_type, LAYER_DENSE );
}

Test( layers, dense_dimensions_stored )
{
  _layer_t *dense = layer_create_dense( &param_arena, 128, 64 );
  /* dims[0] is the batch slot (0 == batch-agnostic); features live in dims[1] */
  cr_assert_eq( dense->in_shape.ndim, 2 );
  cr_assert_eq( dense->in_shape.dims[0], 0 );
  cr_assert_eq( dense->in_shape.dims[1], 128 );
  cr_assert_eq( dense->out_shape.dims[1], 64 );
}

Test( layers, dense_weights_shape )
{
  /* W ∈ ℝ^(out × in) */
  _layer_t *dense = layer_create_dense( &param_arena, 32, 16 );
  cr_assert_not_null( dense->weights );
  cr_assert_eq( dense->weights->dimension, 2 );
  cr_assert_eq( dense->weights->shape[0], 16 ); /* out */
  cr_assert_eq( dense->weights->shape[1], 32 ); /* in  */
}

Test( layers, dense_bias_shape )
{
  /* b ∈ ℝ^out */
  _layer_t *dense = layer_create_dense( &param_arena, 32, 16 );
  cr_assert_not_null( dense->bias );
  cr_assert_eq( dense->bias->dimension, 1 );
  cr_assert_eq( dense->bias->shape[0], 16 );
}

Test( layers, dense_weights_have_gradients )
{
  _layer_t *dense = layer_create_dense( &param_arena, 32, 16 );
  cr_assert( dense->weights->requires_gradients );
  cr_assert_not_null( dense->weights->gradients );
}

Test( layers, dense_bias_has_gradients )
{
  _layer_t *dense = layer_create_dense( &param_arena, 32, 16 );
  cr_assert( dense->bias->requires_gradients );
  cr_assert_not_null( dense->bias->gradients );
}

Test( layers, dense_bias_initialised_to_zero )
{
  _layer_t *dense = layer_create_dense( &param_arena, 32, 16 );
  for ( int i = 0; i < dense->bias->size; i++ ) cr_assert_float_eq( dense->bias->data[i], 0.0f, 1e-7f );
}

Test( layers, dense_weights_xavier_range )
{
  /* Xavier uniform: W ∈ [-limit, +limit], limit = sqrt(6/(in+out)) */
  int in = 128, out = 64;
  float limit = sqrtf( 6.0f / (float)( in + out ) );
  _layer_t *dense = layer_create_dense( &param_arena, in, out );
  for ( int i = 0; i < dense->weights->size; i++ )
  {
    cr_assert( dense->weights->data[i] >= -limit - 1e-5f && dense->weights->data[i] <= limit + 1e-5f,
               "Weight[%d]=%.4f outside Xavier range [%.4f, %.4f]", i, dense->weights->data[i], -limit, limit );
  }
}

Test( layers, dense_forward_backward_fn_not_null )
{
  _layer_t *dense = layer_create_dense( &param_arena, 4, 2 );
  cr_assert_not_null( dense->forward );
  cr_assert_not_null( dense->backward );
}

Test( layers, dense_output_shape )
{
  int input_shape[] = { 4, 128 };
  _tensor_t *input = tensor_zeros( &batch_arena, 2, input_shape );
  _layer_t *dense = layer_create_dense( &param_arena, 128, 64 );

  _tensor_t *out = dense->forward( &batch_arena, dense, input );

  cr_assert_not_null( out );
  cr_assert_eq( out->shape[0], 4 );
  cr_assert_eq( out->shape[1], 64 );
}

Test( layers, dense_output_is_2d )
{
  int input_shape[] = { 2, 8 };
  _tensor_t *input = tensor_zeros( &batch_arena, 2, input_shape );
  _layer_t *dense = layer_create_dense( &param_arena, 8, 4 );
  _tensor_t *out = dense->forward( &batch_arena, dense, input );
  cr_assert_eq( out->dimension, 2 );
}

Test( layers, dense_zero_input_output_equals_bias )
{
  /* Y = X·Wᵀ + b  →  when X=0, Y = b  (broadcast over batch) */
  int in = 4, out = 3, batch = 2;
  _layer_t *dense = layer_create_dense( &param_arena, in, out );

  /* Set bias to known values */
  dense->bias->data[0] = 1.0f;
  dense->bias->data[1] = 2.0f;
  dense->bias->data[2] = 3.0f;

  int input_shape[] = { batch, in };
  _tensor_t *input = tensor_zeros( &batch_arena, 2, input_shape );
  _tensor_t *output = dense->forward( &batch_arena, dense, input );

  for ( int s = 0; s < batch; s++ )
  {
    cr_assert_float_eq( T2( output, s, 0 ), 1.0f, 1e-5f );
    cr_assert_float_eq( T2( output, s, 1 ), 2.0f, 1e-5f );
    cr_assert_float_eq( T2( output, s, 2 ), 3.0f, 1e-5f );
  }
}

Test( layers, dense_forward_known_values )
{
  /* Manual computation:
   *   in=2, out=2, batch=1
   *   W = [[1, 2], [3, 4]]    weight_row0=[1,2], weight_row1=[3,4]
   *   b = [0, 0]
   *   x = [1, 1]
   *   y[0] = 1*1 + 2*1 = 3
   *   y[1] = 3*1 + 4*1 = 7                                        */
  _layer_t *dense = layer_create_dense( &param_arena, 2, 2 );
  dense->weights->data[0] = 1.0f;
  dense->weights->data[1] = 2.0f;
  dense->weights->data[2] = 3.0f;
  dense->weights->data[3] = 4.0f;
  dense->bias->data[0] = 0.0f;
  dense->bias->data[1] = 0.0f;

  int input_shape[] = { 1, 2 };
  _tensor_t *input = tensor_ones( &batch_arena, 2, input_shape );
  _tensor_t *output = dense->forward( &batch_arena, dense, input );

  cr_assert_float_eq( T2( output, 0, 0 ), 3.0f, 1e-5f );
  cr_assert_float_eq( T2( output, 0, 1 ), 7.0f, 1e-5f );
}

Test( layers, dense_forward_caches_last_input )
{
  int input_shape[] = { 2, 4 };
  _tensor_t *input = tensor_zeros( &batch_arena, 2, input_shape );
  _layer_t *dense = layer_create_dense( &param_arena, 4, 2 );
  dense->forward( &batch_arena, dense, input );
  cr_assert_eq( dense->cache, input );
}

Test( layers, dense_forward_wrong_feature_count_returns_null )
{
  int input_shape[] = { 2, 99 }; /* dense expects 4 features */
  _tensor_t *input = tensor_zeros( &batch_arena, 2, input_shape );
  _layer_t *dense = layer_create_dense( &param_arena, 4, 2 );
  _tensor_t *out = dense->forward( &batch_arena, dense, input );
  cr_assert_null( out, "forward() should return NULL on feature mismatch" );
}

Test( layers, dense_forward_wrong_rank_returns_null )
{
  int input_shape[] = { 4 }; /* 1D — dense expects 2D */
  _tensor_t *input = tensor_zeros( &batch_arena, 1, input_shape );
  _layer_t *dense = layer_create_dense( &param_arena, 4, 2 );
  _tensor_t *out = dense->forward( &batch_arena, dense, input );
  cr_assert_null( out, "forward() should return NULL on rank mismatch" );
}

Test( layers, dense_backward_input_gradient_shape )
{
  int in = 4, out = 3, batch = 2;
  _layer_t *dense = layer_create_dense( &param_arena, in, out );

  int input_shape[] = { batch, in };
  _tensor_t *input = tensor_zeros( &batch_arena, 2, input_shape );
  dense->forward( &batch_arena, dense, input );

  int grad_shape[] = { batch, out };
  _tensor_t *out_grad = tensor_ones( &batch_arena, 2, grad_shape );
  _tensor_t *input_grad = dense->backward( &batch_arena, dense, out_grad );

  cr_assert_not_null( input_grad );
  cr_assert_eq( input_grad->shape[0], batch );
  cr_assert_eq( input_grad->shape[1], in );
}

Test( layers, dense_backward_weight_gradient_shape )
{
  int in = 4, out = 3, batch = 2;
  _layer_t *dense = layer_create_dense( &param_arena, in, out );

  int input_shape[] = { batch, in };
  _tensor_t *input = tensor_zeros( &batch_arena, 2, input_shape );
  dense->forward( &batch_arena, dense, input );

  int grad_shape[] = { batch, out };
  _tensor_t *out_grad = tensor_ones( &batch_arena, 2, grad_shape );
  dense->backward( &batch_arena, dense, out_grad );

  cr_assert_eq( dense->weights->gradients[0], dense->weights->gradients[0] ); /* not NaN */
  cr_assert_eq( dense->weights->dimension, 2 );
  cr_assert_eq( dense->weights->shape[0], out );
  cr_assert_eq( dense->weights->shape[1], in );
}

Test( layers, dense_backward_zero_input_grad_is_zero )
{
  /* When X=0, ∂L/∂W = 0 (no input to backprop through).
     ∂L/∂X = g·W — nonzero if W nonzero, so we test ∂L/∂W instead. */
  int in = 3, out = 2, batch = 1;
  _layer_t *dense = layer_create_dense( &param_arena, in, out );

  int input_shape[] = { batch, in };
  _tensor_t *input = tensor_zeros( &batch_arena, 2, input_shape );
  dense->forward( &batch_arena, dense, input );

  int grad_shape[] = { batch, out };
  _tensor_t *out_grad = tensor_ones( &batch_arena, 2, grad_shape );
  dense->backward( &batch_arena, dense, out_grad );

  /* ∂L/∂W[j,k] = Σᵢ g[i,j] * X[i,k] = Σᵢ 1 * 0 = 0 */
  for ( int i = 0; i < dense->weights->size; i++ )
    cr_assert_float_eq( dense->weights->gradients[i], 0.0f, 1e-6f, "Weight gradient[%d] should be 0 when input is 0", i );
}

Test( layers, dense_backward_known_weight_gradient )
{
  /* W=[[1,2],[3,4]], b=[0,0], X=[[1,1]] (batch=1)
   * Forward: Y=[[3,7]]
   * Upstream grad: dY=[[1,1]]
   * ∂L/∂W[j,k] = g[0,j] * X[0,k]
   *   ∂L/∂W[0,0] = 1*1=1, ∂L/∂W[0,1] = 1*1=1
   *   ∂L/∂W[1,0] = 1*1=1, ∂L/∂W[1,1] = 1*1=1    */
  _layer_t *dense = layer_create_dense( &param_arena, 2, 2 );
  dense->weights->data[0] = 1.0f;
  dense->weights->data[1] = 2.0f;
  dense->weights->data[2] = 3.0f;
  dense->weights->data[3] = 4.0f;

  int input_shape[] = { 1, 2 };
  _tensor_t *input = tensor_ones( &batch_arena, 2, input_shape );
  dense->forward( &batch_arena, dense, input );

  int grad_shape[] = { 1, 2 };
  _tensor_t *out_grad = tensor_ones( &batch_arena, 2, grad_shape );
  dense->backward( &batch_arena, dense, out_grad );

  cr_assert_float_eq( G2( dense->weights, 0, 0 ), 1.0f, 1e-5f );
  cr_assert_float_eq( G2( dense->weights, 0, 1 ), 1.0f, 1e-5f );
  cr_assert_float_eq( G2( dense->weights, 1, 0 ), 1.0f, 1e-5f );
  cr_assert_float_eq( G2( dense->weights, 1, 1 ), 1.0f, 1e-5f );
}

Test( layers, dense_backward_known_input_gradient )
{
  /* Continuing above: ∂L/∂X[0,k] = Σⱼ g[0,j] * W[j,k]
   *   ∂L/∂X[0,0] = 1*1 + 1*3 = 4
   *   ∂L/∂X[0,1] = 1*2 + 1*4 = 6                           */
  _layer_t *dense = layer_create_dense( &param_arena, 2, 2 );
  dense->weights->data[0] = 1.0f;
  dense->weights->data[1] = 2.0f;
  dense->weights->data[2] = 3.0f;
  dense->weights->data[3] = 4.0f;

  int input_shape[] = { 1, 2 };
  _tensor_t *input = tensor_ones( &batch_arena, 2, input_shape );
  dense->forward( &batch_arena, dense, input );

  int grad_shape[] = { 1, 2 };
  _tensor_t *out_grad = tensor_ones( &batch_arena, 2, grad_shape );
  _tensor_t *input_grad = dense->backward( &batch_arena, dense, out_grad );

  cr_assert_float_eq( T2( input_grad, 0, 0 ), 4.0f, 1e-5f );
  cr_assert_float_eq( T2( input_grad, 0, 1 ), 6.0f, 1e-5f );
}

Test( layers, dense_backward_gradient_accumulates_over_batch )
{
  /* With batch=2, identical rows X=[[1,1],[1,1]], g=[[1,1],[1,1]]:
   * ∂L/∂W[j,k] = Σᵢ g[i,j]*X[i,k] = 2 * 1 * 1 = 2 for all j,k */
  _layer_t *dense = layer_create_dense( &param_arena, 2, 2 );
  for ( int i = 0; i < dense->weights->size; i++ ) dense->weights->data[i] = 1.0f;

  int input_shape[] = { 2, 2 };
  _tensor_t *input = tensor_ones( &batch_arena, 2, input_shape );
  dense->forward( &batch_arena, dense, input );

  int grad_shape[] = { 2, 2 };
  _tensor_t *out_grad = tensor_ones( &batch_arena, 2, grad_shape );
  dense->backward( &batch_arena, dense, out_grad );

  for ( int i = 0; i < dense->weights->size; i++ )
    cr_assert_float_eq( dense->weights->gradients[i], 2.0f, 1e-5f, "Weight grad[%d] should be 2 (summed over batch=2)", i );
}

Test( layers, dense_weight_gradient_zeroed_between_passes )
{
  /* Simulate two backward passes. If gradients aren't zeroed between
     them, the second pass will accumulate on top of the first.        */
  _layer_t *dense = layer_create_dense( &param_arena, 2, 2 );
  for ( int i = 0; i < dense->weights->size; i++ ) dense->weights->data[i] = 1.0f;

  int input_shape[] = { 1, 2 };
  int grad_shape[] = { 1, 2 };

  /* First pass */
  _tensor_t *input1 = tensor_ones( &batch_arena, 2, input_shape );
  dense->forward( &batch_arena, dense, input1 );
  _tensor_t *out_grad1 = tensor_ones( &batch_arena, 2, grad_shape );
  dense->backward( &batch_arena, dense, out_grad1 );
  float grad_after_first = dense->weights->gradients[0];

  /* Zero gradients between passes (caller's responsibility) */
  tensor_zero_gradients( dense->weights );
  tensor_zero_gradients( dense->bias );

  /* Second pass — identical inputs */
  _tensor_t *input2 = tensor_ones( &batch_arena, 2, input_shape );
  dense->forward( &batch_arena, dense, input2 );
  _tensor_t *out_grad2 = tensor_ones( &batch_arena, 2, grad_shape );
  dense->backward( &batch_arena, dense, out_grad2 );
  float grad_after_second = dense->weights->gradients[0];

  cr_assert_float_eq( grad_after_first, grad_after_second, 1e-5f, "Gradient should be identical after zeroing between passes" );
}

Test( layers, sigmoid_create_not_null )
{
  _layer_t *sigmoid = layer_create_sigmoid( &param_arena );
  cr_assert_not_null( sigmoid );
}

Test( layers, sigmoid_layer_type )
{
  _layer_t *sigmoid = layer_create_sigmoid( &param_arena );
  cr_assert_eq( sigmoid->layer_type, LAYER_SIGMOID );
}

Test( layers, sigmoid_has_no_weights_or_bias )
{
  _layer_t *sigmoid = layer_create_sigmoid( &param_arena );
  cr_assert_null( sigmoid->weights );
  cr_assert_null( sigmoid->bias );
}

Test( layers, sigmoid_dimensions_are_zero )
{
  _layer_t *sigmoid = layer_create_sigmoid( &param_arena );
  /* ndim == 0 is what marks an activation as shape-agnostic. Comparing the
     dims array itself would compare a never-null pointer against 0. */
  cr_assert_eq( sigmoid->in_shape.ndim, 0 );
  cr_assert_eq( sigmoid->out_shape.ndim, 0 );
}

Test( layers, sigmoid_forward_backward_fn_not_null )
{
  _layer_t *sigmoid = layer_create_sigmoid( &param_arena );
  cr_assert_not_null( sigmoid->forward );
  cr_assert_not_null( sigmoid->backward );
}

Test( layers, sigmoid_output_shape_preserved )
{
  int shape[] = { 3, 4 };
  _tensor_t *input = tensor_zeros( &batch_arena, 2, shape );
  _layer_t *sigmoid = layer_create_sigmoid( &param_arena );
  _tensor_t *output = sigmoid->forward( &batch_arena, sigmoid, input );

  cr_assert_not_null( output );
  cr_assert_eq( output->dimension, 2 );
  cr_assert_eq( output->shape[0], 3 );
  cr_assert_eq( output->shape[1], 4 );
}

Test( layers, sigmoid_output_at_zero_is_half )
{
  /* σ(0) = 0.5 exactly */
  int shape[] = { 1, 1 };
  _tensor_t *input = tensor_zeros( &batch_arena, 2, shape );
  _layer_t *sigmoid = layer_create_sigmoid( &param_arena );
  _tensor_t *output = sigmoid->forward( &batch_arena, sigmoid, input );

  cr_assert_float_eq( output->data[0], 0.5f, 1e-6f );
}

Test( layers, sigmoid_output_range )
{
  /* All outputs must be strictly in (0, 1) */
  int shape[] = { 1, 8 };
  _tensor_t *input = tensor_zeros( &batch_arena, 2, shape );
  input->data[0] = -10.0f;
  input->data[1] = -1.0f;
  input->data[2] = -0.5f;
  input->data[3] = 0.0f;
  input->data[4] = 0.5f;
  input->data[5] = 1.0f;
  input->data[6] = 5.0f;
  input->data[7] = 10.0f;

  _layer_t *sigmoid = layer_create_sigmoid( &param_arena );
  _tensor_t *output = sigmoid->forward( &batch_arena, sigmoid, input );

  for ( int i = 0; i < output->size; i++ )
    cr_assert( output->data[i] > 0.0f && output->data[i] < 1.0f, "σ(x[%d]) = %.6f should be in (0, 1)", i, output->data[i] );
}

Test( layers, sigmoid_large_positive_input_near_one )
{
  /* σ(+∞) → 1 */
  int shape[] = { 1, 1 };
  _tensor_t *input = tensor_zeros( &batch_arena, 2, shape );
  input->data[0] = 100.0f;
  _layer_t *sigmoid = layer_create_sigmoid( &param_arena );
  _tensor_t *output = sigmoid->forward( &batch_arena, sigmoid, input );
  cr_assert( output->data[0] > 0.999f, "σ(100) should be near 1, got %.6f", output->data[0] );
}

Test( layers, sigmoid_large_negative_input_near_zero )
{
  /* σ(-∞) → 0 */
  int shape[] = { 1, 1 };
  _tensor_t *input = tensor_zeros( &batch_arena, 2, shape );
  input->data[0] = -100.0f;
  _layer_t *sigmoid = layer_create_sigmoid( &param_arena );
  _tensor_t *output = sigmoid->forward( &batch_arena, sigmoid, input );
  cr_assert( output->data[0] < 0.001f, "σ(-100) should be near 0, got %.6f", output->data[0] );
}

Test( layers, sigmoid_caches_output_not_input )
{
  /* Implementation caches output (σ(x)) for use in backward.
     Verify last_input points to the output tensor, not the input. */
  int shape[] = { 2, 3 };
  _tensor_t *input = tensor_zeros( &batch_arena, 2, shape );
  _layer_t *sigmoid = layer_create_sigmoid( &param_arena );
  _tensor_t *output = sigmoid->forward( &batch_arena, sigmoid, input );

  cr_assert_eq( sigmoid->cache, output, "sigmoid should cache its OUTPUT, not its input" );
  cr_assert_neq( sigmoid->cache, input );
}

Test( layers, sigmoid_backward_shape_preserved )
{
  int shape[] = { 3, 4 };
  _tensor_t *input = tensor_zeros( &batch_arena, 2, shape );
  _layer_t *sigmoid = layer_create_sigmoid( &param_arena );
  sigmoid->forward( &batch_arena, sigmoid, input );

  _tensor_t *upstream = tensor_ones( &batch_arena, 2, shape );
  _tensor_t *grad = sigmoid->backward( &batch_arena, sigmoid, upstream );

  cr_assert_not_null( grad );
  cr_assert_eq( grad->dimension, 2 );
  cr_assert_eq( grad->shape[0], 3 );
  cr_assert_eq( grad->shape[1], 4 );
}

Test( layers, sigmoid_backward_at_zero_input )
{
  /* σ(0) = 0.5  →  σ'(0) = 0.5*(1-0.5) = 0.25
     With upstream gradient = 1: result = 1 * 0.25 = 0.25 */
  int shape[] = { 1, 1 };
  _tensor_t *input = tensor_zeros( &batch_arena, 2, shape );
  _layer_t *sigmoid = layer_create_sigmoid( &param_arena );
  sigmoid->forward( &batch_arena, sigmoid, input );

  _tensor_t *upstream = tensor_ones( &batch_arena, 2, shape );
  _tensor_t *grad = sigmoid->backward( &batch_arena, sigmoid, upstream );

  cr_assert_float_eq( grad->data[0], 0.25f, 1e-5f, "σ'(0) = 0.25, got %.6f", grad->data[0] );
}

Test( layers, sigmoid_backward_multiplied_by_upstream )
{
  /* σ(0) = 0.5, σ'(0) = 0.25
     Upstream gradient = 2.0 → result = 2.0 * 0.25 = 0.5 */
  int shape[] = { 1, 1 };
  _tensor_t *input = tensor_zeros( &batch_arena, 2, shape );
  _layer_t *sigmoid = layer_create_sigmoid( &param_arena );
  sigmoid->forward( &batch_arena, sigmoid, input );

  _tensor_t *upstream = tensor_zeros( &batch_arena, 2, shape );
  upstream->data[0] = 2.0f;
  _tensor_t *grad = sigmoid->backward( &batch_arena, sigmoid, upstream );

  cr_assert_float_eq( grad->data[0], 0.5f, 1e-5f, "2.0 * σ'(0) = 0.5, got %.6f", grad->data[0] );
}

Test( layers, sigmoid_backward_zero_upstream_gives_zero_grad )
{
  /* Chain rule: if upstream = 0, local gradient = 0 regardless of σ'(x) */
  int shape[] = { 1, 4 };
  _tensor_t *input = tensor_ones( &batch_arena, 2, shape );
  _layer_t *sigmoid = layer_create_sigmoid( &param_arena );
  sigmoid->forward( &batch_arena, sigmoid, input );

  _tensor_t *upstream = tensor_zeros( &batch_arena, 2, shape );
  _tensor_t *grad = sigmoid->backward( &batch_arena, sigmoid, upstream );

  for ( int i = 0; i < grad->size; i++ ) cr_assert_float_eq( grad->data[i], 0.0f, 1e-7f, "grad[%d] should be 0 when upstream=0", i );
}

Test( layers, sigmoid_backward_gradient_positive )
{
  /* σ'(x) = σ(x)·(1-σ(x)) > 0 for all x — sigmoid is monotone increasing */
  int shape[] = { 1, 4 };
  _tensor_t *input = tensor_zeros( &batch_arena, 2, shape );
  input->data[0] = -5.0f;
  input->data[1] = -1.0f;
  input->data[2] = 1.0f;
  input->data[3] = 5.0f;

  _layer_t *sigmoid = layer_create_sigmoid( &param_arena );
  sigmoid->forward( &batch_arena, sigmoid, input );

  _tensor_t *upstream = tensor_ones( &batch_arena, 2, shape );
  _tensor_t *grad = sigmoid->backward( &batch_arena, sigmoid, upstream );

  for ( int i = 0; i < grad->size; i++ ) cr_assert( grad->data[i] > 0.0f, "σ'(x) must be positive for all x, got %.6f at i=%d", grad->data[i], i );
}

Test( layers, dense_sigmoid_stacked_forward_shape )
{
  /* Dense(4→2) → Sigmoid → output shape [1, 2] */
  _layer_t *dense = layer_create_dense( &param_arena, 4, 2 );
  _layer_t *sigmoid = layer_create_sigmoid( &param_arena );

  int input_shape[] = { 1, 4 };
  _tensor_t *input = tensor_ones( &batch_arena, 2, input_shape );
  _tensor_t *dense_out = dense->forward( &batch_arena, dense, input );
  _tensor_t *sigmoid_out = sigmoid->forward( &batch_arena, sigmoid, dense_out );

  cr_assert_not_null( sigmoid_out );
  cr_assert_eq( sigmoid_out->shape[0], 1 );
  cr_assert_eq( sigmoid_out->shape[1], 2 );
}

Test( layers, dense_sigmoid_stacked_backward_chain_rule )
{
  /* Verify gradients flow through both layers in sequence.
     If sigmoid backward ignores upstream, dense->weights->gradients
     would all be zero regardless of input.                          */
  _layer_t *dense = layer_create_dense( &param_arena, 2, 2 );
  _layer_t *sigmoid = layer_create_sigmoid( &param_arena );

  /* Identity-ish weights for predictable values */
  for ( int i = 0; i < dense->weights->size; i++ ) dense->weights->data[i] = 0.5f;

  int input_shape[] = { 1, 2 };
  _tensor_t *input = tensor_ones( &batch_arena, 2, input_shape );

  _tensor_t *dense_out = dense->forward( &batch_arena, dense, input );
  sigmoid->forward( &batch_arena, sigmoid, dense_out );

  int upstream_shape[] = { 1, 2 };
  _tensor_t *upstream = tensor_ones( &batch_arena, 2, upstream_shape );
  _tensor_t *sig_grad = sigmoid->backward( &batch_arena, sigmoid, upstream );
  dense->backward( &batch_arena, dense, sig_grad );

  /* Dense weight gradients must be nonzero — gradient flowed through */
  int all_zero = 1;
  for ( int i = 0; i < dense->weights->size; i++ )
    if ( fabsf( dense->weights->gradients[i] ) > 1e-7f )
    {
      all_zero = 0;
      break;
    }

  cr_assert( !all_zero, "Dense weight gradients should be nonzero after sigmoid backward" );
}
