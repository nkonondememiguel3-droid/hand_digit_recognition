#include "ds_arena.h"
#include "loses.h"
#include "tensor.h"
#include <criterion/criterion.h>
#include <criterion/internal/assert.h>
#include <criterion/internal/test.h>
#include <criterion/logging.h>
#include <math.h>

static _ds_arena_t_ arena;

void setup( void )
{
  arena = ds_arena_new( 0 );
}
void teardown( void )
{
  ds_arena_destroy( &arena );
}

TestSuite( loss, .init = setup, .fini = teardown );

/* Build a one-hot label tensor [batch, classes] */
static _tensor_t *make_one_hot( _ds_arena_t_ *a, int batch, int classes, const int *labels )
{
  int shape[] = { batch, classes };
  _tensor_t *t = tensor_zeros( a, 2, shape );
  for ( int i = 0; i < batch; i++ ) T2( t, i, labels[i] ) = 1.0f;
  return t;
}

/* Build a logits tensor [batch, classes] with given flat values */
static _tensor_t *make_logits( _ds_arena_t_ *a, int batch, int classes, const float *vals )
{
  int shape[] = { batch, classes };
  _tensor_t *t = tensor_zeros( a, 2, shape );
  for ( int i = 0; i < batch * classes; i++ ) t->data[i] = vals[i];
  return t;
}

Test( loss, sce_result_loss_not_null )
{
  float logits_data[] = { 1.0f, 2.0f, 3.0f };
  int label[] = { 2 };
  _tensor_t *logits = make_logits( &arena, 1, 3, logits_data );
  _tensor_t *labels = make_one_hot( &arena, 1, 3, label );

  _loss_result_t r = loss_softmax_cross_entropy( &arena, logits, labels );
  cr_assert_not_null( r.loss );
}

Test( loss, sce_result_gradients_not_null )
{
  float logits_data[] = { 1.0f, 2.0f, 3.0f };
  int label[] = { 2 };
  _tensor_t *logits = make_logits( &arena, 1, 3, logits_data );
  _tensor_t *labels = make_one_hot( &arena, 1, 3, label );

  _loss_result_t r = loss_softmax_cross_entropy( &arena, logits, labels );
  cr_assert_not_null( r.gradients );
}

Test( loss, sce_loss_is_scalar )
{
  float logits_data[] = { 1.0f, 2.0f, 3.0f };
  int label[] = { 2 };
  _tensor_t *logits = make_logits( &arena, 1, 3, logits_data );
  _tensor_t *labels = make_one_hot( &arena, 1, 3, label );

  _loss_result_t r = loss_softmax_cross_entropy( &arena, logits, labels );
  cr_assert_eq( r.loss->dimension, 1 );
  cr_assert_eq( r.loss->shape[0], 1 );
  cr_assert_eq( r.loss->size, 1 );
}

Test( loss, sce_gradient_shape_matches_logits )
{
  float logits_data[] = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f };
  int labels[] = { 0, 1 };
  _tensor_t *logits = make_logits( &arena, 2, 3, logits_data );
  _tensor_t *labs = make_one_hot( &arena, 2, 3, labels );

  _loss_result_t r = loss_softmax_cross_entropy( &arena, logits, labs );
  cr_assert_eq( r.gradients->dimension, 2 );
  cr_assert_eq( r.gradients->shape[0], 2 );
  cr_assert_eq( r.gradients->shape[1], 3 );
}

Test( loss, sce_loss_is_nonnegative )
{
  float logits_data[] = { 2.0f, 1.0f, 0.1f };
  int label[] = { 0 };
  _tensor_t *logits = make_logits( &arena, 1, 3, logits_data );
  _tensor_t *labels = make_one_hot( &arena, 1, 3, label );

  _loss_result_t r = loss_softmax_cross_entropy( &arena, logits, labels );
  cr_assert( r.loss->data[0] >= 0.0f, "Cross-entropy loss must be >= 0, got %.6f", r.loss->data[0] );
}

Test( loss, sce_loss_perfect_prediction_near_zero )
{
  /* Very large logit for correct class → softmax ≈ 1 → loss ≈ 0 */
  float logits_data[] = { 100.0f, -100.0f, -100.0f };
  int label[] = { 0 };
  _tensor_t *logits = make_logits( &arena, 1, 3, logits_data );
  _tensor_t *labels = make_one_hot( &arena, 1, 3, label );

  _loss_result_t r = loss_softmax_cross_entropy( &arena, logits, labels );
  cr_assert( r.loss->data[0] < 0.01f, "Loss should be near 0 for perfect prediction, got %.6f", r.loss->data[0] );
}

Test( loss, sce_loss_worst_prediction_is_large )
{
  float logits_data[] = { -100.0f, -100.0f, 100.0f };
  int label[] = { 0 };
  _tensor_t *logits = make_logits( &arena, 1, 3, logits_data );
  _tensor_t *labels = make_one_hot( &arena, 1, 3, label );

  _loss_result_t r = loss_softmax_cross_entropy( &arena, logits, labels );

  float max_clamped_loss = -logf( 1e-7f );

  cr_assert( r.loss->data[0] > 10.0f, "Loss should be large for wrong prediction, got %.4f", r.loss->data[0] );

  cr_assert_float_eq( r.loss->data[0], max_clamped_loss, 1e-3f,
                      "Confidently wrong prediction should hit epsilon clamp: "
                      "expected %.4f, got %.4f",
                      max_clamped_loss, r.loss->data[0] );
}

Test( loss, sce_loss_uniform_logits_equals_log_classes )
{
  /* When all logits are equal, softmax gives uniform distribution p=1/C.
     Loss = -log(1/C) = log(C)                                          */
  int C = 5;
  float logits_data[] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
  int label[] = { 2 };
  _tensor_t *logits = make_logits( &arena, 1, C, logits_data );
  _tensor_t *labels = make_one_hot( &arena, 1, C, label );

  _loss_result_t r = loss_softmax_cross_entropy( &arena, logits, labels );
  float expected = logf( (float)C );

  cr_assert_float_eq( r.loss->data[0], expected, 1e-4f, "Uniform logits: loss should be log(%d)=%.4f, got %.4f", C, expected, r.loss->data[0] );
}

Test( loss, sce_loss_is_mean_over_batch )
{
  /* Two identical samples — mean loss = single-sample loss */
  float logits_data[] = { 2.0f, 1.0f, 0.5f, 2.0f, 1.0f, 0.5f };
  int labels[] = { 0, 0 };
  _tensor_t *logits2 = make_logits( &arena, 2, 3, logits_data );
  _tensor_t *labs2 = make_one_hot( &arena, 2, 3, labels );

  float single_data[] = { 2.0f, 1.0f, 0.5f };
  int single_label[] = { 0 };
  _tensor_t *logits1 = make_logits( &arena, 1, 3, single_data );
  _tensor_t *labs1 = make_one_hot( &arena, 1, 3, single_label );

  _loss_result_t r1 = loss_softmax_cross_entropy( &arena, logits1, labs1 );
  _loss_result_t r2 = loss_softmax_cross_entropy( &arena, logits2, labs2 );

  cr_assert_float_eq( r1.loss->data[0], r2.loss->data[0], 1e-4f, "Mean over batch=2 with identical samples should equal batch=1 loss" );
}

Test( loss, sce_loss_numerical_stability_large_logits )
{
  /* Should not produce NaN or Inf with large logit values */
  float logits_data[] = { 1000.0f, 999.0f, 998.0f };
  int label[] = { 0 };
  _tensor_t *logits = make_logits( &arena, 1, 3, logits_data );
  _tensor_t *labels = make_one_hot( &arena, 1, 3, label );

  _loss_result_t r = loss_softmax_cross_entropy( &arena, logits, labels );
  cr_assert( isfinite( r.loss->data[0] ), "Loss should be finite for large logits, got %f", r.loss->data[0] );
}

Test( loss, sce_loss_numerical_stability_negative_logits )
{
  float logits_data[] = { -1000.0f, -999.0f, -998.0f };
  int label[] = { 2 };
  _tensor_t *logits = make_logits( &arena, 1, 3, logits_data );
  _tensor_t *labels = make_one_hot( &arena, 1, 3, label );

  _loss_result_t r = loss_softmax_cross_entropy( &arena, logits, labels );
  cr_assert( isfinite( r.loss->data[0] ), "Loss should be finite for large negative logits, got %f", r.loss->data[0] );
}

Test( loss, sce_gradient_is_p_minus_y )
{
  /* For a single sample:  ∂L/∂z[j] = (p[j] - y[j]) / batch_size
   * With logits [0,0,0], softmax gives p=[1/3, 1/3, 1/3], label=class 1:
   *   grad[0] = (1/3 - 0) / 1 =  1/3
   *   grad[1] = (1/3 - 1) / 1 = -2/3
   *   grad[2] = (1/3 - 0) / 1 =  1/3                               */
  float logits_data[] = { 0.0f, 0.0f, 0.0f };
  int label[] = { 1 };
  _tensor_t *logits = make_logits( &arena, 1, 3, logits_data );
  _tensor_t *labels = make_one_hot( &arena, 1, 3, label );

  _loss_result_t r = loss_softmax_cross_entropy( &arena, logits, labels );

  cr_assert_float_eq( T2( r.gradients, 0, 0 ), 1.0f / 3.0f, 1e-5f );
  cr_assert_float_eq( T2( r.gradients, 0, 1 ), -2.0f / 3.0f, 1e-5f );
  cr_assert_float_eq( T2( r.gradients, 0, 2 ), 1.0f / 3.0f, 1e-5f );
}

Test( loss, sce_gradient_sums_to_zero_per_sample )
{
  /* Σⱼ (p[j] - y[j]) = Σⱼp[j] - Σⱼy[j] = 1 - 1 = 0 */
  float logits_data[] = { 1.5f, -0.5f, 2.0f, 0.3f };
  int label[] = { 2 };
  _tensor_t *logits = make_logits( &arena, 1, 4, logits_data );
  _tensor_t *labels = make_one_hot( &arena, 1, 4, label );

  _loss_result_t r = loss_softmax_cross_entropy( &arena, logits, labels );

  float grad_sum = 0.0f;
  for ( int j = 0; j < 4; j++ ) grad_sum += T2( r.gradients, 0, j );

  cr_assert_float_eq( grad_sum, 0.0f, 1e-5f, "Gradient should sum to 0 per sample, got %.6f", grad_sum );
}

Test( loss, sce_gradient_correct_class_is_negative )
{
  /* For the correct class c: grad[c] = p[c] - 1 < 0 (since p[c] < 1) */
  float logits_data[] = { 0.5f, 1.5f, -0.5f };
  int label[] = { 1 }; /* correct class is 1 */
  _tensor_t *logits = make_logits( &arena, 1, 3, logits_data );
  _tensor_t *labels = make_one_hot( &arena, 1, 3, label );

  _loss_result_t r = loss_softmax_cross_entropy( &arena, logits, labels );

  cr_assert( T2( r.gradients, 0, 1 ) < 0.0f, "Gradient at correct class should be negative (p-1), got %.6f", T2( r.gradients, 0, 1 ) );
}

Test( loss, sce_gradient_wrong_classes_are_positive )
{
  /* For wrong classes j≠c: grad[j] = p[j] > 0 */
  float logits_data[] = { 0.5f, 1.5f, -0.5f };
  int label[] = { 1 };
  _tensor_t *logits = make_logits( &arena, 1, 3, logits_data );
  _tensor_t *labels = make_one_hot( &arena, 1, 3, label );

  _loss_result_t r = loss_softmax_cross_entropy( &arena, logits, labels );

  cr_assert( T2( r.gradients, 0, 0 ) > 0.0f, "Gradient at wrong class 0 should be positive, got %.6f", T2( r.gradients, 0, 0 ) );
  cr_assert( T2( r.gradients, 0, 2 ) > 0.0f, "Gradient at wrong class 2 should be positive, got %.6f", T2( r.gradients, 0, 2 ) );
}

Test( loss, sce_gradient_normalised_by_batch )
{
  /* Gradient should be divided by batch_size.
     Single sample gradient at class 0 with uniform logits = (1/3-0)/1 = 1/3.
     With batch=3 identical samples = (1/3-0)/3 = 1/9.                */
  float logits_data[] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
  int labels[] = { 1, 1, 1 };
  _tensor_t *logits = make_logits( &arena, 3, 3, logits_data );
  _tensor_t *labs = make_one_hot( &arena, 3, 3, labels );

  _loss_result_t r = loss_softmax_cross_entropy( &arena, logits, labs );

  cr_assert_float_eq( T2( r.gradients, 0, 0 ), 1.0f / 9.0f, 1e-5f, "Gradient at wrong class should be 1/9 for batch=3, got %.6f",
                      T2( r.gradients, 0, 0 ) );
  cr_assert_float_eq( T2( r.gradients, 0, 1 ), -2.0f / 9.0f, 1e-5f, "Gradient at correct class should be -2/9 for batch=3, got %.6f",
                      T2( r.gradients, 0, 1 ) );
}

Test( loss, sce_gradient_no_nan_or_inf )
{
  float logits_data[] = { 3.0f, 1.0f, 0.2f, -1.0f, 2.5f };
  int label[] = { 4 };
  _tensor_t *logits = make_logits( &arena, 1, 5, logits_data );
  _tensor_t *labels = make_one_hot( &arena, 1, 5, label );

  _loss_result_t r = loss_softmax_cross_entropy( &arena, logits, labels );

  for ( int j = 0; j < 5; j++ ) cr_assert( isfinite( T2( r.gradients, 0, j ) ), "Gradient[%d] = %f is not finite", j, T2( r.gradients, 0, j ) );
}

Test( loss, sce_null_logits_returns_null_loss )
{
  int label[] = { 0 };
  int shape[] = { 1, 3 };
  _tensor_t *labels = make_one_hot( &arena, 1, 3, label );
  (void)shape;

  _loss_result_t r = loss_softmax_cross_entropy( &arena, NULL, labels );
  cr_assert_null( r.loss );
  cr_assert_null( r.gradients );
}

Test( loss, sce_null_labels_returns_null_loss )
{
  float logits_data[] = { 1.0f, 2.0f, 3.0f };
  _tensor_t *logits = make_logits( &arena, 1, 3, logits_data );

  _loss_result_t r = loss_softmax_cross_entropy( &arena, logits, NULL );
  cr_assert_null( r.loss );
  cr_assert_null( r.gradients );
}

Test( loss, sce_shape_mismatch_returns_null_loss )
{
  float logits_data[] = { 1.0f, 2.0f, 3.0f };
  _tensor_t *logits = make_logits( &arena, 1, 3, logits_data );

  /* Labels have wrong class count */
  float wrong_data[] = { 1.0f, 0.0f };
  int wrong_shape[] = { 1, 2 };
  _tensor_t *labels = tensor_zeros( &arena, 2, wrong_shape );
  labels->data[0] = 1.0f;
  (void)wrong_data;

  _loss_result_t r = loss_softmax_cross_entropy( &arena, logits, labels );
  cr_assert_null( r.loss );
  cr_assert_null( r.gradients );
}

Test( loss, sce_1d_logits_returns_null_loss )
{
  /* loss expects 2D [batch, classes], not 1D */
  int shape[] = { 3 };
  _tensor_t *logits = tensor_zeros( &arena, 1, shape );
  _tensor_t *labels = tensor_zeros( &arena, 1, shape );

  _loss_result_t r = loss_softmax_cross_entropy( &arena, logits, labels );
  cr_assert_null( r.loss );
}

Test( loss, sce_mnist_scale_batch64_classes10 )
{
  /* batch=64, classes=10 — simulate real training dimensions */
  int batch = 64, classes = 10;
  int shape[] = { batch, classes };

  _tensor_t *logits = tensor_zeros( &arena, 2, shape );
  _tensor_t *labels = tensor_zeros( &arena, 2, shape );

  /* Fill logits with small random-ish values, one-hot labels */
  for ( int i = 0; i < batch; i++ )
  {
    int correct = i % classes;
    T2( labels, i, correct ) = 1.0f;
    for ( int j = 0; j < classes; j++ ) T2( logits, i, j ) = ( j == correct ) ? 0.5f : -0.1f;
  }

  _loss_result_t r = loss_softmax_cross_entropy( &arena, logits, labels );

  cr_assert_not_null( r.loss );
  cr_assert_not_null( r.gradients );
  cr_assert( isfinite( r.loss->data[0] ), "Loss should be finite at MNIST scale, got %f", r.loss->data[0] );
  cr_assert( r.loss->data[0] > 0.0f, "Loss should be positive, got %f", r.loss->data[0] );

  /* All gradients must be finite */
  for ( int i = 0; i < r.gradients->size; i++ ) cr_assert( isfinite( r.gradients->data[i] ), "Gradient[%d] is not finite at MNIST scale", i );
}
