#include "ds_arena.h"
#include "tensor.h"
#include <criterion/internal/assert.h>
#include <criterion/internal/test.h>
#include <math.h>
#include <stdbool.h>

static _ds_arena_t_ arena;
void setup( void )
{
  arena = ds_arena_new( 0 );
}
void teardown( void )
{
  ds_arena_destroy( &arena );
}
TestSuite( tensor, .init = setup, .fini = teardown );

Test( tensor, create_shape_and_size )
{
  int shape[] = { 2, 3, 4, 4 };
  _tensor_t *t = tensor_create( &arena, 4, shape, false );

  cr_assert_not_null( t );
  cr_assert_eq( t->dimension, 4 );
  cr_assert_eq( t->shape[0], 2 );
  cr_assert_eq( t->shape[1], 3 );
  cr_assert_eq( t->shape[2], 4 );
  cr_assert_eq( t->shape[3], 4 );
  cr_assert_eq( t->size, 96 ); // 2x3x4x4
}

Test( tensor, zeros_initialized )
{
  int shape[] = { 4, 4 };

  _tensor_t *t = tensor_zeros( &arena, 2, shape );

  for ( int i = 0; i < t->size; i++ ) cr_assert_float_eq( t->data[i], 0.0f, 1e-7f );
}

Test( tensor, ones_initialized )
{
  int shape[] = { 4, 4 };

  _tensor_t *t = tensor_ones( &arena, 2, shape );

  for ( int i = 0; i < t->size; i++ ) cr_assert_float_eq( t->data[i], 1.0f, 1e-7f );
}

Test( tensor, strides_row_major_4d )
{
  // shape (2, 3, 4, 4) -> strides (48, 16, 4, 1)
  int shape[] = { 2, 3, 4, 4 };

  _tensor_t *t = tensor_create( &arena, 4, shape, false );

  cr_assert_not_null( t );

  cr_assert_eq( t->strides[0], 48 );
  cr_assert_eq( t->strides[1], 16 );
  cr_assert_eq( t->strides[2], 4 );
  cr_assert_eq( t->strides[3], 1 );
}

Test( tensor, strides_2d )
{
  int shape[] = { 5, 7 };

  _tensor_t *t = tensor_create( &arena, 2, shape, false );

  cr_assert_not_null( t );
  cr_assert_eq( t->strides[0], 7 );
  cr_assert_eq( t->strides[1], 1 );
}

Test( tensor, indexing_macro_correct )
{
  int shape[] = { 2, 3, 4, 4 };
  _tensor_t *t = tensor_zeros( &arena, 4, shape );

  // [1][2][3][1] -> 1*48 + 2*16 + 3*4 + 1 = 93
  T4( t, 1, 2, 3, 1 ) = 7.5f;

  cr_assert_float_eq( t->data[93], 7.5f, 1e-7f );
  cr_assert_float_eq( T4( t, 1, 2, 3, 1 ), 7.5f, 1e-7f );
}

Test( tensor, gradients_allocated_when_requried )
{
  int shape[] = { 4 };

  _tensor_t *t = tensor_create( &arena, 1, shape, true );

  cr_assert_not_null( t );
  cr_assert_not_null( t->gradients );

  for ( int i = 0; i < t->size; i++ ) cr_assert_float_eq( t->gradients[i], 0.0f, 1e-7f );
}

Test( tensor, gradients_nulll_when_not_required )
{
  int shape[] = { 4 };

  _tensor_t *t = tensor_create( &arena, 1, shape, false );

  cr_assert_not_null( t );
  cr_assert_null( t->gradients );
}

Test( tensor, zero_gradients_resets_gradients )
{
  int shape[] = { 3 };

  _tensor_t *t = tensor_create( &arena, 1, shape, true );

  t->gradients[0] = 1.5f;
  t->gradients[1] = -3.2f;
  t->gradients[2] = 0.7f;

  tensor_zero_gradients( t );

  for ( int i = 0; i < t->size; i++ ) cr_assert_float_eq( t->gradients[i], 0.0f, 1e-7f );
}

Test( tensor, clip_gradients_limits_magnitude )
{
  int shape[] = { 4 };

  _tensor_t *t = tensor_create( &arena, 1, shape, true );

  t->gradients[0] = 10.0f;
  t->gradients[1] = -10.0f;
  t->gradients[2] = 0.5f;
  t->gradients[3] = -0.5f;

  tensor_clip_gradient( t, 1.0f );

  cr_assert_float_eq( t->gradients[0], 1.0f, 1e-7f );
  cr_assert_float_eq( t->gradients[1], -1.0f, 1e-7f );
  cr_assert_float_eq( t->gradients[2], 0.5f, 1e-7f );
  cr_assert_float_eq( t->gradients[3], -0.5f, 1e-7f );
}

// Random Normal (He init)
Test( tensor, normal_mean_near_target )
{
  int shape[] = { 5000 };
  _tensor_t *t = tensor_random_normal( &arena, 1, shape, 2.0f, 1.0f );

  float mean = 0.0f;
  for ( int i = 0; i < t->size; i++ ) mean += t->data[i];
  mean /= t->size;

  cr_assert( fabsf( mean - 2.0f ) < 0.1f, "Mean %.4f should be near 2.0", mean );
}

Test( tensor, normal_std_matches_he_init )
{
  // He init for C_in=3, k=3: std = sqrt(2/(3*3*3))
  int C_in = 3, k = 3;
  float expected_std = sqrtf( 2.0f / ( C_in * k * k ) );

  int shape[] = { 10000 };
  _tensor_t *t = tensor_random_normal( &arena, 1, shape, 0.0f, expected_std );

  float mean = 0.0f;
  for ( int i = 0; i < t->size; i++ ) mean += t->data[i];
  mean /= t->size;

  float var = 0.0f;
  for ( int i = 0; i < t->size; i++ )
  {
    float d = t->data[i] - mean;
    var += d * d;
  }
  var /= t->size;

  cr_assert( fabsf( sqrtf( var ) - expected_std ) < 0.02f, "Std %.4f should be near %.4f", sqrtf( var ), expected_std );
}

// Arena checkpoint/reset interaction
Test( tensor, checkpoint_reset_reclaims_memory )
{
  _ds_arena_checkpoint_t_ cp = ds_arena_checkpoint( &arena );

  int shape[] = { 100, 100 }; // large allocation
  _tensor_t *t = tensor_ones( &arena, 2, shape );
  cr_assert_eq( t->data[0], 1.0f );

  size_t used_before = arena.head->chunk_size_used;
  cr_assert( used_before > 0 );

  ds_arena_reset_to( &arena, cp );

  // After reset, used bytes should be back to checkpoint
  size_t used_after = arena.head ? arena.head->chunk_size_used : 0;
  cr_assert_eq( used_after, cp.checkpoint_size_used );
}
