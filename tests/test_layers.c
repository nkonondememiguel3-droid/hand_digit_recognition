#include "ds_arena.h"
#include "layers.h"
#include "tensor.h"
#include <criterion/criterion.h>
#include <criterion/internal/assert.h>
#include <criterion/internal/test.h>

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

// Dense tests
Test( layers, dense_output_shape )
{
  int input_shape[] = { 4, 128 };
  _tensor_t *input = tensor_zeros( &batch_arena, 2, input_shape );

  _layer_t *dense = layer_create_dense( &param_arena, 128, 64 );
  _tensor_t *out = dense->forward( &param_arena, dense, input );

  cr_assert_eq( out->shape[0], 4 );
  cr_assert_eq( out->shape[1], 64 );
}
