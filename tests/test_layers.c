#include "ds_arena.h"
#include <criterion/criterion.h>
#include <criterion/internal/test.h>

static _ds_arena_t_ arena;

void setup( void )
{
  arena = ds_arena_new( 0 );
}

void teardown( void )
{
  ds_arena_destroy( &arena );
}

TestSuite( layer, .init = setup, .fini = teardown );
