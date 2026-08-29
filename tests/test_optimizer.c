#include "ds_arena.h"
#include "layers.h"
#include "networkd.h"
#include "optimizer.h"
#include "tensor.h"
#include <criterion/criterion.h>
#include <criterion/internal/assert.h>
#include <criterion/internal/test.h>
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

TestSuite( optimizer, .init = setup, .fini = teardown );

/* dense -> relu -> dense -> relu.
 *
 * Two parameterized layers, and crucially a parameterless layer in last
 * position: optimizer state is indexed over parameterized layers only, so a
 * trailing activation used to write one slot past the end of every state
 * array and corrupt whatever the arena had placed after it.
 */
static _network_t *make_network_ending_in_activation( _ds_arena_t_ *a )
{
  _network_t *net = network_create( a );
  network_add_layer( a, net, layer_create_dense( a, 4, 3 ) );
  network_add_layer( a, net, layer_create_relu( a ) );
  network_add_layer( a, net, layer_create_dense( a, 3, 2 ) );
  network_add_layer( a, net, layer_create_relu( a ) );
  return net;
}

Test( optimizer, sgd_counts_only_parameterized_layers )
{
  _network_t *net = make_network_ending_in_activation( &arena );
  _optimizer_t *opt = optimizer_create_sgd( &arena, net, optimizer_sgd_defaults() );

  cr_assert_eq( opt->num_layers, 2, "only the two dense layers own optimizer state" );
}

Test( optimizer, sgd_velocity_slots_survive_a_trailing_activation )
{
  _network_t *net = make_network_ending_in_activation( &arena );
  _optimizer_t *opt = optimizer_create_sgd( &arena, net, optimizer_sgd_defaults() );

  /* Regression: the trailing relu wrote NULL at index num_layers, which
     landed on velocities_b[0] and silently disabled bias momentum. */
  for ( int i = 0; i < opt->num_layers; i++ )
  {
    cr_assert_not_null( opt->sgd_velocities_w[i], "weight velocities missing for parameterized layer %d", i );
    cr_assert_not_null( opt->sgd_velocities_b[i], "bias velocities missing for parameterized layer %d", i );
  }
}

Test( optimizer, adam_state_slots_survive_a_trailing_activation )
{
  _network_t *net = make_network_ending_in_activation( &arena );
  _optimizer_t *opt = optimizer_create_adam( &arena, net, optimizer_adam_defaults() );

  for ( int i = 0; i < opt->num_layers; i++ )
  {
    cr_assert_not_null( opt->adam_states[i].weights.m, "adam m missing for layer %d", i );
    cr_assert_not_null( opt->adam_states[i].weights.v, "adam v missing for layer %d", i );
    cr_assert_not_null( opt->adam_states[i].bias.m, "adam bias m missing for layer %d", i );
    cr_assert_not_null( opt->adam_states[i].bias.v, "adam bias v missing for layer %d", i );
  }
}

/* One gradient-descent step on a layer with a known gradient must move the
   parameters downhill by exactly learning_rate * gradient. */
Test( optimizer, sgd_step_moves_parameters_downhill )
{
  _network_t *net = network_create( &arena );
  network_add_layer( &arena, net, layer_create_dense( &arena, 2, 1 ) );
  network_add_layer( &arena, net, layer_create_relu( &arena ) );

  _layer_t *dense = net->head->layer;

  _sgd_config_t cfg = optimizer_sgd_defaults();
  cfg.momentum = 0.0f;
  cfg.weight_decay = 0.0f;
  _optimizer_t *opt = optimizer_create_sgd( &arena, net, cfg );

  float before = dense->weights->data[0];
  dense->weights->gradients[0] = 2.0f;

  optimizer_step( opt, net );

  cr_assert_float_eq( dense->weights->data[0], before - cfg.learning_rate * 2.0f, 1e-6f, "plain SGD should step by learning_rate * gradient" );
}

Test( optimizer, adam_step_moves_parameters_downhill )
{
  _network_t *net = network_create( &arena );
  network_add_layer( &arena, net, layer_create_dense( &arena, 2, 1 ) );
  network_add_layer( &arena, net, layer_create_relu( &arena ) );

  _layer_t *dense = net->head->layer;
  _optimizer_t *opt = optimizer_create_adam( &arena, net, optimizer_adam_defaults() );

  float before = dense->weights->data[0];
  dense->weights->gradients[0] = 2.0f;

  optimizer_step( opt, net );

  cr_assert( dense->weights->data[0] < before, "a positive gradient must decrease the parameter" );
  cr_assert( isfinite( dense->weights->data[0] ), "Adam step produced a non-finite parameter" );
}
