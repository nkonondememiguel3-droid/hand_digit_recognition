#include "ds_arena.h"
#include "layers.h"
#include "loses.h"
#include "networkd.h"
#include "tensor.h"
#include <criterion/criterion.h>
#include <criterion/internal/assert.h>
#include <criterion/internal/test.h>
#include <math.h>
#include <stdbool.h>

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

TestSuite( network, .init = setup, .fini = teardown );

/* Build a one-hot label tensor [batch, classes] */
static _tensor_t *make_one_hot( _ds_arena_t_ *a, int batch, int classes, const int *labels )
{
  int shape[] = { batch, classes };
  _tensor_t *t = tensor_zeros( a, 2, shape );
  for ( int i = 0; i < batch; i++ ) T2( t, i, labels[i] ) = 1.0f;
  return t;
}

/* Build a simple Dense→Sigmoid→Dense network */
static _network_t *make_simple_network( _ds_arena_t_ *pa, int in, int hidden, int out )
{
  _network_t *net = network_create( pa );
  network_add_layer( pa, net, layer_create_dense( pa, in, hidden ) );
  network_add_layer( pa, net, layer_create_sigmoid( pa ) );
  network_add_layer( pa, net, layer_create_dense( pa, hidden, out ) );
  return net;
}

Test( network, create_not_null )
{
  _network_t *net = network_create( &param_arena );
  cr_assert_not_null( net );
}

Test( network, create_empty_head_tail_null )
{
  _network_t *net = network_create( &param_arena );
  cr_assert_null( net->head );
  cr_assert_null( net->tail );
}

Test( network, create_num_layers_zero )
{
  _network_t *net = network_create( &param_arena );
  cr_assert_eq( net->number_layers, 0 );
}

Test( network, create_forward_backward_fn_not_null )
{
  _network_t *net = network_create( &param_arena );
  cr_assert_not_null( net->forward );
  cr_assert_not_null( net->backward );
}

Test( network, add_one_layer_sets_head_and_tail )
{
  _network_t *net = network_create( &param_arena );
  _layer_t *dense = layer_create_dense( &param_arena, 4, 2 );
  network_add_layer( &param_arena, net, dense );

  cr_assert_not_null( net->head );
  cr_assert_not_null( net->tail );
  cr_assert_eq( net->head, net->tail ); /* single node */
}

Test( network, add_one_layer_increments_count )
{
  _network_t *net = network_create( &param_arena );
  network_add_layer( &param_arena, net, layer_create_dense( &param_arena, 4, 2 ) );
  cr_assert_eq( net->number_layers, 1 );
}

Test( network, add_three_layers_correct_count )
{
  _network_t *net = make_simple_network( &param_arena, 4, 8, 2 );
  cr_assert_eq( net->number_layers, 3 );
}

Test( network, linked_list_forward_links_correct )
{
  _network_t *net = network_create( &param_arena );
  _layer_t *d1 = layer_create_dense( &param_arena, 4, 8 );
  _layer_t *s1 = layer_create_sigmoid( &param_arena );
  _layer_t *d2 = layer_create_dense( &param_arena, 8, 2 );
  network_add_layer( &param_arena, net, d1 );
  network_add_layer( &param_arena, net, s1 );
  network_add_layer( &param_arena, net, d2 );

  cr_assert_eq( net->head->layer, d1 );
  cr_assert_eq( net->head->next->layer, s1 );
  cr_assert_eq( net->head->next->next->layer, d2 );
  cr_assert_null( net->head->next->next->next );
}

Test( network, linked_list_backward_links_correct )
{
  _network_t *net = network_create( &param_arena );
  _layer_t *d1 = layer_create_dense( &param_arena, 4, 8 );
  _layer_t *s1 = layer_create_sigmoid( &param_arena );
  _layer_t *d2 = layer_create_dense( &param_arena, 8, 2 );
  network_add_layer( &param_arena, net, d1 );
  network_add_layer( &param_arena, net, s1 );
  network_add_layer( &param_arena, net, d2 );

  cr_assert_eq( net->tail->layer, d2 );
  cr_assert_eq( net->tail->prev->layer, s1 );
  cr_assert_eq( net->tail->prev->prev->layer, d1 );
  cr_assert_null( net->tail->prev->prev->prev );
}

Test( network, head_prev_is_null )
{
  _network_t *net = make_simple_network( &param_arena, 4, 8, 2 );
  cr_assert_null( net->head->prev );
}

Test( network, tail_next_is_null )
{
  _network_t *net = make_simple_network( &param_arena, 4, 8, 2 );
  cr_assert_null( net->tail->next );
}

Test( network, skip_source_null_by_default )
{
  _network_t *net = make_simple_network( &param_arena, 4, 4, 4 );
  _network_node_t *node = net->head;
  while ( node )
  {
    cr_assert_null( node->skip_source, "skip_source should be NULL for nodes without skip connections" );
    node = node->next;
  }
}

Test( network, add_skip_sets_destination_source )
{
  _network_t *net = network_create( &param_arena );
  _layer_t *d1 = layer_create_dense( &param_arena, 4, 4 );
  _layer_t *d2 = layer_create_dense( &param_arena, 4, 4 );
  _layer_t *d3 = layer_create_dense( &param_arena, 4, 4 );
  network_add_layer( &param_arena, net, d1 );
  network_add_layer( &param_arena, net, d2 );
  network_add_layer( &param_arena, net, d3 );

  _network_node_t *node1 = net->head;
  _network_node_t *node3 = net->tail;
  network_add_skip( node1, node3 );

  cr_assert_eq( node3->skip_source, node1 );
}

Test( network, add_skip_null_source_is_safe )
{
  _network_t *net = network_create( &param_arena );
  _layer_t *d1 = layer_create_dense( &param_arena, 4, 2 );
  network_add_layer( &param_arena, net, d1 );
  /* Should not crash — just print an error */
  network_add_skip( NULL, net->head );
  cr_assert_null( net->head->skip_source );
}

Test( network, add_skip_same_node_is_safe )
{
  _network_t *net = network_create( &param_arena );
  _layer_t *d1 = layer_create_dense( &param_arena, 4, 2 );
  network_add_layer( &param_arena, net, d1 );
  network_add_skip( net->head, net->head );
  /* Should print error and leave skip_source unchanged (still NULL) */
  cr_assert_null( net->head->skip_source );
}

Test( network, forward_output_not_null )
{
  _network_t *net = make_simple_network( &param_arena, 4, 8, 3 );
  int input_shape[] = { 2, 4 };
  _tensor_t *input = tensor_ones( &batch_arena, 2, input_shape );
  _tensor_t *output = net->forward( net, &batch_arena, input );
  cr_assert_not_null( output );
}

Test( network, forward_output_shape )
{
  /* Dense(784→128) → Sigmoid → Dense(128→10) */
  _network_t *net = make_simple_network( &param_arena, 784, 128, 10 );
  int input_shape[] = { 4, 784 }; /* batch=4 */
  _tensor_t *input = tensor_zeros( &batch_arena, 2, input_shape );
  _tensor_t *output = net->forward( net, &batch_arena, input );

  cr_assert_not_null( output );
  cr_assert_eq( output->shape[0], 4 );
  cr_assert_eq( output->shape[1], 10 );
}

Test( network, forward_caches_node_outputs )
{
  _network_t *net = make_simple_network( &param_arena, 4, 8, 3 );
  int input_shape[] = { 1, 4 };
  _tensor_t *input = tensor_ones( &batch_arena, 2, input_shape );
  net->forward( net, &batch_arena, input );

  /* Every node should have a cached output after forward */
  _network_node_t *node = net->head;
  while ( node )
  {
    cr_assert_not_null( node->output, "Node '%s' output should be cached after forward", node->layer->layer_name );
    node = node->next;
  }
}

Test( network, forward_single_dense_layer )
{
  /* Verify a 1-layer network passes through correctly:
     Dense(2→2), W=[[1,0],[0,1]] (identity), b=[0,0], input=[1,2] → [1,2] */
  _network_t *net = network_create( &param_arena );
  _layer_t *dense = layer_create_dense( &param_arena, 2, 2 );
  dense->weights->data[0] = 1.0f;
  dense->weights->data[1] = 0.0f;
  dense->weights->data[2] = 0.0f;
  dense->weights->data[3] = 1.0f;
  dense->bias->data[0] = 0.0f;
  dense->bias->data[1] = 0.0f;
  network_add_layer( &param_arena, net, dense );

  int input_shape[] = { 1, 2 };
  _tensor_t *input = tensor_zeros( &batch_arena, 2, input_shape );
  input->data[0] = 1.0f;
  input->data[1] = 2.0f;

  _tensor_t *output = net->forward( net, &batch_arena, input );
  cr_assert_float_eq( output->data[0], 1.0f, 1e-5f );
  cr_assert_float_eq( output->data[1], 2.0f, 1e-5f );
}

Test( network, forward_null_network_returns_null )
{
  int shape[] = { 1, 4 };
  _tensor_t *input = tensor_ones( &batch_arena, 2, shape );
  _network_t dummy = { 0 };
  dummy.forward = network_create( &param_arena )->forward;
  _tensor_t *out = dummy.forward( NULL, &batch_arena, input );
  cr_assert_null( out );
}

Test( network, forward_null_input_returns_null )
{
  _network_t *net = make_simple_network( &param_arena, 4, 8, 2 );
  _tensor_t *out = net->forward( net, &batch_arena, NULL );
  cr_assert_null( out );
}

Test( network, forward_empty_network_returns_null )
{
  _network_t *net = network_create( &param_arena );
  int shape[] = { 1, 4 };
  _tensor_t *input = tensor_ones( &batch_arena, 2, shape );
  _tensor_t *out = net->forward( net, &batch_arena, input );
  cr_assert_null( out );
}

Test( network, backward_returns_not_null )
{
  _network_t *net = make_simple_network( &param_arena, 4, 8, 3 );
  int input_shape[] = { 2, 4 };
  _tensor_t *input = tensor_ones( &batch_arena, 2, input_shape );
  net->forward( net, &batch_arena, input );

  int labels[] = { 0, 2 };
  _tensor_t *labs = make_one_hot( &batch_arena, 2, 3, labels );
  _tensor_t *logits = net->tail->output;
  _loss_result_t loss = loss_softmax_cross_entropy( &batch_arena, logits, labs );

  network_zero_gradients( net );
  _tensor_t *input_grad = net->backward( net, &batch_arena, loss.gradients );
  cr_assert_not_null( input_grad );
}

Test( network, backward_input_gradient_shape )
{
  _network_t *net = make_simple_network( &param_arena, 4, 8, 3 );
  int batch = 2;
  int input_shape[] = { batch, 4 };
  _tensor_t *input = tensor_ones( &batch_arena, 2, input_shape );
  net->forward( net, &batch_arena, input );

  int labels[] = { 0, 1 };
  _tensor_t *labs = make_one_hot( &batch_arena, batch, 3, labels );
  _loss_result_t r = loss_softmax_cross_entropy( &batch_arena, net->tail->output, labs );

  network_zero_gradients( net );
  _tensor_t *input_grad = net->backward( net, &batch_arena, r.gradients );

  cr_assert_eq( input_grad->shape[0], batch );
  cr_assert_eq( input_grad->shape[1], 4 );
}

Test( network, backward_populates_weight_gradients )
{
  _network_t *net = make_simple_network( &param_arena, 4, 8, 3 );
  int input_shape[] = { 2, 4 };
  _tensor_t *input = tensor_ones( &batch_arena, 2, input_shape );
  net->forward( net, &batch_arena, input );

  int labels[] = { 0, 2 };
  _tensor_t *labs = make_one_hot( &batch_arena, 2, 3, labels );
  _loss_result_t r = loss_softmax_cross_entropy( &batch_arena, net->tail->output, labs );

  network_zero_gradients( net );
  net->backward( net, &batch_arena, r.gradients );

  /* Every dense layer's weights should have nonzero gradients */
  _network_node_t *node = net->head;
  while ( node )
  {
    if ( node->layer->weights )
    {
      int nonzero = 0;
      for ( int i = 0; i < node->layer->weights->size; i++ )
        if ( fabsf( node->layer->weights->gradients[i] ) > 1e-8f )
        {
          nonzero = 1;
          break;
        }
      cr_assert( nonzero, "Layer '%s' weights should have nonzero gradients after backward", node->layer->layer_name );
    }
    node = node->next;
  }
}

Test( network, zero_gradients_clears_all_layers )
{
  _network_t *net = make_simple_network( &param_arena, 4, 8, 3 );

  /* Manually set some gradients */
  _network_node_t *node = net->head;
  while ( node )
  {
    if ( node->layer->weights )
      for ( int i = 0; i < node->layer->weights->size; i++ ) node->layer->weights->gradients[i] = 99.9f;
    if ( node->layer->bias )
      for ( int i = 0; i < node->layer->bias->size; i++ ) node->layer->bias->gradients[i] = 99.9f;
    node = node->next;
  }

  network_zero_gradients( net );

  node = net->head;
  while ( node )
  {
    if ( node->layer->weights )
      for ( int i = 0; i < node->layer->weights->size; i++ )
        cr_assert_float_eq( node->layer->weights->gradients[i], 0.0f, 1e-7f, "Weight gradient not zeroed in layer '%s'", node->layer->layer_name );
    if ( node->layer->bias )
      for ( int i = 0; i < node->layer->bias->size; i++ )
        cr_assert_float_eq( node->layer->bias->gradients[i], 0.0f, 1e-7f, "Bias gradient not zeroed in layer '%s'", node->layer->layer_name );
    node = node->next;
  }
}

Test( network, full_training_step_does_not_crash )
{
  /* MNIST-scale: 784 → 128 → 10 */
  _network_t *net = make_simple_network( &param_arena, 784, 128, 10 );

  int batch = 8;
  int input_shape[] = { batch, 784 };
  _tensor_t *input = tensor_zeros( &batch_arena, 2, input_shape );

  /* Simulate a forward pass */
  _tensor_t *logits = net->forward( net, &batch_arena, input );
  cr_assert_not_null( logits );

  /* Compute loss */
  int raw_labels[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
  _tensor_t *labs = make_one_hot( &batch_arena, batch, 10, raw_labels );
  _loss_result_t r = loss_softmax_cross_entropy( &batch_arena, logits, labs );
  cr_assert_not_null( r.loss );
  cr_assert( isfinite( r.loss->data[0] ) );

  /* Backward pass */
  network_zero_gradients( net );
  _tensor_t *input_grad = net->backward( net, &batch_arena, r.gradients );
  cr_assert_not_null( input_grad );
}

Test( network, two_training_steps_give_deterministic_gradients )
{
  /* Running two forward+backward passes with identical inputs and
     zeroing gradients between them should give identical gradients. */
  _network_t *net = make_simple_network( &param_arena, 4, 8, 3 );

  int input_shape[] = { 2, 4 };
  int labels[] = { 0, 2 };

  /* First step */
  _tensor_t *input1 = tensor_ones( &batch_arena, 2, input_shape );
  net->forward( net, &batch_arena, input1 );
  _tensor_t *labs1 = make_one_hot( &batch_arena, 2, 3, labels );
  _loss_result_t r1 = loss_softmax_cross_entropy( &batch_arena, net->tail->output, labs1 );
  network_zero_gradients( net );
  net->backward( net, &batch_arena, r1.gradients );

  /* Capture first-step gradients of the last dense layer's first weight */
  float grad1 = net->tail->layer->weights->gradients[0];

  /* Second step — identical setup */
  network_zero_gradients( net );
  _tensor_t *input2 = tensor_ones( &batch_arena, 2, input_shape );
  net->forward( net, &batch_arena, input2 );
  _tensor_t *labs2 = make_one_hot( &batch_arena, 2, 3, labels );
  _loss_result_t r2 = loss_softmax_cross_entropy( &batch_arena, net->tail->output, labs2 );
  net->backward( net, &batch_arena, r2.gradients );

  float grad2 = net->tail->layer->weights->gradients[0];

  cr_assert_float_eq( grad1, grad2, 1e-5f,
                      "Identical inputs should give identical gradients: "
                      "step1=%.6f, step2=%.6f",
                      grad1, grad2 );
}

Test( network, arena_checkpoint_reset_after_batch )
{
  _network_t *net = make_simple_network( &param_arena, 4, 8, 3 );
  int input_shape[] = { 2, 4 };

  _ds_arena_checkpoint_t_ cp = ds_arena_checkpoint( &batch_arena );

  _tensor_t *input = tensor_ones( &batch_arena, 2, input_shape );
  net->forward( net, &batch_arena, input );

  cr_assert( batch_arena.head->chunk_size_used > cp.checkpoint_size_used, "Batch arena should have grown during forward pass" );

  ds_arena_reset_to( &batch_arena, cp );

  size_t used_after = batch_arena.head ? batch_arena.head->chunk_size_used : 0;
  cr_assert_eq( used_after, cp.checkpoint_size_used, "Batch arena should return to checkpoint after reset" );
}

/* ── Skip connections ─────────────────────────────────────────────────────
 *
 * A skip destination computes  out = f(in) + out_source, so the gradient
 * arriving at the destination must flow back into the source unchanged.
 * These check the gradient numerically: the network's own backward pass has
 * to agree with a central finite difference of the loss.
 */

#define SKIP_DIM 3

/* Loss = Σ w·out, so dL/dout = w and the gradient wrt the input is exactly
   what network_backward should return. */
static float skip_probe_loss( _network_t *net, _ds_arena_t_ *arena, const float *x, const float *w )
{
  _ds_arena_checkpoint_t_ cp = ds_arena_checkpoint( arena );

  int shape[] = { 1, SKIP_DIM };
  _tensor_t *in = tensor_zeros( arena, 2, shape );
  for ( int i = 0; i < SKIP_DIM; i++ ) in->data[i] = x[i];

  _tensor_t *out = net->forward( net, arena, in );

  float loss = 0.0f;
  for ( int i = 0; i < SKIP_DIM; i++ ) loss += w[i] * out->data[i];

  ds_arena_reset_to( arena, cp );
  return loss;
}

/* dense -> relu -> dense, optionally with a skip from the first dense to the
   second. Both dense layers are square so the skip shapes line up. */
static _network_t *make_skip_network( _ds_arena_t_ *a, bool with_skip )
{
  _network_t *net = network_create( a );

  network_add_layer( a, net, layer_create_dense( a, SKIP_DIM, SKIP_DIM ) );
  _network_node_t *source = net->tail;

  network_add_layer( a, net, layer_create_relu( a ) );

  network_add_layer( a, net, layer_create_dense( a, SKIP_DIM, SKIP_DIM ) );
  _network_node_t *destination = net->tail;

  if ( with_skip ) network_add_skip( source, destination );
  return net;
}

static void assert_input_gradient_matches_finite_difference( bool with_skip )
{
  _network_t *net = make_skip_network( &param_arena, with_skip );

  const float w[SKIP_DIM] = { 0.3f, 0.5f, 0.7f };
  float x[SKIP_DIM] = { 0.5f, -0.25f, 0.8f };

  int shape[] = { 1, SKIP_DIM };
  _tensor_t *input = tensor_zeros( &batch_arena, 2, shape );
  for ( int i = 0; i < SKIP_DIM; i++ ) input->data[i] = x[i];

  net->forward( net, &batch_arena, input );

  _tensor_t *output_gradients = tensor_zeros( &batch_arena, 2, shape );
  for ( int i = 0; i < SKIP_DIM; i++ ) output_gradients->data[i] = w[i];

  network_zero_gradients( net );
  _tensor_t *input_gradients = net->backward( net, &batch_arena, output_gradients );
  cr_assert_not_null( input_gradients );

  const float eps = 1e-3f;
  for ( int i = 0; i < SKIP_DIM; i++ )
  {
    float saved = x[i];

    x[i] = saved + eps;
    float plus = skip_probe_loss( net, &batch_arena, x, w );

    x[i] = saved - eps;
    float minus = skip_probe_loss( net, &batch_arena, x, w );

    x[i] = saved;

    float numeric = ( plus - minus ) / ( 2.0f * eps );
    float analytic = input_gradients->data[i];

    cr_assert_float_eq( analytic, numeric, 1e-2f, "d/dx[%d]: backward gave %.6f, finite difference gave %.6f%s", i, (double)analytic, (double)numeric,
                        with_skip ? " (with skip connection)" : "" );
  }
}

Test( network, backward_matches_finite_difference_without_skip )
{
  assert_input_gradient_matches_finite_difference( false );
}

Test( network, backward_matches_finite_difference_with_skip )
{
  /* Regression: backward used to fold the skip destination's forward
     activations into the gradient instead of the gradient itself. */
  assert_input_gradient_matches_finite_difference( true );
}

Test( network, skip_gradients_are_cleared_between_passes )
{
  _network_t *net = make_skip_network( &param_arena, true );

  int shape[] = { 1, SKIP_DIM };
  _tensor_t *input = tensor_ones( &batch_arena, 2, shape );
  _tensor_t *output_gradients = tensor_ones( &batch_arena, 2, shape );

  for ( int pass = 0; pass < 2; pass++ )
  {
    net->forward( net, &batch_arena, input );
    network_zero_gradients( net );
    cr_assert_not_null( net->backward( net, &batch_arena, output_gradients ) );
  }

  for ( _network_node_t *node = net->head; node != NULL; node = node->next )
    cr_assert_null( node->skip_gradients, "skip gradients must not survive a backward pass" );
}
