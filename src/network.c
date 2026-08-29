#include "ds_arena.h"
#include "networkd.h"
#include "tensor.h"
#include <stdbool.h>
#include <stdio.h>

static _tensor_t *network_forward( _network_t *self, _ds_arena_t_ *batch_arena, _tensor_t *input )
{
  if ( !self || !input )
  {
    fprintf( stderr, "network_forward: NULL network or input.\n" );
    return NULL;
  }
  if ( !self->head )
  {
    fprintf( stderr, "network_forward: network has not input\n" );
    return NULL;
  }

  _tensor_t *current = input;
  _network_node_t *node = self->head;

  while ( node != NULL )
  {
    _tensor_t *output = node->layer->forward( batch_arena, node->layer, current );
    if ( !output )
    {
      fprintf( stderr, "network_forward: layer '%s' returned NULL\n", node->layer->layer_name );
      return NULL;
    }

    // applied skip connection if this node is a skip destination
    if ( node->skip_source != NULL )
    {
      if ( !node->skip_source->output )
      {
        fprintf( stderr, "network_forward: skip source '%s' has no cached output\n", node->skip_source->layer->layer_name );
        return NULL;
      }

      if ( !tensor_add_inplace( output, node->skip_source->output ) )
      {
        fprintf( stderr, "network_forwar: skip connecction shape mismatch at '%s'\n", node->layer->layer_name );
        return NULL;
      }
    }

    // cache output : it is needed for skip destinations and the backward pass
    node->output = output;
    current = output;
    node = node->next;
  }

  // output the last layer
  return current;
}

/* Route `grad` back to a skip source.
 *
 * A skip destination computes  out_dest = f_dest(in_dest) + out_source,
 * so the gradient reaching the destination's output flows unchanged into
 * the source's output as well. It is accumulated (not overwritten) because
 * several destinations may share one source.
 *
 * A copy is taken rather than a reference: the gradient tensor the caller
 * holds is folded into and mutated further down the chain.
 */
static bool accumulate_skip_gradient( _ds_arena_t_ *batch_arena, _network_node_t *source, const _tensor_t *grad )
{
  if ( source->skip_gradients == NULL )
  {
    source->skip_gradients = tensor_zeros( batch_arena, grad->dimension, grad->shape );
    if ( !source->skip_gradients )
    {
      fprintf( stderr, "network_backward: failed to allocate skip gradients for '%s'\n", source->layer->layer_name );
      return false;
    }
  }

  return tensor_add_inplace( source->skip_gradients, grad );
}

static _tensor_t *network_backward( _network_t *self, _ds_arena_t_ *batch_arena, _tensor_t *loss_gradients )
{
  if ( !self || !loss_gradients )
  {
    fprintf( stderr, "network_backward: NULL network or gradients\n" );
    return NULL;
  }

  /* Skip gradients live for exactly one backward pass */
  for ( _network_node_t *n = self->head; n != NULL; n = n->next ) n->skip_gradients = NULL;

  _tensor_t *grad = loss_gradients;
  _network_node_t *node = self->tail;

  while ( node != NULL )
  {
    /* Gradient routed back from every layer that used this node as a skip
       source. Folded in first, so a node that is both a destination and a
       source passes the complete gradient on.                            */
    if ( node->skip_gradients != NULL )
    {
      if ( !tensor_add_inplace( grad, node->skip_gradients ) )
      {
        fprintf( stderr, "network_backward: skip gradient shape mismatch at '%s'\n", node->layer->layer_name );
        return NULL;
      }
      node->skip_gradients = NULL;
    }

    if ( node->skip_source != NULL )
    {
      if ( !accumulate_skip_gradient( batch_arena, node->skip_source, grad ) )
      {
        fprintf( stderr, "network_backward: skip connection shape mismatch at '%s'\n", node->layer->layer_name );
        return NULL;
      }
    }

    _tensor_t *input_gradients = node->layer->backward( batch_arena, node->layer, grad );
    if ( !input_gradients )
    {
      fprintf( stderr, "network_backward: layer '%s' returned NULL gradients\n", node->layer->layer_name );
      return NULL;
    }

    grad = input_gradients;
    node = node->prev;
  }

  return grad;
}

_network_t *network_create( _ds_arena_t_ *persist_arena )
{
  _network_t *net = ARENA_NEW( persist_arena, _network_t );
  net->head = NULL;
  net->tail = NULL;
  net->number_layers = 0;
  net->forward = network_forward;
  net->backward = network_backward;

  return net;
}

void network_add_layer( _ds_arena_t_ *persist_arena, _network_t *network, _layer_t *layer )
{
  _network_node_t *node = ARENA_NEW( persist_arena, _network_node_t );
  node->layer = layer;
  node->output = NULL;
  node->skip_source = NULL;
  node->skip_gradients = NULL;
  node->next = NULL;
  node->prev = network->tail;

  if ( network->tail != NULL ) network->tail->next = node;
  else network->head = node;

  network->tail = node;
  network->number_layers++;
}

void network_add_skip( _network_node_t *source, _network_node_t *destination )
{
  if ( !source || !destination )
  {
    fprintf( stderr, "network_add_skip: NULL source or destination\n" );
    return;
  }

  if ( source == destination )
  {
    fprintf( stderr, "network_add_skip: source and destination are the same node\n" );
    return;
  }

  destination->skip_source = source;
}

void network_zero_gradients( _network_t *network )
{
  _network_node_t *node = network->head;
  while ( node != NULL )
  {
    if ( node->layer->weights ) tensor_zero_gradients( node->layer->weights );
    if ( node->layer->bias ) tensor_zero_gradients( node->layer->bias );
    node = node->next;
  }
}

void network_print( const _network_t *network )
{
  printf( "network (%d layers)\n", network->number_layers );
  printf( "-------------------------------------------------------\n" );

  _network_node_t *node = network->head;
  int index = 0;
  while ( node != NULL )
  {
    /* ✅ in_shape/out_shape replace in_dimension/out_dimension.
       dims[1] is the "feature" dimension for dense-style layers;
       for shape-agnostic layers (relu/sigmoid) ndim==0 so we print
       "-" instead of a meaningless 0.                              */
    if ( node->layer->in_shape.ndim > 0 && node->layer->out_shape.ndim > 0 )
    {
      printf( "[%2d] %-20s in: %-6d out: %-6d", index, node->layer->layer_name, node->layer->in_shape.dims[1], node->layer->out_shape.dims[1] );
    }
    else { printf( "[%2d] %-20s in: %-6s out: %-6s", index, node->layer->layer_name, "-", "-" ); }

    if ( node->skip_source ) printf( " <- skip from [%s]", node->skip_source->layer->layer_name );

    printf( "\n" );
    node = node->next;
    index++;
  }
  printf( "------------------------------------------------\n" );
}

/* ── network.c addition ──────────────────────────────────────────────── */
bool network_add_layer_checked( _ds_arena_t_ *persist_arena, _network_t *network, _layer_t *layer )
{
  if ( network->tail != NULL && layer->in_shape.ndim > 0 )
  {
    _layer_shape_t *prev_out = &network->tail->layer->out_shape;
    if ( prev_out->ndim > 0 )
    {
      bool compatible = ( prev_out->ndim == layer->in_shape.ndim );
      if ( compatible )
        for ( int i = 1; i < prev_out->ndim; i++ ) /* skip batch dim */
          if ( prev_out->dims[i] != layer->in_shape.dims[i] )
          {
            compatible = false;
            break;
          }

      if ( !compatible )
      {
        fprintf( stderr,
                 "network_add_layer_checked: shape mismatch -- '%s' outputs "
                 "ndim=%d but '%s' expects ndim=%d\n",
                 network->tail->layer->layer_name, prev_out->ndim, layer->layer_name, layer->in_shape.ndim );
        return false;
      }
    }
  }

  network_add_layer( persist_arena, network, layer );
  return true;
}
