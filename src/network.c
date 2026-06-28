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

static _tensor_t *network_backward( _network_t *self, _ds_arena_t_ *batch_arena, _tensor_t *loss_gradients )
{
  if ( !self || !loss_gradients )
  {
    fprintf( stderr, "network_backward: NULL network or gradients\n" );
    return NULL;
  }

  _tensor_t *grad = loss_gradients;
  _network_node_t *node = self->tail;

  while ( node != NULL )
  {
    _network_node_t *scan = node->next;
    while ( scan != NULL )
    {
      if ( scan->skip_source == node )
      {
        if ( scan->output ) tensor_add_inplace( grad, scan->output );
      }
      scan = scan->next;
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
  net->backwar = network_backward;

  return net;
}

void network_add_layer( _ds_arena_t_ *persist_arena, _network_t *network, _layer_t *layer )
{
  _network_node_t *node = ARENA_NEW( persist_arena, _network_node_t );
  node->layer = layer;
  node->output = NULL;
  node->skip_source = NULL;
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
  printf( "-------------------------------------------------------" );

  _network_node_t *node = network->head;
  int index = 0;
  while ( node != NULL )
  {
    printf( "[%2d] %-20s in: %-6d out: %-6d", index, node->layer->layer_name, node->layer->in_dimension, node->layer->out_dimension );

    if ( node->skip_source ) printf( " <- skip from [%s]", node->skip_source->layer->layer_name );

    printf( "\n" );
    node = node->next;
    index++;
  }
  printf( "------------------------------------------------" );
}
