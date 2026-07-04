#ifndef hand_digit_recognition_network
#define hand_digit_recognition_network

#include "ds_arena.h"
#include "layers.h"
#include "tensor.h"

typedef struct _network_node_ _network_node_t;
typedef struct _network_ _network_t;

// one node in the layer double linked list
struct _network_node_
{
  _layer_t *layer;
  _tensor_t *output;
  _network_node_t *skip_source;
  _network_node_t *next;
  _network_node_t *prev;
};

// restnet-style skip connections.
// layers are stored as a double linked list so both forward and backward traversal are O(1) per step.
// NOTE: memory lifetime contract:
//    - the network struct and all the nodes are allocated from 'presist_arena'.
//      XXX: this arena must not be reset while the network is in use.
//    - forward and backward take a batch_arena for intermediate activations.
//      XXX: batch_arena is only reset after the backward completes.
struct _network_
{
  _network_node_t *head; // first layer
  _network_node_t *tail; // output layer
  int number_layers;

  _tensor_t *( *forward )( _network_t *self, _ds_arena_t_ *batch_arena, _tensor_t *input );
  _tensor_t *( *backward )( _network_t *self, _ds_arena_t_ *batch_arena, _tensor_t *loss_gradients );
};

extern _network_t *network_create( _ds_arena_t_ *persist_arena );

extern void network_add_layer( _ds_arena_t_ *persist_arena, _network_t *network, _layer_t *layer );

extern void network_add_skip( _network_node_t *source, _network_node_t *destination );

// NOTE: call this function before each backward() pass.
extern void network_zero_gradients( _network_t *network );

extern void network_print( const _network_t *network );
/* ── Add to network.h ─────────────────────────────────────────────────── */

/*
 * network_add_layer_checked — like network_add_layer, but validates that
 * the new layer's in_shape matches the network's current output shape
 * (i.e. the tail node's out_shape). Returns false and logs an error on
 * mismatch, leaving the network unchanged.
 *
 * Pass NULL for `layer->in_shape` validation if the layer is shape-
 * agnostic (e.g. activation layers that pass through whatever shape
 * they receive) -- ndim=0 is treated as "accepts anything".
 */
extern bool network_add_layer_checked( _ds_arena_t_ *persist_arena, _network_t *network, _layer_t *layer );

#endif // hand_digit_recognition_network
