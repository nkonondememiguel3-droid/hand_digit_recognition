#ifndef hand_digit_recognition_network_config_h
#define hand_digit_recognition_network_config_h

#include "ds_arena.h"
#include "layers.h"
#include "networkd.h"
#include "optimizer.h"
#include <stdbool.h>
#include <stdio.h>

/* ── Config-time constants ───────────────────────────────────────────── */
#define CONFIG_MAX_HIDDEN_LAYERS 8
#define CONFIG_MIN_DIM 1
#define CONFIG_MAX_DIM 1024

#define CONFIG_INPUT_DIM 784 /* 28x28 MNIST, fixed                   */
#define CONFIG_OUTPUT_DIM 10 /* 10 digit classes, fixed              */

/* ── Activation choice per hidden layer ──────────────────────────────── */
typedef enum
{
  CONFIG_ACTIVATION_RELU = 0,
  CONFIG_ACTIVATION_SIGMOID,
} _config_activation_t;

/* ── Optimizer choice for the whole network ──────────────────────────── */
typedef enum
{
  CONFIG_OPTIMIZER_ADAM = 0,
  CONFIG_OPTIMIZER_SGD,
} _config_optimizer_t;

/* ── One hidden layer's config row ───────────────────────────────────── */
typedef struct
{
  int dimension;
  _config_activation_t activation;
} _hidden_layer_config_t;

/* ── Full network configuration, edited by the Architecture tab ───────── */
typedef struct
{
  _hidden_layer_config_t hidden_layers[CONFIG_MAX_HIDDEN_LAYERS];
  int num_hidden_layers;

  _config_optimizer_t optimizer;
  float learning_rate;
  float sgd_momentum;

  bool is_applied;
  bool is_dirty;
} _network_config_t;

/* ── Defaults ─────────────────────────────────────────────────────────── */
static inline _network_config_t network_config_defaults( void )
{
  _network_config_t cfg;
  cfg.num_hidden_layers = 2;

  cfg.hidden_layers[0].dimension = 128;
  cfg.hidden_layers[0].activation = CONFIG_ACTIVATION_RELU;

  cfg.hidden_layers[1].dimension = 64;
  cfg.hidden_layers[1].activation = CONFIG_ACTIVATION_RELU;

  for ( int i = 2; i < CONFIG_MAX_HIDDEN_LAYERS; i++ )
  {
    cfg.hidden_layers[i].dimension = 32;
    cfg.hidden_layers[i].activation = CONFIG_ACTIVATION_RELU;
  }

  cfg.optimizer = CONFIG_OPTIMIZER_ADAM;
  cfg.learning_rate = 1e-3f;
  cfg.sgd_momentum = 0.9f;
  cfg.is_applied = false;
  cfg.is_dirty = true;

  return cfg;
}

/* ── Build a network + optimizer from a config ──────────────────────────
 *
 * Allocates the network and every layer from `weight_arena`. Caller must
 * ensure weight_arena is fresh before calling this.
 *
 * Returns true on success.
 */
static inline bool network_config_build( _ds_arena_t_ *weight_arena, const _network_config_t *cfg, _network_t **out_network,
                                         _optimizer_t **out_optimizer )
{
  if ( cfg->num_hidden_layers < 1 || cfg->num_hidden_layers > CONFIG_MAX_HIDDEN_LAYERS ) return false;

  for ( int i = 0; i < cfg->num_hidden_layers; i++ )
  {
    int d = cfg->hidden_layers[i].dimension;
    if ( d < CONFIG_MIN_DIM || d > CONFIG_MAX_DIM ) return false;
  }

  _network_t *net = network_create( weight_arena );

  int in_dim = CONFIG_INPUT_DIM;

  for ( int i = 0; i < cfg->num_hidden_layers; i++ )
  {
    int out_dim = cfg->hidden_layers[i].dimension;

    network_add_layer( weight_arena, net, layer_create_dense( weight_arena, in_dim, out_dim ) );

    if ( cfg->hidden_layers[i].activation == CONFIG_ACTIVATION_RELU ) network_add_layer( weight_arena, net, layer_create_relu( weight_arena ) );
    else network_add_layer( weight_arena, net, layer_create_sigmoid( weight_arena ) );

    in_dim = out_dim;
  }

  network_add_layer( weight_arena, net, layer_create_dense( weight_arena, in_dim, CONFIG_OUTPUT_DIM ) );

  _optimizer_t *opt = NULL;
  if ( cfg->optimizer == CONFIG_OPTIMIZER_ADAM )
  {
    _adam_config_t adam_cfg = optimizer_adam_defaults();
    adam_cfg.learning_rate = cfg->learning_rate;
    opt = optimizer_create_adam( weight_arena, net, adam_cfg );
  }
  else
  {
    _sgd_config_t sgd_cfg = optimizer_sgd_defaults();
    sgd_cfg.learning_rate = cfg->learning_rate;
    sgd_cfg.momentum = cfg->sgd_momentum;
    opt = optimizer_create_sgd( weight_arena, net, sgd_cfg );
  }

  *out_network = net;
  *out_optimizer = opt;
  return true;
}

/* ── Human-readable architecture summary ─────────────────────────────── */
static inline void network_config_summary( const _network_config_t *cfg, char *buf, int buf_size )
{
  int written = snprintf( buf, (size_t)buf_size, "%d", CONFIG_INPUT_DIM );
  for ( int i = 0; i < cfg->num_hidden_layers && written < buf_size; i++ )
    written += snprintf( buf + written, (size_t)( buf_size - written ), " -> %d", cfg->hidden_layers[i].dimension );
  snprintf( buf + written, (size_t)( buf_size - written ), " -> %d", CONFIG_OUTPUT_DIM );
}

#endif /* hand_digit_recognition_network_config_h */
