#include "optimizer.h"
#include "ds_arena.h"
#include "networkd.h"
#include "tensor.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

/* Internal: count layers that have learnable parameters */
static int count_param_layers( _network_t *network )
{
  int count = 0;
  _network_node_t *node = network->head;
  while ( node )
  {
    if ( node->layer->weights || node->layer->bias ) count++;
    node = node->next;
  }
  return count;
}

/* ══════════════════════════════════════════════════════════════════════════
 * SGD with momentum
 *
 * Update rule per parameter θ with gradient g and velocity v:
 *
 *   v_t  = µ·v_{t-1} + g_t + λ·θ_{t-1}    (weight decay applied to v)
 *   θ_t  = θ_{t-1} - η·v_t
 *
 * When µ=0 this reduces to plain SGD:  θ = θ - η·(g + λ·θ)
 * ══════════════════════════════════════════════════════════════════════════ */
_optimizer_t *optimizer_create_sgd( _ds_arena_t_ *arena, _network_t *network, _sgd_config_t config )
{
  _optimizer_t *opt = ARENA_NEW( arena, _optimizer_t );
  opt->type = OPTIMIZER_SGD;
  opt->config.sgd = config;
  opt->step = 0;

  int n = count_param_layers( network );
  opt->num_layers = n;

  /* Allocate velocity pointer arrays */
  opt->sgd_velocities_w = ARENA_ARRAY( arena, float *, n );
  opt->sgd_velocities_b = ARENA_ARRAY( arena, float *, n );

  int idx = 0;
  _network_node_t *node = network->head;
  while ( node )
  {
    _layer_t *l = node->layer;

    /* Parameterless layers own no slot -- writing one here would run off
       the end of the arrays when the network ends in an activation. */
    if ( !l->weights && !l->bias )
    {
      node = node->next;
      continue;
    }

    opt->sgd_velocities_w[idx] = l->weights ? ARENA_ARRAY( arena, float, l->weights->size ) : NULL;
    opt->sgd_velocities_b[idx] = l->bias ? ARENA_ARRAY( arena, float, l->bias->size ) : NULL;

    idx++;
    node = node->next;
  }

  return opt;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Adam
 *
 * Update rule per parameter θ with gradient g at step t:
 *
 *   m_t  = β₁·m_{t-1} + (1-β₁)·g_t
 *   v_t  = β₂·v_{t-1} + (1-β₂)·g_t²
 *   m̂_t = m_t / (1 - β₁ᵗ)                 (bias-corrected)
 *   v̂_t = v_t / (1 - β₂ᵗ)                 (bias-corrected)
 *   θ_t  = θ_{t-1} - η · m̂_t / (√v̂_t + ε) - η·λ·θ_{t-1}
 *
 * The last term is decoupled weight decay (AdamW style), applied only
 * when config.weight_decay > 0.
 * ══════════════════════════════════════════════════════════════════════════ */
_optimizer_t *optimizer_create_adam( _ds_arena_t_ *arena, _network_t *network, _adam_config_t config )
{
  _optimizer_t *opt = ARENA_NEW( arena, _optimizer_t );
  opt->type = OPTIMIZER_ADAM;
  opt->config.adam = config;
  opt->step = 0;

  int n = count_param_layers( network );
  opt->num_layers = n;
  opt->adam_states = ARENA_ARRAY( arena, _adam_layer_state_t, n );

  int idx = 0;
  _network_node_t *node = network->head;
  while ( node )
  {
    _layer_t *l = node->layer;

    /* See optimizer_create_sgd: only parameterized layers own a slot. */
    if ( !l->weights && !l->bias )
    {
      node = node->next;
      continue;
    }

    if ( l->weights )
    {
      opt->adam_states[idx].weights.size = l->weights->size;
      opt->adam_states[idx].weights.m = ARENA_ARRAY( arena, float, l->weights->size );
      opt->adam_states[idx].weights.v = ARENA_ARRAY( arena, float, l->weights->size );
    }
    else
    {
      opt->adam_states[idx].weights.size = 0;
      opt->adam_states[idx].weights.m = NULL;
      opt->adam_states[idx].weights.v = NULL;
    }

    if ( l->bias )
    {
      opt->adam_states[idx].bias.size = l->bias->size;
      opt->adam_states[idx].bias.m = ARENA_ARRAY( arena, float, l->bias->size );
      opt->adam_states[idx].bias.v = ARENA_ARRAY( arena, float, l->bias->size );
    }
    else
    {
      opt->adam_states[idx].bias.size = 0;
      opt->adam_states[idx].bias.m = NULL;
      opt->adam_states[idx].bias.v = NULL;
    }

    idx++;
    node = node->next;
  }

  return opt;
}

/* Internal: apply SGD update to one parameter tensor */
static void sgd_update_param( float *params, float *grads, float *velocity, int size, _sgd_config_t cfg )
{
  float lr = cfg.learning_rate;
  float mu = cfg.momentum;
  float wd = cfg.weight_decay;

  for ( int i = 0; i < size; i++ )
  {
    float g = grads[i] + wd * params[i]; /* L2 gradient */

    if ( velocity )
    {
      velocity[i] = mu * velocity[i] + g;
      params[i] -= lr * velocity[i];
    }
    else { params[i] -= lr * g; /* plain SGD, no momentum */ }
  }
}

/* Internal: apply Adam update to one parameter tensor */
static void adam_update_param( float *params, float *grads, float *m, float *v, int size, _adam_config_t cfg, int step )
{
  float lr = cfg.learning_rate;
  float b1 = cfg.beta1;
  float b2 = cfg.beta2;
  float eps = cfg.epsilon;
  float wd = cfg.weight_decay;

  /* Bias-correction denominators — computed once per call */
  float bc1 = 1.0f - powf( b1, (float)step );
  float bc2 = 1.0f - powf( b2, (float)step );

  for ( int i = 0; i < size; i++ )
  {
    float g = grads[i];

    /* Update biased first and second moment estimates */
    m[i] = b1 * m[i] + ( 1.0f - b1 ) * g;
    v[i] = b2 * v[i] + ( 1.0f - b2 ) * g * g;

    /* Compute bias-corrected estimates */
    float m_hat = m[i] / bc1;
    float v_hat = v[i] / bc2;

    /* Parameter update with optional decoupled weight decay (AdamW) */
    params[i] -= lr * m_hat / ( sqrtf( v_hat ) + eps ) + lr * wd * params[i];
  }
}

/* optimizer_step */
void optimizer_step( _optimizer_t *opt, _network_t *network )
{
  opt->step++;

  int idx = 0;
  _network_node_t *node = network->head;

  while ( node )
  {
    _layer_t *l = node->layer;

    if ( !l->weights && !l->bias )
    {
      /* Activation layers (ReLU, Sigmoid) - no parameters to update */
      node = node->next;
      continue;
    }

    if ( opt->type == OPTIMIZER_SGD )
    {
      if ( l->weights && l->weights->gradients )
        sgd_update_param( l->weights->data, l->weights->gradients, opt->sgd_velocities_w[idx], l->weights->size, opt->config.sgd );

      if ( l->bias && l->bias->gradients )
        sgd_update_param( l->bias->data, l->bias->gradients, opt->sgd_velocities_b[idx], l->bias->size, opt->config.sgd );
    }
    else /* OPTIMIZER_ADAM */
    {
      _adam_layer_state_t *state = &opt->adam_states[idx];

      if ( l->weights && l->weights->gradients && state->weights.m )
        adam_update_param( l->weights->data, l->weights->gradients, state->weights.m, state->weights.v, l->weights->size, opt->config.adam,
                           opt->step );

      if ( l->bias && l->bias->gradients && state->bias.m )
        adam_update_param( l->bias->data, l->bias->gradients, state->bias.m, state->bias.v, l->bias->size, opt->config.adam, opt->step );
    }

    idx++;
    node = node->next;
  }
}
