#ifndef hand_digit_recognition_optimizer_h
#define hand_digit_recognition_optimizer_h

#include "ds_arena.h"
#include "networkd.h"
#include <stdbool.h>

typedef enum
{
  OPTIMIZER_SGD,
  OPTIMIZER_ADAM,
} _optimizer_type_t;

typedef struct
{
  float learning_rate;
  float momentum;
  float weight_decay;
} _sgd_config_t;

typedef struct
{
  float learning_rate;
  float beta1;
  float beta2;
  float epsilon;
  float weight_decay;
} _adam_config_t;

typedef struct
{
  float *m;
  float *v;
  int size;
} _adam_param_state_t;

typedef struct
{
  _adam_param_state_t weights;
  _adam_param_state_t bias;
} _adam_layer_state_t;

typedef struct
{
  _optimizer_type_t type;

  union {
    _sgd_config_t sgd;
    _adam_config_t adam;
  } config;

  _adam_layer_state_t *adam_states;
  int num_layers;

  float **sgd_velocities_w;
  float **sgd_velocities_b;

  int step;
} _optimizer_t;

extern _optimizer_t *optimizer_create_sgd( _ds_arena_t_ *arena, _network_t *network, _sgd_config_t config );

extern _optimizer_t *optimizer_create_adam( _ds_arena_t_ *arena, _network_t *network, _adam_config_t config );

static inline _sgd_config_t optimizer_sgd_defaults( void )
{
  return ( _sgd_config_t ){
    .learning_rate = 0.01f,
    .momentum = 0.9f,
    .weight_decay = 0.0f,
  };
}

static inline _adam_config_t optimizer_adam_defaults( void )
{
  return ( _adam_config_t ){
    .learning_rate = 1e-3f,
    .beta1 = 0.9f,
    .beta2 = 0.999f,
    .epsilon = 1e-8f,
    .weight_decay = 0.0f,
  };
}

extern void optimizer_step( _optimizer_t *opt, _network_t *network );

#endif // hand_digit_recognition_optimizer_h
