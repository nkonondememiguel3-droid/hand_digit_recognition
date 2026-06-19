#ifndef hand_digit_recognition_h
#define hand_digit_recognition_h

#include "ds_arena.h"
#include <stdint.h>

#define TENSOR_MAX_DIMS 4

typedef struct
{
  float *data;
  uint32_t ndim;
  uint32_t shape[TENSOR_MAX_DIMS];
  uint32_t size;
} _tensor_t;

extern _tensor_t tensor_create( _ds_arena_t_ *arena, const uint32_t *shape, uint32_t ndim );

#endif // hand_digit_recognition_h
