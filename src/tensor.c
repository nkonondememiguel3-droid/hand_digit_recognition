#include "tensor.h"
#include "ds_arena.h"
#include <stdint.h>

_tensor_t tensor_create(_ds_arena_t_ *arena, const uint32_t *shape, uint32_t ndim)
{
  _tensor_t tensor = {0};
  tensor.ndim  = ndim;

  for (uint32_t i = 0; i < TENSOR_MAX_DIMS; i++)
    tensor.shape[i] = (i < ndim) ? shape[i] : 1;

  uint32_t size = 0;
  for (uint32_t i = 0; i < ndim; i++)
    size *= shape[i];

  tensor.data = ARENA_ARRAY(arena, float, size);

  return tensor;
}
