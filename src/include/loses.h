#ifndef hand_digit_recognition_losses_h
#define hand_digit_recognition_losses_h

#include "ds_arena.h"
#include "tensor.h"
typedef struct
{
  _tensor_t *loss; // a scalar tensor representing the loss over a batch of samples
  _tensor_t *gradients;
} _loss_result_t;

extern _loss_result_t loss_softmax_cross_entropy( _ds_arena_t_ *arena, _tensor_t *logits, _tensor_t *labels );

#endif // hand_digit_recognition_losses_h
