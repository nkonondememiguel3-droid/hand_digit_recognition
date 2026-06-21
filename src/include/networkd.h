#ifndef hand_digit_recognition_network
#define hand_digit_recognition_network

#include "tensor.h"

extern float forwardpropagation(_tensor_t *self, _tensor_t *target);
extern float backpropagation(_tensor_t *self, _tensor_t *previous);

#endif // hand_digit_recognition_network
