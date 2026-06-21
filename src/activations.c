#include "activations.h"
#include <math.h>

float sigmoid( float x )
{
  return 1.0f / ( 1.0f + expf( -x ) );
}

float sigmoid_gradients( float x )
{
  return sigmoid( x ) * ( 1 - sigmoid( x ) );
}
