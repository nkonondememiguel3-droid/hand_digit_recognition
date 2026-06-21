#ifndef hand_digit_recognition_common_h
#define hand_digit_recognition_common_h

#if defined( __clang__ )
#define ARENA_ALLOC __attribute__( ( malloc( ds_arena_destroy, 1 ) ) ) _Nullable
#elif defined( __GNU__ ) && ( __GNU__ >= 11 )
#define ARENA_ALLOC __attribute__( ( malloc( arena_cannot_free_directly, 1 ) ) )
#else
#define ARENA_ALLOC
#endif

// litthe big endian into little endian
#if defined( __clang__ ) || defined( __GNU__ )
#define SWAP32( x ) __builtin_bswap32( x )
#elif defined( MSC_VER )
#include <intrin.h>
#define SWAP32( x ) _byteswap_ulong( x )
#else
static inline uint32_t SWAP32( uint32_t val )
{
  return ( ( val & 0xFF000000 ) >> 24 ) | ( ( val & 0x00FF0000 ) >> 8 ) | ( ( val & 0x0000FF00 ) << 8 ) | ( ( val & 0x000000FF ) << 24 );
}
#endif

#endif // hand_digit_recognition_common_h
