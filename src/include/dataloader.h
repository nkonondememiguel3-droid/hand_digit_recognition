#ifndef hand_digit_recognition_dataloader_h
#define hand_digit_recognition_dataloader_h

#include "ds_arena.h"
#include <stdbool.h>
#include <stdint.h>
#include "common.h"

#define IMG_WIDTH 28
#define IMG_HEIGHT 28
#define IMG_MAGIC_NUMBER 2051
#define LABEL_MAGIC_NUMBER 2049
#define IMG_NUMBER 60000
#define LABEL_NUMBER 60000

/*
 * The first thing is :
 *  - the magic number
 *  - number of images
 *  - number of rows
 *  - number of   columns
 *  - pixel data following
 * The magic number is  is composed of : 2 bytes + data type (1 byte)( 8 for unsigned byte) + 1 byte of dimensions(here is 3)
 */
typedef struct
{
  uint32_t magic_number;
  uint32_t count;
} _mnist_head_;

typedef struct
{
  _mnist_head_ header;
  uint32_t number_of_rows;
  uint32_t number_of_columns;

  uint8_t *pixels;
} __mnist_image__;

typedef struct
{
  _mnist_head_ header;
  uint32_t number_of_label;

  uint8_t *label;
} __mnist_label__;

typedef struct
{
  __mnist_image__ *images;
  __mnist_label__ *labels;
} __dataset__;

extern ARENA_ALLOC __dataset__ *load_dataset( _ds_arena_t_ *arena, const char *img_path, const char *label_path );

#endif // hand_digit_recognition_dataloader_h
