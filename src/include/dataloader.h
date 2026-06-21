#ifndef hand_digit_recognition_dataloader_h
#define hand_digit_recognition_dataloader_h

#include "common.h"
#include "ds_arena.h"
#include "tensor.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

/* MNIST file constants */
#define IMG_WIDTH 28
#define IMG_HEIGHT 28
#define IMG_MAGIC_NUMBER 2051
#define LABEL_MAGIC_NUMBER 2049

/*
 * MNIST binary format offsets
 *
 * Image file layout:
 *   [0..3]   magic number  (big-endian uint32)
 *   [4..7]   image count   (big-endian uint32)
 *   [8..11]  rows          (big-endian uint32)
 *   [12..15] cols          (big-endian uint32)
 *   [16..]   pixel data    (uint8, rows*cols bytes per image)
 *
 * Label file layout:
 *   [0..3]   magic number  (big-endian uint32)
 *   [4..7]   label count   (big-endian uint32)
 *   [8..]    label data    (uint8, 1 byte per label, value 0-9)
 */
#define MNIST_IMAGE_HEADER_BYTES 16u
#define MNIST_LABEL_HEADER_BYTES 8u
#define MNIST_IMAGE_BYTES ( (uint32_t)( IMG_WIDTH * IMG_HEIGHT ) )

/* Header struct (shared by image and label files) */
typedef struct
{
  uint32_t magic_number;
  uint32_t count; /* number of items in this file */
} _mnist_head_;

/* Streaming dataset
 *
 * Holds open file handles and headers for both the image and label files.
 * Images and labels are read on demand via load_image() / load_label(),
 * so only one image's worth of data is in memory at any given time.
 *
 * Rows and columns are cached from the image file header so callers do
 * not need to re-derive them.
 *
 * Call dataset_close() when done to flush and close the file handles.
 * The dataset struct itself is arena-allocated and does not need freeing.
 */
typedef struct
{
  FILE *image_fp; /* open handle to the image IDX file  */
  FILE *label_fp; /* open handle to the label IDX file  */

  _mnist_head_ image_header; /* cached from image file             */
  _mnist_head_ label_header; /* cached from label file             */

  uint32_t rows; /* image height in pixels (28)        */
  uint32_t cols; /* image width  in pixels (28)        */
} __dataset__;

/* API */

/*
 * dataset_init - open both files, validate magic numbers, cache headers.
 *
 * Returns a heap-style (arena-allocated) __dataset__ on success,
 * or NULL on any error (file not found, magic mismatch, etc.).
 *
 * The returned struct owns the open FILE handles; call dataset_close()
 * when training is complete.
 */
extern __dataset__ *dataset_init( _ds_arena_t_ *arena, const char *image_path, const char *label_path );

/*
 * load_image - read one image from the dataset by absolute index.
 *
 * Seeks to the correct byte offset in the file, reads IMG_WIDTH*IMG_HEIGHT
 * raw uint8 pixels, and normalises them to [0.0, 1.0] as float32.
 *
 * The returned tensor has shape [1, IMG_HEIGHT, IMG_WIDTH] — channel-first,
 * matching the conv layer convention (C, H, W).
 *
 * The tensor is allocated from `arena`. For training loops, pass a
 * per-batch scratch arena and reset it after the batch completes.
 *
 * Returns NULL if idx is out of range or the dataset has no image file.
 */
extern _tensor_t *load_image( _ds_arena_t_ *arena, __dataset__ *dataset, int idx );

/*
 * load_label — read one label from the dataset by absolute index.
 *
 * Returns a 1-element float tensor containing the digit class (0.0–9.0).
 * The float representation is convenient for loss computation.
 *
 * Returns NULL if idx is out of range or the dataset has no label file.
 */
extern _tensor_t *load_label( _ds_arena_t_ *arena, __dataset__ *dataset, int idx );

/*
 * dataset_close - flush and close both file handles.
 *
 * Safe to call with NULL. Does not free the struct (it is arena-owned).
 */
extern void dataset_close( __dataset__ *dataset );

#endif /* hand_digit_recognition_dataloader_h */
