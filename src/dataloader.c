#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "dataloader.h"

#include "common.h"
#include "ds_arena.h"
#include "tensor.h"

static _mnist_head_ read_header( FILE *fp )
{
  uint32_t magic = 0, count = 0;
  fread( &magic, sizeof( uint32_t ), 1, fp );
  fread( &count, sizeof( uint32_t ), 1, fp );

  /* MNIST files are big-endian; x86/ARM hosts are little-endian */
  magic = SWAP32( magic );
  count = SWAP32( count );

  return (_mnist_head_){ .magic_number = magic, .count = count };
}

/* dataset_init
 *
 * Opens both files, validates magic numbers, reads and caches the full
 * header (including rows/cols from the image file).  The file position is
 * left pointing at the first data byte so subsequent load_image /
 * load_label calls can seek to any index efficiently.
 */
__dataset__ *dataset_init( _ds_arena_t_ *arena, const char *image_path, const char *label_path )
{
  /* ── Open image file ── */
  FILE *img_fp = fopen( image_path, "rb" );
  if ( img_fp == NULL )
  {
    fprintf( stderr, "dataset_init: failed to open image file: %s\n", image_path );
    return NULL;
  }

  _mnist_head_ image_header = read_header( img_fp );
  if ( image_header.magic_number != IMG_MAGIC_NUMBER )
  {
    fprintf( stderr, "dataset_init: image magic mismatch — expected %u, got %u\n", IMG_MAGIC_NUMBER, image_header.magic_number );
    fclose( img_fp );
    return NULL;
  }

  /* Read rows and cols from image header (bytes 8-15) */
  uint32_t rows = 0, cols = 0;
  fread( &rows, sizeof( uint32_t ), 1, img_fp );
  fread( &cols, sizeof( uint32_t ), 1, img_fp );
  rows = SWAP32( rows );
  cols = SWAP32( cols );

  /* Sanity-check dimensions against our compile-time constants */
  if ( rows != IMG_HEIGHT || cols != IMG_WIDTH )
  {
    fprintf( stderr, "dataset_init: unexpected image dimensions %ux%u (expected %dx%d)\n", rows, cols, IMG_HEIGHT, IMG_WIDTH );
    fclose( img_fp );
    return NULL;
  }

  /* Open label file */
  FILE *label_fp = fopen( label_path, "rb" );
  if ( label_fp == NULL )
  {
    fprintf( stderr, "dataset_init: failed to open label file: %s\n", label_path );
    fclose( img_fp );
    return NULL;
  }

  _mnist_head_ label_header = read_header( label_fp );
  if ( label_header.magic_number != LABEL_MAGIC_NUMBER )
  {
    fprintf( stderr, "dataset_init: label magic mismatch — expected %u, got %u\n", LABEL_MAGIC_NUMBER, label_header.magic_number );
    fclose( img_fp );
    fclose( label_fp );
    return NULL;
  }

  /* Verify both files agree on count */
  if ( image_header.count != label_header.count )
  {
    fprintf( stderr, "dataset_init: image count %u != label count %u\n", image_header.count, label_header.count );
    fclose( img_fp );
    fclose( label_fp );
    return NULL;
  }

  /* ── Build the dataset struct ── */
  __dataset__ *ds = ARENA_NEW( arena, __dataset__ );
  ds->image_fp = img_fp;
  ds->label_fp = label_fp;
  ds->image_header = image_header;
  ds->label_header = label_header;
  ds->rows = rows;
  ds->cols = cols;

  return ds;
}

/* load_image
 *
 * Reads one image by absolute index.
 *
 * Byte layout of the image file after the 16-byte header:
 *   image[i] occupies bytes [16 + i*784 .. 16 + i*784 + 783]
 *
 * Each raw uint8 pixel is normalised to [0.0, 1.0] by dividing by 255.
 * MNIST convention: 0 = background (black), 255 = ink (white).
 *
 * Output tensor shape: [1, IMG_HEIGHT, IMG_WIDTH]  (C, H, W)
 *   Channel dim = 1 because MNIST is greyscale.
 */
_tensor_t *load_image( _ds_arena_t_ *arena, __dataset__ *dataset, int idx )
{
  if ( dataset == NULL || dataset->image_fp == NULL )
  {
    fprintf( stderr, "load_image: NULL dataset or closed file\n" );
    return NULL;
  }

  if ( idx < 0 || (uint32_t)idx >= dataset->image_header.count )
  {
    fprintf( stderr, "load_image: index %d out of range [0, %u)\n", idx, dataset->image_header.count );
    return NULL;
  }

  /* Seek to the start of image[idx] in the file */
  long byte_offset = (long)MNIST_IMAGE_HEADER_BYTES + (long)idx * (long)MNIST_IMAGE_BYTES;
  if ( fseek( dataset->image_fp, byte_offset, SEEK_SET ) != 0 )
  {
    fprintf( stderr, "load_image: fseek failed for index %d\n", idx );
    return NULL;
  }

  /* Read raw uint8 pixels into a temporary stack buffer */
  uint8_t raw[MNIST_IMAGE_BYTES];
  size_t n_read = fread( raw, sizeof( uint8_t ), MNIST_IMAGE_BYTES, dataset->image_fp );
  if ( n_read != MNIST_IMAGE_BYTES )
  {
    fprintf( stderr, "load_image: expected %u bytes, got %zu (index %d)\n", MNIST_IMAGE_BYTES, n_read, idx );
    return NULL;
  }

  /* Allocate output tensor [1, H, W] and normalise pixels to [0, 1] */
  const int shape[3] = { 1, (int)dataset->rows, (int)dataset->cols };
  _tensor_t *image = tensor_create( arena, 3, shape, false );
  if ( !image ) return NULL;

  for ( uint32_t i = 0; i < MNIST_IMAGE_BYTES; i++ ) image->data[i] = (float)raw[i] / 255.0f;

  return image;
}

/* load_label
 *
 * Reads one label by absolute index.
 *
 * Byte layout of the label file after the 8-byte header:
 *   label[i] occupies byte [8 + i]
 *
 * The raw uint8 value (0-9) is cast to float for convenient use in
 * cross-entropy loss functions (no conversion needed at training time).
 *
 * Output tensor shape: [1]
 */
_tensor_t *load_label( _ds_arena_t_ *arena, __dataset__ *dataset, int idx )
{
  if ( dataset == NULL || dataset->label_fp == NULL )
  {
    fprintf( stderr, "load_label: NULL dataset or closed file\n" );
    return NULL;
  }

  if ( idx < 0 || (uint32_t)idx >= dataset->label_header.count )
  {
    fprintf( stderr, "load_label: index %d out of range [0, %u)\n", idx, dataset->label_header.count );
    return NULL;
  }

  /* Seek to label[idx] */
  long byte_offset = (long)MNIST_LABEL_HEADER_BYTES + (long)idx;
  if ( fseek( dataset->label_fp, byte_offset, SEEK_SET ) != 0 )
  {
    fprintf( stderr, "load_label: fseek failed for index %d\n", idx );
    return NULL;
  }

  /* Read one byte */
  uint8_t raw = 0;
  size_t n_read = fread( &raw, sizeof( uint8_t ), 1, dataset->label_fp );
  if ( n_read != 1 )
  {
    fprintf( stderr, "load_label: fread failed for index %d\n", idx );
    return NULL;
  }

  /* Allocate output tensor [1] */
  const int shape[1] = { 1 };
  _tensor_t *label = tensor_create( arena, 1, shape, false );
  if ( !label ) return NULL;

  label->data[0] = (float)raw; /* digit class 0.0 – 9.0 */
  return label;
}

/* dataset_close
 *
 * Closes both file handles. Safe to call with NULL.
 * The __dataset__ struct itself is arena-owned — do not free it here.
 */
void dataset_close( __dataset__ *dataset )
{
  if ( dataset == NULL ) return;

  if ( dataset->image_fp != NULL )
  {
    fclose( dataset->image_fp );
    dataset->image_fp = NULL;
  }
  if ( dataset->label_fp != NULL )
  {
    fclose( dataset->label_fp );
    dataset->label_fp = NULL;
  }
}
