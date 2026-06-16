#include "dataloader.h"
#include "common.h"
#include "ds_arena.h"
#include <SDL3/SDL_stdinc.h>
#include <stdint.h>
#include <stdio.h>

static _mnist_head_ read_header( FILE *fp )
{
  uint32_t img_magic = 0, count = 0;
  fread( &img_magic, sizeof( uint32_t ), 1, fp );
  fread( &count, sizeof( uint32_t ), 1, fp );

  img_magic = SWAP32( img_magic );
  count = SWAP32( count );

  return ( _mnist_head_ ){
    .magic_number = img_magic,
    .count = count,
  };
}

static __mnist_image__ *load_images( _ds_arena_t_ *arena, const char *img_path )
{
  FILE *fp = fopen( img_path, "rb" );
  if ( fp == NULL )
  {
    fprintf( stderr, "Failed to read the image binary file. Path: %s\n", img_path );
    return NULL;
  }

  _mnist_head_ header = read_header( fp );
  uint32_t rows = 0, cols = 0;
  fread( &rows, sizeof( uint32_t ), 1, fp );
  fread( &cols, sizeof( uint32_t ), 1, fp );
  rows = SWAP32( rows );
  cols = SWAP32( cols );

  if ( header.magic_number != IMG_MAGIC_NUMBER ) // ✅ keep — validates file type
  {
    fprintf( stderr, "Image Magic number does not match. Expected %d but got %d\n", IMG_MAGIC_NUMBER, header.magic_number );
    fclose( fp );
    return NULL;
  }
  // ✅ no count check — trust the header

  uint8_t *pixels = ARENA_ARRAY( arena, uint8_t, rows * cols * header.count );
  fread( pixels, sizeof( uint8_t ) * rows * cols * header.count, 1, fp );
  fclose( fp );

  __mnist_image__ *img = ARENA_NEW( arena, __mnist_image__ );
  img->header = header;
  img->number_of_rows = rows;
  img->number_of_columns = cols;
  img->pixels = pixels;
  return img;
}

static __mnist_label__ *load_labels( _ds_arena_t_ *arena, const char *label_path )
{
  FILE *fp = fopen( label_path, "rb" );
  if ( fp == NULL )
  {
    fprintf( stderr, "Failed to read the label binary file. Path: %s\n", label_path );
    return NULL;
  }

  _mnist_head_ header = read_header( fp );

  if ( header.magic_number != LABEL_MAGIC_NUMBER ) // ✅ keep — validates file type
  {
    fprintf( stderr, "Label Magic number does not match. Expected %d but got %d\n", LABEL_MAGIC_NUMBER, header.magic_number );
    fclose( fp );
    return NULL;
  }
  // ✅ no count check — trust the header

  uint8_t *labels = ARENA_ARRAY( arena, uint8_t, header.count );
  fread( labels, sizeof( uint8_t ) * header.count, 1, fp );
  fclose( fp );

  __mnist_label__ *lab = ARENA_NEW( arena, __mnist_label__ );
  lab->header = header;
  lab->label = labels;
  lab->number_of_label = header.count;
  return lab;
}

__dataset__ *load_dataset( _ds_arena_t_ *arena, const char *img_path, const char *label_path )
{

  __mnist_image__ *images = load_images( arena, img_path );
  if ( images == NULL )
  {
    fprintf( stderr, "Failed to load the images from mnist.\n" );
    return NULL;
  }

  __mnist_label__ *labels = load_labels( arena, label_path );
  if ( labels == NULL )
  {
    fprintf( stderr, "Failed to load all the labels from mnist.\n" );
    return NULL;
  }

  __dataset__ *dataset = ARENA_NEW( arena, __dataset__ );
  dataset->images = images;
  dataset->labels = labels;

  return dataset;
}
