#include <criterion/criterion.h>
#include <criterion/internal/test.h>

#include <criterion/internal/assert.h>
#include <stdint.h>

#include "dataloader.h"
#include "ds_arena.h"

#define TRAIN_IMG_PATH "../datasets/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "../datasets/train-labels.idx1-ubyte"
#define TEST_IMG_PATH "../datasets/t10k-images.idx3-ubyte"
#define TEST_LBL_PATH "../datasets/t10k-labels.idx1-ubyte"
#define BAD_PATH "../datasets/does_not_exist.bin"

/* Suite setup */
static _ds_arena_t_ arena;

void setup( void )
{
  arena = ds_arena_new( 0 );
}
void teardown( void )
{
  ds_arena_destroy( &arena );
}

TestSuite( dataloader, .init = setup, .fini = teardown );

Test( dataloader, train_dataset_not_null )
{
  __dataset__ *ds = dataset_init( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds, "dataset_init() returned NULL for training set" );
  dataset_close( ds );
}

Test( dataloader, train_image_magic_number )
{
  __dataset__ *ds = dataset_init( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds );
  cr_assert_eq( ds->image_header.magic_number, IMG_MAGIC_NUMBER, "Expected image magic %d, got %u", IMG_MAGIC_NUMBER, ds->image_header.magic_number );
  dataset_close( ds );
}

Test( dataloader, train_label_magic_number )
{
  __dataset__ *ds = dataset_init( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds );
  cr_assert_eq( ds->label_header.magic_number, LABEL_MAGIC_NUMBER, "Expected label magic %d, got %u", LABEL_MAGIC_NUMBER,
                ds->label_header.magic_number );
  dataset_close( ds );
}

Test( dataloader, train_image_count )
{
  __dataset__ *ds = dataset_init( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds );
  cr_assert_eq( ds->image_header.count, 60000, "Expected 60000 images, got %u", ds->image_header.count );
  dataset_close( ds );
}

Test( dataloader, train_label_count )
{
  __dataset__ *ds = dataset_init( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds );
  cr_assert_eq( ds->label_header.count, 60000, "Expected 60000 labels, got %u", ds->label_header.count );
  dataset_close( ds );
}

Test( dataloader, train_image_dimensions )
{
  __dataset__ *ds = dataset_init( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds );
  cr_assert_eq( ds->rows, IMG_HEIGHT, "Expected %d rows, got %u", IMG_HEIGHT, ds->rows );
  cr_assert_eq( ds->cols, IMG_WIDTH, "Expected %d columns, got %u", IMG_WIDTH, ds->cols );
  dataset_close( ds );
}

Test( dataloader, test_set_image_count )
{
  __dataset__ *ds = dataset_init( &arena, TEST_IMG_PATH, TEST_LBL_PATH );
  cr_assert_not_null( ds );
  cr_assert_eq( ds->image_header.count, 10000, "Expected 10000 test images, got %u", ds->image_header.count );
  dataset_close( ds );
}

Test( dataloader, test_set_label_count )
{
  __dataset__ *ds = dataset_init( &arena, TEST_IMG_PATH, TEST_LBL_PATH );
  cr_assert_not_null( ds );
  cr_assert_eq( ds->label_header.count, 10000, "Expected 10000 test labels, got %u", ds->label_header.count );
  dataset_close( ds );
}

Test( dataloader, test_set_dimensions )
{
  __dataset__ *ds = dataset_init( &arena, TEST_IMG_PATH, TEST_LBL_PATH );
  cr_assert_not_null( ds );
  cr_assert_eq( ds->rows, IMG_HEIGHT );
  cr_assert_eq( ds->cols, IMG_WIDTH );
  dataset_close( ds );
}

Test( dataloader, load_image_returns_correct_shape )
{
  __dataset__ *ds = dataset_init( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds );

  _tensor_t *img = load_image( &arena, ds, 0 );
  cr_assert_not_null( img, "load_image() should not return NULL for a valid index" );

  /* Output tensor shape is [1, H, W] -- channel-first, matching conv conv. */
  cr_assert_eq( img->dimension, 3 );
  cr_assert_eq( img->shape[0], 1 );
  cr_assert_eq( img->shape[1], IMG_HEIGHT );
  cr_assert_eq( img->shape[2], IMG_WIDTH );
  cr_assert_eq( img->size, IMG_HEIGHT * IMG_WIDTH );

  dataset_close( ds );
}

/* Pixels are normalised to [0,1] by load_image(). Sample a slice of the
   training set rather than all 60,000 images to keep the test fast while
   still exercising the seek/read path broadly. */
Test( dataloader, load_image_normalises_pixels_to_unit_range )
{
  __dataset__ *ds = dataset_init( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds );

  int found_nonzero = 0;
  for ( int idx = 0; idx < 1000; idx++ )
  {
    _tensor_t *img = load_image( &arena, ds, idx );
    cr_assert_not_null( img, "load_image() failed at index %d", idx );

    for ( int i = 0; i < img->size; i++ )
    {
      cr_assert( img->data[i] >= 0.0f && img->data[i] <= 1.0f, "Pixel %d of image %d out of range: %f", i, idx, img->data[i] );
      if ( img->data[i] > 0.0f ) found_nonzero = 1;
    }
  }
  cr_assert( found_nonzero, "All sampled pixels are zero — data was not loaded" );

  dataset_close( ds );
}

/* MNIST image 0 is a '5' — its top-left corner pixel is background (0). */
Test( dataloader, first_image_has_background_corner )
{
  __dataset__ *ds = dataset_init( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds );

  _tensor_t *img = load_image( &arena, ds, 0 );
  cr_assert_not_null( img );
  cr_assert_float_eq( img->data[0], 0.0f, 1e-7f, "First pixel of first image should be background (0)" );

  dataset_close( ds );
}

Test( dataloader, load_image_out_of_range_returns_null )
{
  __dataset__ *ds = dataset_init( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds );

  cr_assert_null( load_image( &arena, ds, -1 ), "Negative index should return NULL" );
  cr_assert_null( load_image( &arena, ds, (int)ds->image_header.count ), "Index == count should return NULL" );

  dataset_close( ds );
}

Test( dataloader, load_label_returns_correct_shape )
{
  __dataset__ *ds = dataset_init( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds );

  _tensor_t *label = load_label( &arena, ds, 0 );
  cr_assert_not_null( label );
  cr_assert_eq( label->dimension, 1 );
  cr_assert_eq( label->shape[0], 1 );
  cr_assert_eq( label->size, 1 );

  dataset_close( ds );
}

/* Every label must decode to a digit class in [0, 9] */
Test( dataloader, labels_in_valid_range )
{
  __dataset__ *ds = dataset_init( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds );

  for ( uint32_t i = 0; i < ds->label_header.count; i++ )
  {
    _tensor_t *label = load_label( &arena, ds, (int)i );
    cr_assert_not_null( label, "load_label() failed at index %u", i );
    cr_assert( label->data[0] >= 0.0f && label->data[0] <= 9.0f, "Label[%u] = %.1f is out of range [0,9]", i, label->data[0] );
  }

  dataset_close( ds );
}

/* All 10 digit classes must appear in 60,000 training labels */
Test( dataloader, all_digit_classes_present )
{
  __dataset__ *ds = dataset_init( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds );

  int seen[10] = { 0 };
  for ( uint32_t i = 0; i < ds->label_header.count; i++ )
  {
    _tensor_t *label = load_label( &arena, ds, (int)i );
    cr_assert_not_null( label );
    seen[(int)label->data[0]] = 1;
  }

  for ( int d = 0; d < 10; d++ ) cr_assert( seen[d], "Digit class %d never appears in training labels", d );

  dataset_close( ds );
}

/* MNIST label 0 in the training set is '5' */
Test( dataloader, first_train_label_is_5 )
{
  __dataset__ *ds = dataset_init( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds );

  _tensor_t *label = load_label( &arena, ds, 0 );
  cr_assert_not_null( label );
  cr_assert_float_eq( label->data[0], 5.0f, 1e-7f, "First training label should be 5, got %.1f", label->data[0] );

  dataset_close( ds );
}

Test( dataloader, load_label_out_of_range_returns_null )
{
  __dataset__ *ds = dataset_init( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds );

  cr_assert_null( load_label( &arena, ds, -1 ), "Negative index should return NULL" );
  cr_assert_null( load_label( &arena, ds, (int)ds->label_header.count ), "Index == count should return NULL" );

  dataset_close( ds );
}

Test( dataloader, image_and_label_counts_match )
{
  __dataset__ *ds = dataset_init( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds );
  cr_assert_eq( ds->image_header.count, ds->label_header.count, "Image count %u != label count %u", ds->image_header.count, ds->label_header.count );
  dataset_close( ds );
}

Test( dataloader, null_on_bad_image_path )
{
  __dataset__ *ds = dataset_init( &arena, BAD_PATH, TRAIN_LBL_PATH );
  cr_assert_null( ds, "Should return NULL when image path is invalid" );
}

Test( dataloader, null_on_bad_label_path )
{
  __dataset__ *ds = dataset_init( &arena, TRAIN_IMG_PATH, BAD_PATH );
  cr_assert_null( ds, "Should return NULL when label path is invalid" );
}

Test( dataloader, null_on_both_bad_paths )
{
  __dataset__ *ds = dataset_init( &arena, BAD_PATH, BAD_PATH );
  cr_assert_null( ds, "Should return NULL when both paths are invalid" );
}

Test( dataloader, null_on_null_image_path )
{
  __dataset__ *ds = dataset_init( &arena, NULL, TRAIN_LBL_PATH );
  cr_assert_null( ds, "Should return NULL when image path is NULL" );
}

Test( dataloader, null_on_null_label_path )
{
  __dataset__ *ds = dataset_init( &arena, TRAIN_IMG_PATH, NULL );
  cr_assert_null( ds, "Should return NULL when label path is NULL" );
}
