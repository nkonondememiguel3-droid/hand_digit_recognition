#include "dataloader.h"
#include "ds_arena.h"
#include <criterion/criterion.h>
#include <criterion/internal/assert.h>
#include <criterion/internal/test.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Paths */
#define TRAIN_IMG_PATH "../datasets/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "../datasets/train-labels.idx1-ubyte"
#define TEST_IMG_PATH "../datasets/t10k-images.idx3-ubyte"
#define TEST_LBL_PATH "../datasets/t10k-labels.idx1-ubyte"
#define BAD_PATH "../datasets/does_not_exist.bin"

/* Suite setup  */
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

/* ══════════════════════════════════════════════════════════════════════════
 * 1. HAPPY PATH — training set
 * ══════════════════════════════════════════════════════════════════════════ */

Test( dataloader, train_dataset_not_null )
{
  __dataset__ *ds = load_dataset( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds, "load_dataset() returned NULL for training set" );
}

Test( dataloader, train_image_magic_number )
{
  __dataset__ *ds = load_dataset( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds );
  cr_assert_eq( ds->images->header.magic_number, IMG_MAGIC_NUMBER, "Expected image magic %d, got %d", IMG_MAGIC_NUMBER,
                ds->images->header.magic_number );
}

Test( dataloader, train_label_magic_number )
{
  __dataset__ *ds = load_dataset( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds );
  cr_assert_eq( ds->labels->header.magic_number, LABEL_MAGIC_NUMBER, "Expected label magic %d, got %d", LABEL_MAGIC_NUMBER,
                ds->labels->header.magic_number );
}

Test( dataloader, train_image_count )
{
  __dataset__ *ds = load_dataset( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds );
  cr_assert_eq( ds->images->header.count, 60000, "Expected 60000 images, got %u", ds->images->header.count );
}

Test( dataloader, train_label_count )
{
  __dataset__ *ds = load_dataset( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds );
  cr_assert_eq( ds->labels->header.count, 60000, "Expected 60000 labels, got %u", ds->labels->header.count );
}

Test( dataloader, train_image_dimensions )
{
  __dataset__ *ds = load_dataset( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds );
  cr_assert_eq( ds->images->number_of_rows, IMG_HEIGHT, "Expected %d rows, got %u", IMG_HEIGHT, ds->images->number_of_rows );
  cr_assert_eq( ds->images->number_of_columns, IMG_WIDTH, "Expected %d columns, got %u", IMG_WIDTH, ds->images->number_of_columns );
}

Test( dataloader, train_pixels_not_null )
{
  __dataset__ *ds = load_dataset( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds );
  cr_assert_not_null( ds->images->pixels, "Pixel buffer should not be NULL" );
}

Test( dataloader, train_labels_not_null )
{
  __dataset__ *ds = load_dataset( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds );
  cr_assert_not_null( ds->labels->label, "Label buffer should not be NULL" );
}

/* ══════════════════════════════════════════════════════════════════════════
 * 2. HAPPY PATH — test set (t10k)
 * ══════════════════════════════════════════════════════════════════════════ */

Test( dataloader, test_set_image_count )
{
  __dataset__ *ds = load_dataset( &arena, TEST_IMG_PATH, TEST_LBL_PATH );
  cr_assert_not_null( ds );
  cr_assert_eq( ds->images->header.count, 10000, "Expected 10000 test images, got %u", ds->images->header.count );
}

Test( dataloader, test_set_label_count )
{
  __dataset__ *ds = load_dataset( &arena, TEST_IMG_PATH, TEST_LBL_PATH );
  cr_assert_not_null( ds );
  cr_assert_eq( ds->labels->header.count, 10000, "Expected 10000 test labels, got %u", ds->labels->header.count );
}

Test( dataloader, test_set_dimensions )
{
  __dataset__ *ds = load_dataset( &arena, TEST_IMG_PATH, TEST_LBL_PATH );
  cr_assert_not_null( ds );
  cr_assert_eq( ds->images->number_of_rows, IMG_HEIGHT );
  cr_assert_eq( ds->images->number_of_columns, IMG_WIDTH );
}

/* ══════════════════════════════════════════════════════════════════════════
 * 3. PIXEL VALUE SANITY
 * ══════════════════════════════════════════════════════════════════════════ */

/* Every pixel must be in [0, 255] — trivially true for uint8_t,
   but this also exercises that the buffer is fully readable. */
Test( dataloader, pixels_in_valid_range )
{
  __dataset__ *ds = load_dataset( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds );

  uint32_t total = ds->images->header.count * ds->images->number_of_rows * ds->images->number_of_columns;

  int found_nonzero = 0;
  for ( uint32_t i = 0; i < total; i++ )
  {
    /* uint8_t can never exceed 255, but let's confirm data isn't all-zero */
    if ( ds->images->pixels[i] > 0 ) found_nonzero = 1;
  }
  cr_assert( found_nonzero, "All pixels are zero — data was not loaded" );
}

/* Spot-check: the first image pixel buffer is non-trivially populated.
   MNIST image 0 is a '5' — its top-left corner pixels are background (0). */
Test( dataloader, first_image_has_background_corner )
{
  __dataset__ *ds = load_dataset( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds );
  /* Top-left pixel of image 0 should be 0 (background) */
  cr_assert_eq( ds->images->pixels[0], 0, "First pixel of first image should be background (0)" );
}

/* ══════════════════════════════════════════════════════════════════════════
 * 4. LABEL VALUE SANITY
 * ══════════════════════════════════════════════════════════════════════════ */

/* Every label must be 0–9 */
Test( dataloader, labels_in_valid_range )
{
  __dataset__ *ds = load_dataset( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds );

  for ( uint32_t i = 0; i < ds->labels->header.count; i++ )
  {
    cr_assert_leq( ds->labels->label[i], 9, "Label[%u] = %d is out of range [0,9]", i, ds->labels->label[i] );
  }
}

/* All 10 digit classes must appear in 60,000 training labels */
Test( dataloader, all_digit_classes_present )
{
  __dataset__ *ds = load_dataset( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds );

  int seen[10] = { 0 };
  for ( uint32_t i = 0; i < ds->labels->header.count; i++ ) seen[ds->labels->label[i]] = 1;

  for ( int d = 0; d < 10; d++ ) cr_assert( seen[d], "Digit class %d never appears in training labels", d );
}

/* MNIST label 0 in the training set is '5' */
Test( dataloader, first_train_label_is_5 )
{
  __dataset__ *ds = load_dataset( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds );
  cr_assert_eq( ds->labels->label[0], 5, "First training label should be 5, got %d", ds->labels->label[0] );
}

/* ══════════════════════════════════════════════════════════════════════════
 * 5. IMAGE ↔ LABEL ALIGNMENT
 * ══════════════════════════════════════════════════════════════════════════ */

Test( dataloader, image_and_label_counts_match )
{
  __dataset__ *ds = load_dataset( &arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  cr_assert_not_null( ds );
  cr_assert_eq( ds->images->header.count, ds->labels->header.count, "Image count %u != label count %u", ds->images->header.count,
                ds->labels->header.count );
}

/* ══════════════════════════════════════════════════════════════════════════
 * 6. ERROR HANDLING — bad paths
 * ══════════════════════════════════════════════════════════════════════════ */

Test( dataloader, null_on_bad_image_path )
{
  __dataset__ *ds = load_dataset( &arena, BAD_PATH, TRAIN_LBL_PATH );
  cr_assert_null( ds, "Should return NULL when image path is invalid" );
}

Test( dataloader, null_on_bad_label_path )
{
  __dataset__ *ds = load_dataset( &arena, TRAIN_IMG_PATH, BAD_PATH );
  cr_assert_null( ds, "Should return NULL when label path is invalid" );
}

Test( dataloader, null_on_both_bad_paths )
{
  __dataset__ *ds = load_dataset( &arena, BAD_PATH, BAD_PATH );
  cr_assert_null( ds, "Should return NULL when both paths are invalid" );
}

Test( dataloader, null_on_null_image_path )
{
  __dataset__ *ds = load_dataset( &arena, NULL, TRAIN_LBL_PATH );
  cr_assert_null( ds, "Should return NULL when image path is NULL" );
}

Test( dataloader, null_on_null_label_path )
{
  __dataset__ *ds = load_dataset( &arena, TRAIN_IMG_PATH, NULL );
  cr_assert_null( ds, "Should return NULL when label path is NULL" );
}
