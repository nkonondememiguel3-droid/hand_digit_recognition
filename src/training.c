#include "training.h"
#include "dataloader.h"
#include "ds_arena.h"
#include "loses.h"
#include "networkd.h"
#include "optimizer.h"
#include "tensor.h"

#include <SDL3/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* -  Internal: build a one-hot (or label-smoothed) label batch
 *
 * Label smoothing replaces hard targets:
 *   y_smooth[correct] = 1 - ε + ε/C
 *   y_smooth[other]   = ε/C
 * where ε = label_smoothing, C = num_classes.
 *
 * When label_smoothing=0 this is a standard one-hot vector.
 */
static _tensor_t *build_label_batch( _ds_arena_t_ *arena, __dataset__ *ds, int batch_start, int batch_size, int num_classes, float smoothing )
{
  int shape[] = { batch_size, num_classes };
  _tensor_t *labels = tensor_zeros( arena, 2, shape );
  if ( !labels ) return NULL;

  float smooth_val = smoothing / (float)num_classes;
  float correct_val = 1.0f - smoothing + smooth_val;

  for ( int s = 0; s < batch_size; s++ )
  {
    // load one label
    _tensor_t *lbl = load_label( arena, ds, batch_start + s );
    if ( !lbl ) return NULL;

    // TODO: do not rely on the actual value of the label('cause this is specific for MNIST dataset), instead rely on the index of the labels array so
    // it can be generalize.
    int cls = (int)lbl->data[0];
    // silently remove invalide classes
    if ( cls < 0 || cls >= num_classes ) continue;

    // fill smoothed baseline first
    if ( smoothing > 0.0f )
      for ( int j = 0; j < num_classes; j++ ) T2( labels, s, j ) = smooth_val;

    T2( labels, s, cls ) = correct_val;
  }

  return labels;
}

/* internal: build an image batch [batch_size, 784] */
static _tensor_t *build_image_batch( _ds_arena_t_ *arena, __dataset__ *ds, int batch_start, int batch_size, int img_pixels )
{
  int shape[] = { batch_size, img_pixels };
  _tensor_t *batch = tensor_zeros( arena, 2, shape );
  if ( !batch ) return NULL;

  for ( int s = 0; s < batch_size; s++ )
  {
    _tensor_t *img = load_image( arena, ds, batch_start + s );
    if ( !img ) return NULL;

    /* Copy the flat pixel buffer into row s of the batch matrix */
    float *dst = batch->data + s * img_pixels;
    memcpy( dst, img->data, (size_t)img_pixels * sizeof( float ) );
  }

  return batch;
}

/* internal: evaluate accuracy on up to `max_samples` from test_ds */
static float evaluate_accuracy( _network_t *network, __dataset__ *test_ds, int num_classes, int max_samples, _ds_arena_t_ *scratch )
{
  int total = 0;
  int correct = 0;
  int img_pixels = IMG_WIDTH * IMG_HEIGHT;

  _ds_arena_checkpoint_t_ cp = ds_arena_checkpoint( scratch );

  for ( int i = 0; i < max_samples; i++ )
  {
    _ds_arena_checkpoint_t_ inner_cp = ds_arena_checkpoint( scratch );

    _tensor_t *img = load_image( scratch, test_ds, i );
    _tensor_t *lbl = load_label( scratch, test_ds, i );
    if ( !img || !lbl ) break;

    /* wrap single image as [1, 784] */
    int shape[] = { 1, img_pixels };
    _tensor_t *input = tensor_zeros( scratch, 2, shape );
    if ( !input ) break;
    memcpy( input->data, img->data, (size_t)img_pixels * sizeof( float ) );

    _tensor_t *logits = network->forward( network, scratch, input );
    if ( !logits ) break;

    /* find argmax of logits */
    int pred = 0;
    float best_val = logits->data[0];
    for ( int j = 1; j < num_classes; j++ )
      if ( logits->data[j] > best_val )
      {
        best_val = logits->data[j];
        pred = j;
      }

    if ( pred == (int)lbl->data[0] ) correct++;
    total++;

    ds_arena_reset_to( scratch, inner_cp );
  }

  ds_arena_reset_to( scratch, cp );
  return total > 0 ? (float)correct / (float)total : 0.0f;
}

/* Internal: write one metric sample into the ring buffer (mutex held) */
static void push_metric( _training_state_t *state, float loss, float acc )
{
  state->loss_history[state->history_head] = loss;
  state->acc_history[state->history_head] = acc;
  state->history_head = ( state->history_head + 1 ) % TRAINING_HISTORY_CAP;
  if ( state->history_count < TRAINING_HISTORY_CAP ) state->history_count++;
}

/* Training thread entry point */
static int training_thread_fn( void *userdata )
{
  _training_ctx_t *ctx = (_training_ctx_t *)userdata;
  _training_state_t *state = ctx->state;
  _network_t *net = ctx->network;
  _optimizer_t *opt = ctx->optimizer;
  __dataset__ *ds = ctx->train_ds;
  _training_config_t cfg = ctx->config;

  int total_images = (int)ds->image_header.count;
  int img_pixels = IMG_WIDTH * IMG_HEIGHT;
  int total_batches = total_images / cfg.batch_size;

  /* Seed random per thread so weights initialise differently each run */
  srand( (unsigned int)time( NULL ) );

  /* Epoch loop */
  for ( int epoch = 1; epoch <= cfg.epochs; epoch++ )
  {
    /* Check stop signal before starting a new epoch */
    SDL_LockMutex( state->mutex );
    bool stop = state->should_stop;
    SDL_UnlockMutex( state->mutex );
    if ( stop ) break;

    float epoch_loss_sum = 0.0f;
    int epoch_batches = 0;

    /* Batch loop */
    for ( int b = 0; b < total_batches; b++ )
    {
      /* Pause support - spin-wait while paused */
      for ( ;; )
      {
        SDL_LockMutex( state->mutex );
        bool paused = state->is_paused;
        stop = state->should_stop;
        SDL_UnlockMutex( state->mutex );
        if ( stop ) goto training_done;
        if ( !paused ) break;
        SDL_Delay( 50 );
      }

      int batch_start = b * cfg.batch_size;

      // Checkpoint batch arena - reset at end of each batch
      _ds_arena_checkpoint_t_ cp = ds_arena_checkpoint( &ctx->batch_arena );

      // Build input and label batch
      _tensor_t *input = build_image_batch( &ctx->batch_arena, ds, batch_start, cfg.batch_size, img_pixels );
      _tensor_t *labels = build_label_batch( &ctx->batch_arena, ds, batch_start, cfg.batch_size, cfg.num_classes, cfg.label_smoothing );
      if ( !input || !labels )
      {
        ds_arena_reset_to( &ctx->batch_arena, cp );
        continue;
      }

      // forward
      _tensor_t *logits = net->forward( net, &ctx->batch_arena, input );
      if ( !logits )
      {
        ds_arena_reset_to( &ctx->batch_arena, cp );
        continue;
      }

      // loss
      _loss_result_t result = loss_softmax_cross_entropy( &ctx->batch_arena, logits, labels );
      if ( !result.loss )
      {
        ds_arena_reset_to( &ctx->batch_arena, cp );
        continue;
      }

      /* float batch_loss = result.loss->data[0]; */
      float batch_loss = T1( result.loss, 0 );
      epoch_loss_sum += batch_loss;
      epoch_batches++;

      // backward
      network_zero_gradients( net );
      net->backward( net, &ctx->batch_arena, result.gradients );

      // optimizer step
      optimizer_step( opt, net );

      // reclaim batch memory
      ds_arena_reset_to( &ctx->batch_arena, cp );

      // update shared state
      SDL_LockMutex( state->mutex );
      state->epoch = epoch;
      state->total_epochs = cfg.epochs;
      state->batch = b + 1;
      state->total_batches = total_batches;
      push_metric( state, batch_loss, 0.0f ); // acc updated per epoch
      SDL_UnlockMutex( state->mutex );

      // log to stdout every 100 batches
      if ( ( b + 1 ) % 100 == 0 ) printf( "Epoch %d/%d  Batch %d/%d  Loss: %.4f\n", epoch, cfg.epochs, b + 1, total_batches, batch_loss );
    }

    /* End-of-epoch accuracy evaluation (1000 test samples) */
    float epoch_loss = epoch_batches > 0 ? epoch_loss_sum / (float)epoch_batches : 0.0f;

    float epoch_acc = ctx->test_ds ? evaluate_accuracy( net, ctx->test_ds, cfg.num_classes, 1000, &ctx->batch_arena ) : 0.0f;

    printf( "Epoch %d/%d  Loss: %.4f  Acc: %.2f%%\n", epoch, cfg.epochs, epoch_loss, epoch_acc * 100.0f );

    SDL_LockMutex( state->mutex );
    state->epoch_loss = epoch_loss;
    state->epoch_acc = epoch_acc;
    /* Back-fill the last batch metric with the real epoch accuracy */
    int prev = ( state->history_head - 1 + TRAINING_HISTORY_CAP ) % TRAINING_HISTORY_CAP;
    state->acc_history[prev] = epoch_acc;
    SDL_UnlockMutex( state->mutex );
  }

training_done:
  SDL_LockMutex( state->mutex );
  state->is_training = false;
  SDL_UnlockMutex( state->mutex );

  printf( "Training complete.\n" );
  return 0;
}

/* public api */
_training_state_t *training_state_create( _ds_arena_t_ *arena )
{
  _training_state_t *state = ARENA_NEW( arena, _training_state_t );
  state->mutex = SDL_CreateMutex();
  if ( !state->mutex )
  {
    fprintf( stderr, "training_state_create: SDL_CreateMutex failed: %s\n", SDL_GetError() );
    return NULL;
  }
  state->is_training = false;
  state->should_stop = false;
  state->is_paused = false;
  state->history_head = 0;
  state->history_count = 0;
  return state;
}

void training_state_destroy( _training_state_t *state )
{
  if ( state && state->mutex )
  {
    SDL_DestroyMutex( state->mutex );
    state->mutex = NULL;
  }
}

SDL_Thread *training_start( _training_ctx_t *ctx )
{
  SDL_LockMutex( ctx->state->mutex );
  ctx->state->is_training = true;
  ctx->state->should_stop = false;
  ctx->state->is_paused = false;
  ctx->state->epoch = 0;
  ctx->state->batch = 0;
  ctx->state->history_head = 0;
  ctx->state->history_count = 0;
  SDL_UnlockMutex( ctx->state->mutex );

  return SDL_CreateThread( training_thread_fn, "training", ctx );
}

void training_stop( SDL_Thread *thread, _training_state_t *state )
{
  if ( !thread ) return;

  SDL_LockMutex( state->mutex );
  state->should_stop = true;
  state->is_paused = false; /* unblock if paused so thread can exit */
  SDL_UnlockMutex( state->mutex );

  SDL_WaitThread( thread, NULL );
}

void training_read_metrics( _training_state_t *state, _training_metrics_t *out )
{
  SDL_LockMutex( state->mutex );

  out->is_training = state->is_training;
  out->is_paused = state->is_paused;
  out->epoch = state->epoch;
  out->total_epochs = state->total_epochs;
  out->batch = state->batch;
  out->total_batches = state->total_batches;
  out->epoch_loss = state->epoch_loss;
  out->epoch_acc = state->epoch_acc;
  out->history_head = state->history_head;
  out->history_count = state->history_count;

  memcpy( out->loss_history, state->loss_history, sizeof( float ) * TRAINING_HISTORY_CAP );
  memcpy( out->acc_history, state->acc_history, sizeof( float ) * TRAINING_HISTORY_CAP );

  SDL_UnlockMutex( state->mutex );
}
