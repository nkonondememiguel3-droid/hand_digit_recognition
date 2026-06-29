#ifndef hand_digit_recognition_training_h
#define hand_digit_recognition_training_h

#include "dataloader.h"
#include "ds_arena.h"
#include "loses.h"
#include "networkd.h"
#include "optimizer.h"
#include "tensor.h"

#include <SDL3/SDL.h>
#include <stdbool.h>
#include <stdint.h>

/* ── Ring buffer size for the live loss/accuracy curves ──────────────── */
#define TRAINING_HISTORY_CAP 2000

/* ── Training hyperparameters ─────────────────────────────────────────── */
typedef struct
{
  int epochs;
  int batch_size;
  int num_classes;       /* 10 for MNIST                          */
  float label_smoothing; /* 0.0 = hard one-hot, 0.1 = smoothed   */
} _training_config_t;

static inline _training_config_t training_config_defaults( void )
{
  return ( _training_config_t ){
    .epochs = 10,
    .batch_size = 64,
    .num_classes = 10,
    .label_smoothing = 0.0f,
  };
}

/* ── Shared state between training thread and UI thread ──────────────────
 *
 * All fields except `mutex` itself must be accessed under the mutex.
 * The UI thread reads; the training thread writes.
 *
 * loss_history and acc_history are ring buffers — history_head is the
 * index of the next slot to write. Wrap at TRAINING_HISTORY_CAP.
 */
typedef struct
{
  SDL_Mutex *mutex;

  /* Status */
  bool is_training; /* true while the thread is running     */
  bool should_stop; /* UI sets true to request early stop   */
  bool is_paused;   /* UI toggles to pause/resume           */

  /* Progress */
  int epoch; /* 1-based current epoch                */
  int total_epochs;
  int batch;         /* 0-based batch within epoch           */
  int total_batches; /* batches per epoch                    */

  /* Metrics — ring buffers, written after every batch */
  float loss_history[TRAINING_HISTORY_CAP];
  float acc_history[TRAINING_HISTORY_CAP];
  int history_head;  /* next write index                     */
  int history_count; /* total items written (capped)         */

  /* Last completed epoch summary */
  float epoch_loss;
  float epoch_acc;
} _training_state_t;

/* ── Training thread context — passed via SDL_CreateThread ───────────── */
typedef struct
{
  _training_state_t *state; /* shared with UI thread — mutex-protected */
  _network_t *network;
  _optimizer_t *optimizer;
  __dataset__ *train_ds;
  __dataset__ *test_ds; /* used for epoch-end accuracy eval */
  _training_config_t config;

  /* Private arenas owned by the training thread — never touched by UI */
  _ds_arena_t_ weight_arena; /* weights, optimizer state (persistent) */
  _ds_arena_t_ batch_arena;  /* per-batch activations (reset each batch) */
} _training_ctx_t;

/* ── API ──────────────────────────────────────────────────────────────── */

/*
 * training_state_create — allocate and initialise a training state.
 * The mutex is created here. Must be freed with training_state_destroy().
 */
extern _training_state_t *training_state_create( _ds_arena_t_ *arena );

/*
 * training_state_destroy — destroy the mutex.
 * Does NOT free the struct itself (it is arena-owned).
 */
extern void training_state_destroy( _training_state_t *state );

/*
 * training_start — spawn the training thread.
 *
 * Returns the SDL_Thread handle. The caller must eventually call
 * SDL_WaitThread() on it to join cleanly.
 *
 * ctx must remain valid for the entire duration of the thread.
 * ctx->state->should_stop = true signals the thread to exit early.
 */
extern SDL_Thread *training_start( _training_ctx_t *ctx );

/*
 * training_stop — set should_stop and wait for the thread to finish.
 */
extern void training_stop( SDL_Thread *thread, _training_state_t *state );

/*
 * training_read_metrics — safely copy current metrics under the mutex.
 * The UI thread calls this every frame — does a short lock, copies, unlocks.
 */
typedef struct
{
  bool is_training;
  bool is_paused;
  int epoch;
  int total_epochs;
  int batch;
  int total_batches;
  float epoch_loss;
  float epoch_acc;

  /* snapshot of ring buffer — copied so UI can read without holding lock */
  float loss_history[TRAINING_HISTORY_CAP];
  float acc_history[TRAINING_HISTORY_CAP];
  int history_head;
  int history_count;
} _training_metrics_t;

extern void training_read_metrics( _training_state_t *state, _training_metrics_t *out );

#endif /* hand_digit_recognition_training_h */
