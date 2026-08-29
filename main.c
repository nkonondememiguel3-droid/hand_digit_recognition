#include "network_config.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_log.h>
#include <SDL3/SDL_render.h>
#include <SDL3/SDL_surface.h>
#include <SDL3/SDL_video.h>

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#define NK_INCLUDE_COMMAND_USERDATA
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_IMPLEMENTATION
#include "nuklear.h"

#define NK_SDL3_RENDERER_IMPLEMENTATION
#include "nuklear_sdl3_renderer.h"

#include "dataloader.h"
#include "ds_arena.h"
#include "networkd.h"
#include "optimizer.h"
#include "tensor.h"
#include "training.h"

/* Constants */
#define WINDOW_W 1600
#define WINDOW_H 860
#define IMG_NATIVE_SIZE 28
#define NUM_CLASSES 10

#define TRAIN_IMG_PATH "../datasets/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "../datasets/train-labels.idx1-ubyte"
#define TEST_IMG_PATH "../datasets/t10k-images.idx3-ubyte"
#define TEST_LBL_PATH "../datasets/t10k-labels.idx1-ubyte"

#define BG_R 245
#define BG_G 245
#define BG_B 240

/* UI tabs */
typedef enum
{
  TAB_VIEWER = 0,
  TAB_TRAINING,
  TAB_TESTING,
  TAB_CONFIG,
} _app_tab_t;

/* Test result for a single image */
typedef struct
{
  bool has_result;
  int true_label;
  int predicted_label;
  float probabilities[NUM_CLASSES];
  bool is_correct;
} _test_result_t;

/* Full-dataset evaluation state */
typedef struct
{
  bool is_running;
  int total;
  int done;
  int correct;
  float accuracy;
  SDL_Thread *thread;
} _eval_state_t;

/* App state  */
typedef struct
{
  SDL_Window *window;
  SDL_Renderer *renderer;
  bool is_running;

  /* Arenas */
  _ds_arena_t_ arena;
  _ds_arena_t_ scratch_arena;
  _ds_arena_t_ infer_arena;
  _ds_arena_t_ weight_arena;

  /* Datasets — the UI thread reads through these */
  __dataset__ *train_ds;
  __dataset__ *test_ds;

  /* Separate handles owned by the training thread. Seek+read is not atomic
     on a shared FILE*, so every thread that reads a dataset gets its own. */
  __dataset__ *trainer_train_ds;
  __dataset__ *trainer_test_ds;

  /* Viewer */
  uint32_t current_index;
  _tensor_t *current_image;
  uint32_t current_label;
  SDL_Texture *mnist_texture;
  struct nk_image nk_mnist_image;

  /* Network */
  _network_t *network;
  _optimizer_t *optimizer;

  /* Training */
  _training_state_t *training_state;
  _training_ctx_t *training_ctx;
  SDL_Thread *training_thread;
  _training_metrics_t metrics;

  /* Testing */
  int test_index;
  _test_result_t test_result;
  SDL_Texture *test_texture;
  struct nk_image nk_test_image;
  _eval_state_t eval;
  SDL_Mutex *eval_mutex;

  /* UI */
  _app_tab_t active_tab;
  struct nk_context *nk;
  struct nk_font *mono_font;
  struct nk_font *ui_font;

  // architecture configuration
  _network_config_t config;
  int config_editing_layer;
} App;

static void render_config_panel( App *app, float x, float y, float w, float h );
static void apply_network_config( App *app );
static bool training_is_active( App *app );
static bool eval_is_running( App *app );
static bool network_is_idle( App *app );

/* Forward declarations */
static void apply_white_theme( struct nk_context *nk );
static bool navigate_viewer( App *app, int delta );
static bool navigate_test( App *app, int delta );
static SDL_Texture *tensor_to_texture( SDL_Renderer *r, const _tensor_t *img );
static void run_inference( App *app, int idx );
static void render_tab_bar( App *app, float w );
static void render_viewer_panel( App *app, float x, float y, float w, float h );
static void render_training_panel( App *app, float x, float y, float w, float h );
static void render_testing_panel( App *app, float x, float y, float w, float h );

/* ── Shared-state predicates ──────────────────────────────────────────────
 *
 * The network carries one activation cache per layer, so only one thread may
 * run a forward pass at a time. These guard every entry point that touches it.
 */
static bool training_is_active( App *app )
{
  SDL_LockMutex( app->training_state->mutex );
  bool active = app->training_state->is_training;
  SDL_UnlockMutex( app->training_state->mutex );
  return active;
}

static bool eval_is_running( App *app )
{
  SDL_LockMutex( app->eval_mutex );
  bool running = app->eval.is_running;
  SDL_UnlockMutex( app->eval_mutex );
  return running;
}

/* True when no background thread is using the network. */
static bool network_is_idle( App *app )
{
  return app->network != NULL && !training_is_active( app ) && !eval_is_running( app );
}

/* ════════════════════════════════════════════════════════════════════════
 * Entry point
 * ════════════════════════════════════════════════════════════════════════ */
int main( void )
{
  if ( !SDL_Init( SDL_INIT_VIDEO ) )
  {
    SDL_LogError( SDL_LOG_CATEGORY_ERROR, "SDL init failed: %s", SDL_GetError() );
    return SDL_APP_FAILURE;
  }

  /* Seed before the first network is built -- weight initialisation runs
     through rand(), so without this every run starts from the same weights. */
  srand( (unsigned int)time( NULL ) );

  App app;
  memset( &app, 0, sizeof( App ) );
  app.is_running = true;
  app.active_tab = TAB_VIEWER;
  app.test_index = 0;
  app.config = network_config_defaults();
  app.config_editing_layer = -1;

  if ( !SDL_CreateWindowAndRenderer( "Hand Written Digit Recognizer (HWDR)", WINDOW_W, WINDOW_H, SDL_WINDOW_RESIZABLE, &app.window, &app.renderer ) )
  {
    SDL_LogError( SDL_LOG_CATEGORY_ERROR, "Window failed: %s", SDL_GetError() );
    SDL_Quit();
    return SDL_APP_FAILURE;
  }

  app.arena = ds_arena_new( 0 );
  app.scratch_arena = ds_arena_new( 0 );
  app.infer_arena = ds_arena_new( 0 );
  app.weight_arena = ds_arena_new( 0 );

  app.train_ds = dataset_init( &app.arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  app.test_ds = dataset_init( &app.arena, TEST_IMG_PATH, TEST_LBL_PATH );
  app.trainer_train_ds = dataset_init( &app.arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  app.trainer_test_ds = dataset_init( &app.arena, TEST_IMG_PATH, TEST_LBL_PATH );
  if ( !app.train_ds || !app.test_ds || !app.trainer_train_ds || !app.trainer_test_ds )
  {
    SDL_LogError( SDL_LOG_CATEGORY_ERROR, "Failed to open datasets" );
    return SDL_APP_FAILURE;
  }

  if ( !navigate_viewer( &app, 0 ) )
  {
    SDL_LogError( SDL_LOG_CATEGORY_ERROR, "Failed to load first image" );
    return SDL_APP_FAILURE;
  }

  app.training_state = training_state_create( &app.arena );
  if ( !app.training_state )
  {
    SDL_LogError( SDL_LOG_CATEGORY_ERROR, "Failed to create training state" );
    return SDL_APP_FAILURE;
  }
  app.eval_mutex = SDL_CreateMutex();

  apply_network_config( &app );

  app.nk = nk_sdl_init( app.window, app.renderer, nk_sdl_allocator() );
  {
    struct nk_font_atlas *atlas = nk_sdl_font_stash_begin( app.nk );
    struct nk_font_config cfg = nk_font_config( 0 );
    app.mono_font = nk_font_atlas_add_default( atlas, 14, &cfg );
    app.ui_font = nk_font_atlas_add_default( atlas, 17, NULL );
    nk_sdl_font_stash_end( app.nk );
    nk_style_set_font( app.nk, &app.ui_font->handle );
  }
  apply_white_theme( app.nk );

  SDL_Event event;
  nk_input_begin( app.nk );

  while ( app.is_running )
  {
    while ( SDL_PollEvent( &event ) )
    {
      if ( event.type == SDL_EVENT_QUIT ) app.is_running = false;

      if ( event.type == SDL_EVENT_KEY_DOWN )
      {
        switch ( event.key.key )
        {
        case SDLK_ESCAPE:
          app.is_running = false;
          break;
        case SDLK_RIGHT:
          if ( app.active_tab == TAB_VIEWER ) navigate_viewer( &app, +1 );
          if ( app.active_tab == TAB_TESTING ) navigate_test( &app, +1 );
          break;
        case SDLK_LEFT:
          if ( app.active_tab == TAB_VIEWER ) navigate_viewer( &app, -1 );
          if ( app.active_tab == TAB_TESTING ) navigate_test( &app, -1 );
          break;
        case SDLK_PAGEDOWN:
          if ( app.active_tab == TAB_VIEWER ) navigate_viewer( &app, +100 );
          if ( app.active_tab == TAB_TESTING ) navigate_test( &app, +100 );
          break;
        case SDLK_PAGEUP:
          if ( app.active_tab == TAB_VIEWER ) navigate_viewer( &app, -100 );
          if ( app.active_tab == TAB_TESTING ) navigate_test( &app, -100 );
          break;
          /* case TAB_CONFIG: */
          /*   render_config_panel( &app, 0, content_y, (float)win_w, content_h ); */
          /*   break; */
        }
      }

      SDL_ConvertEventToRenderCoordinates( app.renderer, &event );
      nk_sdl_handle_event( app.nk, &event );
    }
    nk_input_end( app.nk );

    training_read_metrics( app.training_state, &app.metrics );

    /* Join workers as soon as they finish, so a handle is never overwritten
       by the next run and the thread's resources are released. */
    if ( app.training_thread && !app.metrics.is_training )
    {
      SDL_WaitThread( app.training_thread, NULL );
      app.training_thread = NULL;
    }
    if ( app.eval.thread && !eval_is_running( &app ) )
    {
      SDL_WaitThread( app.eval.thread, NULL );
      app.eval.thread = NULL;
    }

    int win_w, win_h;
    SDL_GetWindowSize( app.window, &win_w, &win_h );

    float tab_h = 44.0f;
    float content_y = tab_h;
    float content_h = (float)win_h - tab_h;

    render_tab_bar( &app, (float)win_w );

    switch ( app.active_tab )
    {
    case TAB_VIEWER: {
      float lw = (float)win_w * 0.45f;
      render_viewer_panel( &app, 0, content_y, lw, content_h );
      render_training_panel( &app, lw, content_y, (float)win_w - lw, content_h );
      break;
    }
    case TAB_TRAINING:
      render_training_panel( &app, 0, content_y, (float)win_w, content_h );
      break;
    case TAB_TESTING:
      render_testing_panel( &app, 0, content_y, (float)win_w, content_h );
      break;
    case TAB_CONFIG:
      render_config_panel( &app, 0, content_y, (float)win_w, content_h );
      break;
    }

    SDL_SetRenderDrawColor( app.renderer, BG_R, BG_G, BG_B, 255 );
    SDL_RenderClear( app.renderer );
    nk_sdl_render( app.nk, NK_ANTI_ALIASING_ON );
    nk_sdl_update_TextInput( app.nk );
    SDL_RenderPresent( app.renderer );

    nk_input_begin( app.nk );
  }

  /* Cleanup */
  if ( app.training_thread ) training_stop( app.training_thread, app.training_state );

  if ( app.eval.thread )
  {
    SDL_LockMutex( app.eval_mutex );
    app.eval.is_running = false;
    SDL_UnlockMutex( app.eval_mutex );
    SDL_WaitThread( app.eval.thread, NULL );
  }

  training_state_destroy( app.training_state );
  SDL_DestroyMutex( app.eval_mutex );

  if ( app.mnist_texture ) SDL_DestroyTexture( app.mnist_texture );
  if ( app.test_texture ) SDL_DestroyTexture( app.test_texture );

  nk_input_end( app.nk );
  nk_sdl_shutdown( app.nk );
  SDL_DestroyRenderer( app.renderer );
  SDL_DestroyWindow( app.window );

  if ( app.train_ds ) dataset_close( app.train_ds );
  if ( app.test_ds ) dataset_close( app.test_ds );
  if ( app.trainer_train_ds ) dataset_close( app.trainer_train_ds );
  if ( app.trainer_test_ds ) dataset_close( app.trainer_test_ds );

  ds_arena_destroy( &app.infer_arena );
  ds_arena_destroy( &app.scratch_arena );
  ds_arena_destroy( &app.weight_arena );
  ds_arena_destroy( &app.arena );
  SDL_Quit();
  return SDL_APP_SUCCESS;
}

/* ════════════════════════════════════════════════════════════════════════
 * Inference
 * ════════════════════════════════════════════════════════════════════════ */
static void softmax_inplace( float *logits, float *probs, int n )
{
  float mx = logits[0];
  for ( int i = 1; i < n; i++ )
    if ( logits[i] > mx ) mx = logits[i];
  float s = 0.0f;
  for ( int i = 0; i < n; i++ )
  {
    probs[i] = expf( logits[i] - mx );
    s += probs[i];
  }
  for ( int i = 0; i < n; i++ ) probs[i] /= s;
}

static void run_inference( App *app, int idx )
{
  /* The training and eval threads mutate every layer's activation cache.
     Running a forward pass alongside them corrupts both. */
  if ( !network_is_idle( app ) ) return;

  _ds_arena_checkpoint_t_ cp = ds_arena_checkpoint( &app->infer_arena );

  _tensor_t *img = load_image( &app->infer_arena, app->test_ds, idx );
  _tensor_t *lbl = load_label( &app->infer_arena, app->test_ds, idx );
  if ( !img || !lbl )
  {
    ds_arena_reset_to( &app->infer_arena, cp );
    return;
  }

  if ( app->test_texture ) SDL_DestroyTexture( app->test_texture );
  app->test_texture = tensor_to_texture( app->renderer, img );
  app->nk_test_image = nk_image_ptr( app->test_texture );

  int flat_shape[] = { 1, IMG_NATIVE_SIZE * IMG_NATIVE_SIZE };
  _tensor_t *input = tensor_zeros( &app->infer_arena, 2, flat_shape );
  if ( !input )
  {
    ds_arena_reset_to( &app->infer_arena, cp );
    return;
  }
  memcpy( input->data, img->data, (size_t)( IMG_NATIVE_SIZE * IMG_NATIVE_SIZE ) * sizeof( float ) );

  _tensor_t *logits = app->network->forward( app->network, &app->infer_arena, input );
  if ( !logits )
  {
    ds_arena_reset_to( &app->infer_arena, cp );
    return;
  }

  _test_result_t *r = &app->test_result;
  r->true_label = (int)lbl->data[0];
  softmax_inplace( logits->data, r->probabilities, NUM_CLASSES );

  r->predicted_label = 0;
  float best = r->probabilities[0];
  for ( int j = 1; j < NUM_CLASSES; j++ )
    if ( r->probabilities[j] > best )
    {
      best = r->probabilities[j];
      r->predicted_label = j;
    }

  r->is_correct = ( r->predicted_label == r->true_label );
  r->has_result = true;

  ds_arena_reset_to( &app->infer_arena, cp );
}

/* Full-dataset evaluation thread */
static int eval_thread_fn( void *ud )
{
  App *app = (App *)ud;
  _ds_arena_t_ ea = ds_arena_new( 0 );

  /* Private handles: seek+read is not atomic on a FILE* shared with the UI */
  __dataset__ *test_ds = dataset_init( &ea, TEST_IMG_PATH, TEST_LBL_PATH );
  if ( !test_ds )
  {
    SDL_LogError( SDL_LOG_CATEGORY_APPLICATION, "Eval: failed to open the test dataset" );
    SDL_LockMutex( app->eval_mutex );
    app->eval.is_running = false;
    SDL_UnlockMutex( app->eval_mutex );
    ds_arena_destroy( &ea );
    return 0;
  }

  int total = (int)test_ds->image_header.count;

  SDL_LockMutex( app->eval_mutex );
  app->eval.total = total;
  app->eval.done = 0;
  app->eval.correct = 0;
  app->eval.accuracy = 0.0f;
  SDL_UnlockMutex( app->eval_mutex );

  for ( int i = 0; i < total; i++ )
  {
    SDL_LockMutex( app->eval_mutex );
    bool stop = !app->eval.is_running;
    SDL_UnlockMutex( app->eval_mutex );
    if ( stop ) break;

    _ds_arena_checkpoint_t_ cp = ds_arena_checkpoint( &ea );

    _tensor_t *img = load_image( &ea, test_ds, i );
    _tensor_t *lbl = load_label( &ea, test_ds, i );
    if ( !img || !lbl )
    {
      ds_arena_reset_to( &ea, cp );
      continue;
    }

    int shape[] = { 1, IMG_NATIVE_SIZE * IMG_NATIVE_SIZE };
    _tensor_t *inp = tensor_zeros( &ea, 2, shape );
    if ( !inp )
    {
      ds_arena_reset_to( &ea, cp );
      continue;
    }
    memcpy( inp->data, img->data, (size_t)( IMG_NATIVE_SIZE * IMG_NATIVE_SIZE ) * sizeof( float ) );

    _tensor_t *logits = app->network->forward( app->network, &ea, inp );
    if ( !logits )
    {
      ds_arena_reset_to( &ea, cp );
      continue;
    }

    int pred = 0;
    float bv = logits->data[0];
    for ( int j = 1; j < NUM_CLASSES; j++ )
      if ( logits->data[j] > bv )
      {
        bv = logits->data[j];
        pred = j;
      }

    int correct = ( pred == (int)lbl->data[0] ) ? 1 : 0;

    SDL_LockMutex( app->eval_mutex );
    app->eval.done++;
    app->eval.correct += correct;
    app->eval.accuracy = (float)app->eval.correct / (float)app->eval.done;
    SDL_UnlockMutex( app->eval_mutex );

    ds_arena_reset_to( &ea, cp );
  }

  SDL_LockMutex( app->eval_mutex );
  app->eval.is_running = false;
  SDL_UnlockMutex( app->eval_mutex );

  dataset_close( test_ds );
  ds_arena_destroy( &ea );
  return 0;
}

/* ════════════════════════════════════════════════════════════════════════
 * Navigation
 * ════════════════════════════════════════════════════════════════════════ */
static SDL_Texture *tensor_to_texture( SDL_Renderer *r, const _tensor_t *img )
{
  uint8_t rgba[IMG_NATIVE_SIZE * IMG_NATIVE_SIZE * 4];
  for ( int i = 0; i < IMG_NATIVE_SIZE * IMG_NATIVE_SIZE; i++ )
  {
    uint8_t v = (uint8_t)( img->data[i] * 255.0f + 0.5f );
    rgba[i * 4 + 0] = v;
    rgba[i * 4 + 1] = v;
    rgba[i * 4 + 2] = v;
    rgba[i * 4 + 3] = 255;
  }
  /* RGBA32 is the byte-order-correct alias for the {r,g,b,a} bytes written
     above; RGBA8888 is a packed format and reads them reversed on a
     little-endian host. */
  SDL_Texture *tex = SDL_CreateTexture( r, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STATIC, IMG_NATIVE_SIZE, IMG_NATIVE_SIZE );
  if ( !tex ) return NULL;
  SDL_SetTextureScaleMode( tex, SDL_SCALEMODE_NEAREST );
  SDL_UpdateTexture( tex, NULL, rgba, IMG_NATIVE_SIZE * 4 );
  return tex;
}

static bool navigate_viewer( App *app, int delta )
{
  uint32_t count = app->train_ds->image_header.count;
  app->current_index = ( app->current_index + count + (uint32_t)delta ) % count;

  _ds_arena_t_ ns = ds_arena_new( 0 );
  _tensor_t *img = load_image( &ns, app->train_ds, (int)app->current_index );
  _tensor_t *lbl = load_label( &ns, app->train_ds, (int)app->current_index );
  if ( !img || !lbl )
  {
    ds_arena_destroy( &ns );
    return false;
  }

  ds_arena_destroy( &app->scratch_arena );
  app->scratch_arena = ns;
  app->current_image = img;
  app->current_label = (uint32_t)lbl->data[0];

  if ( app->mnist_texture ) SDL_DestroyTexture( app->mnist_texture );
  app->mnist_texture = tensor_to_texture( app->renderer, img );
  app->nk_mnist_image = nk_image_ptr( app->mnist_texture );
  return true;
}

static bool navigate_test( App *app, int delta )
{
  int count = (int)app->test_ds->image_header.count;
  app->test_index = ( app->test_index + count + delta ) % count;
  run_inference( app, app->test_index );
  return true;
}

/* ════════════════════════════════════════════════════════════════════════
 * Tab bar
 * ════════════════════════════════════════════════════════════════════════ */
static void render_tab_bar( App *app, float w )
{
  if ( nk_begin( app->nk, "tabs", nk_rect( 0, 0, w, 44 ), NK_WINDOW_NO_SCROLLBAR ) )
  {
    nk_layout_row_static( app->nk, 36, (int)( w / 4 ) - 4, 4 );
    if ( nk_button_label( app->nk, "Viewer + Training" ) ) app->active_tab = TAB_VIEWER;
    if ( nk_button_label( app->nk, "Training" ) ) app->active_tab = TAB_TRAINING;
    if ( nk_button_label( app->nk, "Testing" ) ) app->active_tab = TAB_TESTING;
    if ( nk_button_label( app->nk, "Architecture" ) ) app->active_tab = TAB_CONFIG;
  }
  nk_end( app->nk );
}

/* ════════════════════════════════════════════════════════════════════════
 * Viewer panel
 * ════════════════════════════════════════════════════════════════════════ */
static void render_viewer_panel( App *app, float x, float y, float w, float h )
{
  if ( !nk_begin( app->nk, "viewer", nk_rect( x, y, w, h ), NK_WINDOW_BORDER | NK_WINDOW_NO_SCROLLBAR ) )
  {
    nk_end( app->nk );
    return;
  }

  uint32_t total = app->train_ds->image_header.count;

  nk_layout_row_dynamic( app->nk, 28, 1 );
  nk_labelf( app->nk, NK_TEXT_CENTERED, "Image %u / %u     Label: %u", app->current_index + 1, total, app->current_label );

  nk_layout_row_dynamic( app->nk, 32, 2 );
  if ( nk_button_label( app->nk, "<  Prev" ) ) navigate_viewer( app, -1 );
  if ( nk_button_label( app->nk, "Next  >" ) ) navigate_viewer( app, +1 );

  float ch = h - 100.0f;
  if ( ch < 80.0f ) ch = 80.0f;

  nk_layout_row_begin( app->nk, NK_DYNAMIC, ch, 2 );

  struct nk_style_item orig = app->nk->style.window.fixed_background;
  app->nk->style.window.fixed_background = nk_style_item_color( nk_rgb( 0, 0, 0 ) );

  nk_layout_row_push( app->nk, 0.42f );
  if ( nk_group_begin( app->nk, "img_g", NK_WINDOW_BORDER | NK_WINDOW_TITLE ) )
  {
    nk_layout_row_dynamic( app->nk, ch - 38.0f, 1 );
    nk_image( app->nk, app->nk_mnist_image );
    nk_group_end( app->nk );
  }
  app->nk->style.window.fixed_background = orig;

  nk_layout_row_push( app->nk, 0.58f );
  if ( nk_group_begin( app->nk, "pix_g", NK_WINDOW_BORDER | NK_WINDOW_TITLE ) )
  {
    nk_style_set_font( app->nk, &app->mono_font->handle );
    uint32_t rows = app->train_ds->rows;
    uint32_t cols = app->train_ds->cols;
    const _tensor_t *img = app->current_image;

    for ( uint32_t r = 0; r < rows; r++ )
    {
      nk_layout_row_static( app->nk, 16, 18, (int)cols );
      for ( uint32_t c = 0; c < cols; c++ )
      {
        uint8_t v = (uint8_t)( img->data[r * cols + c] * 255.0f + 0.5f );
        char buf[4];
        SDL_snprintf( buf, sizeof( buf ), "%u", v );
        nk_label( app->nk, buf, NK_TEXT_CENTERED );
      }
    }
    nk_style_set_font( app->nk, &app->ui_font->handle );
    nk_group_end( app->nk );
  }
  nk_layout_row_end( app->nk );
  nk_end( app->nk );
}

/* ════════════════════════════════════════════════════════════════════════
 * Training panel
 * ════════════════════════════════════════════════════════════════════════ */
static void render_training_panel( App *app, float x, float y, float w, float h )
{
  _training_metrics_t *m = &app->metrics;

  if ( !nk_begin( app->nk, "training", nk_rect( x, y, w, h ), NK_WINDOW_BORDER | NK_WINDOW_NO_SCROLLBAR ) )
  {
    nk_end( app->nk );
    return;
  }

  nk_layout_row_dynamic( app->nk, 22, 1 );
  char arch_summary[256];
  network_config_summary( &app->config, arch_summary, sizeof( arch_summary ) );
  nk_labelf( app->nk, NK_TEXT_LEFT, "Network: %s  (%s lr=%.5f)", arch_summary, app->config.optimizer == CONFIG_OPTIMIZER_ADAM ? "Adam" : "SGD",
             app->config.learning_rate );
  nk_layout_row_dynamic( app->nk, 36, 3 );

  if ( !m->is_training )
  {
    bool can_start = app->config.is_applied && !app->config.is_dirty && app->network != NULL && !eval_is_running( app );

    if ( !can_start )
    {
      if ( eval_is_running( app ) ) nk_label( app->nk, "Full evaluation is running -- wait for it to finish.", NK_TEXT_LEFT );
      else nk_label( app->nk, "Configure and Apply a network in the Architecture tab first.", NK_TEXT_LEFT );
      nk_label( app->nk, "", NK_TEXT_LEFT );
      nk_label( app->nk, "", NK_TEXT_LEFT );
    }
    else
    {
      if ( nk_button_label( app->nk, "Start Training" ) )
      {
        /* The previous run's thread is joined in the main loop as soon as it
           reports it has stopped, so the handle is free here. */
        app->training_ctx = ARENA_NEW( &app->weight_arena, _training_ctx_t );
        app->training_ctx->state = app->training_state;
        app->training_ctx->network = app->network;
        app->training_ctx->optimizer = app->optimizer;
        app->training_ctx->train_ds = app->trainer_train_ds;
        app->training_ctx->test_ds = app->trainer_test_ds;
        app->training_ctx->config = training_config_defaults();
        app->training_ctx->batch_arena = ds_arena_new( 0 );
        app->training_thread = training_start( app->training_ctx );
      }
      nk_label( app->nk, "", NK_TEXT_LEFT );
      nk_label( app->nk, "", NK_TEXT_LEFT );
    }
  }
  else
  {
    if ( nk_button_label( app->nk, "Stop" ) )
    {
      training_stop( app->training_thread, app->training_state );
      app->training_thread = NULL;
    }
    const char *pl = m->is_paused ? "Resume" : "Pause";
    if ( nk_button_label( app->nk, pl ) )
    {
      SDL_LockMutex( app->training_state->mutex );
      app->training_state->is_paused = !app->training_state->is_paused;
      SDL_UnlockMutex( app->training_state->mutex );
    }
    nk_label( app->nk, "", NK_TEXT_LEFT );
  }

  nk_layout_row_dynamic( app->nk, 22, 1 );
  if ( m->is_training )
  {
    nk_labelf( app->nk, NK_TEXT_LEFT, "Epoch %d / %d    Batch %d / %d", m->epoch, m->total_epochs, m->batch, m->total_batches );
    nk_labelf( app->nk, NK_TEXT_LEFT, "Epoch loss: %.4f    Epoch acc: %.2f%%", m->epoch_loss, m->epoch_acc * 100.0f );
    nk_layout_row_dynamic( app->nk, 16, 1 );
    float prog = m->total_batches > 0 ? (float)m->batch / (float)m->total_batches : 0.0f;
    nk_size pv = (nk_size)( prog * 100.0f );
    nk_progress( app->nk, &pv, 100, NK_FIXED );
  }
  else
  {
    nk_label( app->nk, "Not training", NK_TEXT_LEFT );
    nk_layout_row_dynamic( app->nk, 16, 1 );
    nk_label( app->nk, "", NK_TEXT_LEFT );
    nk_label( app->nk, "", NK_TEXT_LEFT );
  }

  float chart_h = ( h - 260.0f ) * 0.5f;
  if ( chart_h < 70.0f ) chart_h = 70.0f;

  nk_layout_row_dynamic( app->nk, 18, 1 );
  nk_label( app->nk, "Loss (per batch)", NK_TEXT_LEFT );
  nk_layout_row_dynamic( app->nk, chart_h, 1 );
  if ( m->history_count > 1 && nk_chart_begin( app->nk, NK_CHART_LINES, m->history_count, 0.0f, 5.0f ) )
  {
    int start = m->history_count >= TRAINING_HISTORY_CAP ? m->history_head : 0;
    int n = m->history_count >= TRAINING_HISTORY_CAP ? TRAINING_HISTORY_CAP : m->history_count;
    for ( int i = 0; i < n; i++ ) nk_chart_push( app->nk, m->loss_history[( start + i ) % TRAINING_HISTORY_CAP] );
    nk_chart_end( app->nk );
  }

  nk_layout_row_dynamic( app->nk, 18, 1 );
  nk_label( app->nk, "Accuracy (per epoch)", NK_TEXT_LEFT );
  nk_layout_row_dynamic( app->nk, chart_h, 1 );
  if ( m->history_count > 1 && nk_chart_begin( app->nk, NK_CHART_LINES, m->history_count, 0.0f, 1.0f ) )
  {
    int start = m->history_count >= TRAINING_HISTORY_CAP ? m->history_head : 0;
    int n = m->history_count >= TRAINING_HISTORY_CAP ? TRAINING_HISTORY_CAP : m->history_count;
    for ( int i = 0; i < n; i++ ) nk_chart_push( app->nk, m->acc_history[( start + i ) % TRAINING_HISTORY_CAP] );
    nk_chart_end( app->nk );
  }

  nk_end( app->nk );
}

/* ════════════════════════════════════════════════════════════════════════
 * Testing panel
 * ════════════════════════════════════════════════════════════════════════ */
static void render_testing_panel( App *app, float x, float y, float w, float h )
{
  if ( !nk_begin( app->nk, "testing", nk_rect( x, y, w, h ), NK_WINDOW_BORDER | NK_WINDOW_NO_SCROLLBAR ) )
  {
    nk_end( app->nk );
    return;
  }

  uint32_t test_count = app->test_ds->image_header.count;

  bool training_active = training_is_active( app );
  bool eval_running = eval_is_running( app );
  bool network_busy = training_active || eval_running || app->network == NULL;

  /* Top controls */
  nk_layout_row_dynamic( app->nk, 32, 5 );
  if ( nk_button_label( app->nk, "<  Prev" ) ) navigate_test( app, -1 );
  nk_labelf( app->nk, NK_TEXT_CENTERED, "%d / %u", app->test_index + 1, test_count );
  if ( nk_button_label( app->nk, "Next  >" ) ) navigate_test( app, +1 );

  if ( network_busy ) nk_label( app->nk, "", NK_TEXT_LEFT );
  else if ( nk_button_label( app->nk, "Run Inference" ) ) run_inference( app, app->test_index );

  /* Full-eval button */
  if ( eval_running )
  {
    if ( nk_button_label( app->nk, "Stop Eval" ) )
    {
      SDL_LockMutex( app->eval_mutex );
      app->eval.is_running = false;
      SDL_UnlockMutex( app->eval_mutex );
    }
  }
  else if ( network_busy ) { nk_label( app->nk, "", NK_TEXT_LEFT ); }
  else if ( nk_button_label( app->nk, "Eval All (10k)" ) )
  {
    SDL_LockMutex( app->eval_mutex );
    app->eval.is_running = true;
    SDL_UnlockMutex( app->eval_mutex );
    app->eval.thread = SDL_CreateThread( eval_thread_fn, "eval", app );
  }

  if ( network_busy )
  {
    nk_layout_row_dynamic( app->nk, 22, 1 );
    if ( app->network == NULL )
      nk_label_colored( app->nk, "Apply a network in the Architecture tab before testing.", NK_TEXT_LEFT, nk_rgb( 220, 60, 60 ) );
    else if ( training_active )
      nk_label_colored( app->nk, "Training is running -- stop it to test single images.", NK_TEXT_LEFT, nk_rgb( 220, 150, 40 ) );
    else nk_label_colored( app->nk, "Full evaluation is running -- single-image testing is paused.", NK_TEXT_LEFT, nk_rgb( 220, 150, 40 ) );
  }

  /* Full-eval progress bar */
  SDL_LockMutex( app->eval_mutex );
  int eval_done = app->eval.done;
  int eval_total = app->eval.total;
  int eval_correct = app->eval.correct;
  float eval_acc = app->eval.accuracy;
  bool eval_still = app->eval.is_running;
  SDL_UnlockMutex( app->eval_mutex );

  nk_layout_row_dynamic( app->nk, 22, 1 );
  if ( eval_total > 0 )
  {
    nk_labelf( app->nk, NK_TEXT_LEFT, "Full eval: %d / %d   Correct: %d   Accuracy: %.2f%%  %s", eval_done, eval_total, eval_correct,
               eval_acc * 100.0f, eval_still ? "(running...)" : "(done)" );
  }
  else { nk_label( app->nk, "Press 'Eval All' to evaluate accuracy on the full 10k test set.", NK_TEXT_LEFT ); }

  nk_layout_row_dynamic( app->nk, 14, 1 );
  if ( eval_total > 0 )
  {
    nk_size pv = (nk_size)( eval_total > 0 ? (float)eval_done / (float)eval_total * 100.0f : 0.0f );
    nk_progress( app->nk, &pv, 100, NK_FIXED );
  }
  else { nk_label( app->nk, "", NK_TEXT_LEFT ); }

  /* Main content */
  float ch = h - 130.0f;
  if ( ch < 80.0f ) ch = 80.0f;

  nk_layout_row_begin( app->nk, NK_DYNAMIC, ch, 2 );

  /* LEFT: test image */
  nk_layout_row_push( app->nk, 0.25f );

  struct nk_style_item orig = app->nk->style.window.fixed_background;
  app->nk->style.window.fixed_background = nk_style_item_color( nk_rgb( 0, 0, 0 ) );
  if ( nk_group_begin( app->nk, "timg_g", NK_WINDOW_BORDER | NK_WINDOW_TITLE ) )
  {
    if ( app->test_result.has_result && app->test_texture )
    {
      nk_layout_row_dynamic( app->nk, ch - 40.0f, 1 );
      nk_image( app->nk, app->nk_test_image );
    }
    else
    {
      nk_layout_row_dynamic( app->nk, 26, 1 );
      nk_label( app->nk, "No image yet", NK_TEXT_CENTERED );
    }
    nk_group_end( app->nk );
  }
  app->nk->style.window.fixed_background = orig;

  /* RIGHT: results */
  nk_layout_row_push( app->nk, 0.75f );
  if ( nk_group_begin( app->nk, "tres_g", NK_WINDOW_BORDER | NK_WINDOW_TITLE ) )
  {
    _test_result_t *r = &app->test_result;

    if ( !r->has_result )
    {
      nk_layout_row_dynamic( app->nk, 26, 1 );
      nk_label( app->nk, "Navigate to a test image and press 'Run Inference'.", NK_TEXT_LEFT );
    }
    else
    {
      /* Verdict */
      nk_layout_row_dynamic( app->nk, 34, 1 );
      if ( r->is_correct ) nk_label_colored( app->nk, "CORRECT", NK_TEXT_CENTERED, nk_rgb( 60, 180, 60 ) );
      else nk_label_colored( app->nk, "WRONG", NK_TEXT_CENTERED, nk_rgb( 220, 60, 60 ) );

      /* Summary */
      nk_layout_row_dynamic( app->nk, 22, 3 );
      nk_labelf( app->nk, NK_TEXT_LEFT, "True label:  %d", r->true_label );
      nk_labelf( app->nk, NK_TEXT_LEFT, "Predicted:   %d", r->predicted_label );
      nk_labelf( app->nk, NK_TEXT_LEFT, "Confidence:  %.2f%%", r->probabilities[r->predicted_label] * 100.0f );

      /* Probability bars */
      nk_layout_row_dynamic( app->nk, 18, 1 );
      nk_label( app->nk, "Class probabilities:", NK_TEXT_LEFT );

      for ( int cls = 0; cls < NUM_CLASSES; cls++ )
      {
        float prob = r->probabilities[cls];
        bool is_pred = ( cls == r->predicted_label );
        bool is_truth = ( cls == r->true_label );

        /* Digit label */
        nk_layout_row_begin( app->nk, NK_DYNAMIC, 22, 3 );
        nk_layout_row_push( app->nk, 0.06f );

        char digit_buf[3] = { '0' + (char)cls, 0, 0 };
        if ( is_pred && is_truth ) nk_label_colored( app->nk, digit_buf, NK_TEXT_CENTERED, nk_rgb( 60, 160, 60 ) ); /* green: correct pred */
        else if ( is_pred ) nk_label_colored( app->nk, digit_buf, NK_TEXT_CENTERED, nk_rgb( 220, 60, 60 ) );        /* red: wrong pred */
        else if ( is_truth ) nk_label_colored( app->nk, digit_buf, NK_TEXT_CENTERED, nk_rgb( 60, 100, 220 ) );      /* blue: true label */
        else nk_label( app->nk, digit_buf, NK_TEXT_CENTERED );

        /* Progress bar — maps [0, 1] to [0, 1000] for resolution */
        nk_layout_row_push( app->nk, 0.78f );
        nk_size bar_val = (nk_size)( prob * 1000.0f );
        nk_progress( app->nk, &bar_val, 1000, NK_FIXED );

        /* Percentage */
        nk_layout_row_push( app->nk, 0.16f );
        nk_labelf( app->nk, NK_TEXT_RIGHT, "%.2f%%", prob * 100.0f );

        nk_layout_row_end( app->nk );
      }

      /* Colour legend */
      nk_layout_row_dynamic( app->nk, 16, 1 );
      nk_label( app->nk, "", NK_TEXT_LEFT );
      nk_layout_row_dynamic( app->nk, 18, 3 );
      nk_label_colored( app->nk, "Green = correct prediction", NK_TEXT_LEFT, nk_rgb( 60, 160, 60 ) );
      nk_label_colored( app->nk, "Red = wrong prediction", NK_TEXT_LEFT, nk_rgb( 220, 60, 60 ) );
      nk_label_colored( app->nk, "Blue = true label", NK_TEXT_LEFT, nk_rgb( 60, 100, 220 ) );
    }
    nk_group_end( app->nk );
  }

  nk_layout_row_end( app->nk );
  nk_end( app->nk );
}

/* ════════════════════════════════════════════════════════════════════════
 * Theme
 * ════════════════════════════════════════════════════════════════════════ */
static void apply_white_theme( struct nk_context *nk )
{
  /* Zero-initialised so that any colour this file does not name explicitly
     is still defined -- nk_style_from_table reads all NK_COLOR_COUNT entries,
     and this Nuklear carries a knob family past NK_COLOR_TAB_HEADER. */
  struct nk_color table[NK_COLOR_COUNT] = { 0 };
  nk_style_default( nk );
  table[NK_COLOR_TEXT] = nk_rgb( 30, 30, 30 );
  table[NK_COLOR_WINDOW] = nk_rgb( 250, 250, 248 );
  table[NK_COLOR_HEADER] = nk_rgb( 230, 230, 225 );
  table[NK_COLOR_BORDER] = nk_rgb( 200, 200, 195 );
  table[NK_COLOR_BUTTON] = nk_rgb( 225, 225, 220 );
  table[NK_COLOR_BUTTON_HOVER] = nk_rgb( 210, 210, 205 );
  table[NK_COLOR_BUTTON_ACTIVE] = nk_rgb( 195, 195, 190 );
  table[NK_COLOR_TOGGLE] = nk_rgb( 230, 230, 225 );
  table[NK_COLOR_TOGGLE_HOVER] = nk_rgb( 215, 215, 210 );
  table[NK_COLOR_TOGGLE_CURSOR] = nk_rgb( 180, 180, 175 );
  table[NK_COLOR_SELECT] = nk_rgb( 235, 235, 230 );
  table[NK_COLOR_SELECT_ACTIVE] = nk_rgb( 200, 200, 195 );
  table[NK_COLOR_SLIDER] = nk_rgb( 220, 220, 215 );
  table[NK_COLOR_SLIDER_CURSOR] = nk_rgb( 180, 180, 175 );
  table[NK_COLOR_SLIDER_CURSOR_HOVER] = nk_rgb( 165, 165, 160 );
  table[NK_COLOR_SLIDER_CURSOR_ACTIVE] = nk_rgb( 150, 150, 145 );
  table[NK_COLOR_PROPERTY] = nk_rgb( 225, 225, 220 );
  table[NK_COLOR_EDIT] = nk_rgb( 255, 255, 255 );
  table[NK_COLOR_EDIT_CURSOR] = nk_rgb( 30, 30, 30 );
  table[NK_COLOR_COMBO] = nk_rgb( 225, 225, 220 );
  table[NK_COLOR_CHART] = nk_rgb( 235, 235, 230 );
  table[NK_COLOR_CHART_COLOR] = nk_rgb( 100, 140, 220 );
  table[NK_COLOR_CHART_COLOR_HIGHLIGHT] = nk_rgb( 220, 80, 80 );
  table[NK_COLOR_SCROLLBAR] = nk_rgb( 235, 235, 230 );
  table[NK_COLOR_SCROLLBAR_CURSOR] = nk_rgb( 190, 190, 185 );
  table[NK_COLOR_SCROLLBAR_CURSOR_HOVER] = nk_rgb( 170, 170, 165 );
  table[NK_COLOR_SCROLLBAR_CURSOR_ACTIVE] = nk_rgb( 150, 150, 145 );
  table[NK_COLOR_TAB_HEADER] = nk_rgb( 230, 230, 225 );
  table[NK_COLOR_KNOB] = nk_rgb( 230, 230, 225 );
  table[NK_COLOR_KNOB_CURSOR] = nk_rgb( 180, 180, 175 );
  table[NK_COLOR_KNOB_CURSOR_HOVER] = nk_rgb( 165, 165, 160 );
  table[NK_COLOR_KNOB_CURSOR_ACTIVE] = nk_rgb( 150, 150, 145 );
  nk_style_from_table( nk, table );
}

/* ════════════════════════════════════════════════════════════════════════
 * apply_network_config — rebuild network + optimizer from current config
 * ════════════════════════════════════════════════════════════════════════ */
static void apply_network_config( App *app )
{
  /* Refuse to rebuild while a worker thread is running — both the training
     and the evaluation thread hold raw pointers into the old network and
     weight_arena, and swapping them out underneath is a use-after-free. */
  if ( training_is_active( app ) )
  {
    SDL_LogWarn( SDL_LOG_CATEGORY_APPLICATION, "Cannot apply config while training is in progress" );
    return;
  }

  if ( eval_is_running( app ) )
  {
    SDL_LogWarn( SDL_LOG_CATEGORY_APPLICATION, "Cannot apply config while a full evaluation is running" );
    return;
  }

  /* Destroy the old weight arena and everything in it (old network,
     old optimizer, old layer weights) — then build fresh.             */
  ds_arena_destroy( &app->weight_arena );
  app->weight_arena = ds_arena_new( 0 );

  _network_t *new_net = NULL;
  _optimizer_t *new_opt = NULL;

  if ( !network_config_build( &app->weight_arena, &app->config, &new_net, &new_opt ) )
  {
    SDL_LogError( SDL_LOG_CATEGORY_APPLICATION, "Failed to build network from config" );
    /* The old network lived in the arena that was just destroyed -- clear the
       pointers rather than leaving them dangling. */
    app->network = NULL;
    app->optimizer = NULL;
    app->training_ctx = NULL;
    app->test_result.has_result = false;
    app->config.is_applied = false;
    return;
  }

  app->network = new_net;
  app->optimizer = new_opt;
  app->training_ctx = NULL; /* the old context lived in the destroyed arena */

  /* Reset training metrics — old loss/accuracy history belongs to the
     previous architecture and is meaningless for the new one.         */
  SDL_LockMutex( app->training_state->mutex );
  app->training_state->history_head = 0;
  app->training_state->history_count = 0;
  app->training_state->epoch_loss = 0.0f;
  app->training_state->epoch_acc = 0.0f;
  app->training_state->epoch = 0;
  app->training_state->batch = 0;
  SDL_UnlockMutex( app->training_state->mutex );

  /* Clear any stale test result — it referenced the old network's output */
  app->test_result.has_result = false;

  app->config.is_applied = true;
  app->config.is_dirty = false;

  SDL_Log( "Network rebuilt successfully." );
}

/* ════════════════════════════════════════════════════════════════════════
 * Architecture configuration panel
 * ════════════════════════════════════════════════════════════════════════ */
static void render_config_panel( App *app, float x, float y, float w, float h )
{
  _network_config_t *cfg = &app->config;

  if ( !nk_begin( app->nk, "config", nk_rect( x, y, w, h ), NK_WINDOW_BORDER | NK_WINDOW_NO_SCROLLBAR ) )
  {
    nk_end( app->nk );
    return;
  }

  bool worker_busy = training_is_active( app ) || eval_is_running( app );

  if ( worker_busy )
  {
    nk_layout_row_dynamic( app->nk, 28, 1 );
    nk_label_colored( app->nk, "Stop training and evaluation before rebuilding the network.", NK_TEXT_LEFT, nk_rgb( 220, 60, 60 ) );
  }

  /* ── Fixed input/output dims (display only) ── */
  nk_layout_row_dynamic( app->nk, 24, 1 );
  nk_labelf( app->nk, NK_TEXT_LEFT, "Input dimension: %d (fixed, 28x28 MNIST pixels)", CONFIG_INPUT_DIM );

  /* ── Hidden layers list ── */
  nk_layout_row_dynamic( app->nk, 24, 1 );
  nk_label( app->nk, "Hidden layers:", NK_TEXT_LEFT );

  for ( int i = 0; i < cfg->num_hidden_layers; i++ )
  {
    nk_layout_row_begin( app->nk, NK_DYNAMIC, 32, 5 );

    nk_layout_row_push( app->nk, 0.10f );
    nk_labelf( app->nk, NK_TEXT_LEFT, "L%d", i + 1 );

    /* Dimension slider */
    nk_layout_row_push( app->nk, 0.35f );
    int old_dim = cfg->hidden_layers[i].dimension;
    int new_dim = old_dim;
    nk_slider_int( app->nk, CONFIG_MIN_DIM, &new_dim, CONFIG_MAX_DIM, 1 );
    if ( new_dim != old_dim )
    {
      cfg->hidden_layers[i].dimension = new_dim;
      cfg->is_dirty = true;
    }

    /* Dimension value, editable as a number too */
    nk_layout_row_push( app->nk, 0.12f );
    nk_labelf( app->nk, NK_TEXT_CENTERED, "%d", cfg->hidden_layers[i].dimension );

    /* Activation dropdown */
    nk_layout_row_push( app->nk, 0.28f );
    static const char *activation_names[] = { "ReLU", "Sigmoid" };
    int act_idx = (int)cfg->hidden_layers[i].activation;
    int new_act_idx = nk_combo( app->nk, activation_names, 2, act_idx, 24, nk_vec2( 140, 80 ) );
    if ( new_act_idx != act_idx )
    {
      cfg->hidden_layers[i].activation = (_config_activation_t)new_act_idx;
      cfg->is_dirty = true;
    }

    /* Remove button — disabled if only one layer remains */
    nk_layout_row_push( app->nk, 0.15f );
    if ( cfg->num_hidden_layers > 1 )
    {
      if ( nk_button_label( app->nk, "Remove" ) )
      {
        for ( int j = i; j < cfg->num_hidden_layers - 1; j++ ) cfg->hidden_layers[j] = cfg->hidden_layers[j + 1];
        cfg->num_hidden_layers--;
        cfg->is_dirty = true;
      }
    }
    else { nk_label( app->nk, "", NK_TEXT_LEFT ); /* keep layout aligned */ }

    nk_layout_row_end( app->nk );
  }

  /* ── Add layer button ── */
  nk_layout_row_dynamic( app->nk, 32, 1 );
  if ( cfg->num_hidden_layers < CONFIG_MAX_HIDDEN_LAYERS )
  {
    if ( nk_button_label( app->nk, "+ Add Hidden Layer" ) )
    {
      cfg->hidden_layers[cfg->num_hidden_layers].dimension = 64;
      cfg->hidden_layers[cfg->num_hidden_layers].activation = CONFIG_ACTIVATION_RELU;
      cfg->num_hidden_layers++;
      cfg->is_dirty = true;
    }
  }
  else { nk_label( app->nk, "Maximum hidden layers reached", NK_TEXT_CENTERED ); }

  /* ── Fixed output dim (display only) ── */
  nk_layout_row_dynamic( app->nk, 24, 1 );
  nk_labelf( app->nk, NK_TEXT_LEFT, "Output dimension: %d (fixed, one per digit class)", CONFIG_OUTPUT_DIM );

  /* ── Optimizer selection ── */
  nk_layout_row_dynamic( app->nk, 8, 1 );
  nk_label( app->nk, "", NK_TEXT_LEFT ); /* spacer */

  nk_layout_row_dynamic( app->nk, 24, 1 );
  nk_label( app->nk, "Optimizer:", NK_TEXT_LEFT );

  nk_layout_row_dynamic( app->nk, 32, 2 );
  static const char *optimizer_names[] = { "Adam", "SGD" };
  int opt_idx = (int)cfg->optimizer;
  int new_opt_idx = nk_combo( app->nk, optimizer_names, 2, opt_idx, 24, nk_vec2( 200, 60 ) );
  if ( new_opt_idx != opt_idx )
  {
    cfg->optimizer = (_config_optimizer_t)new_opt_idx;
    cfg->is_dirty = true;
  }

  nk_label( app->nk, "", NK_TEXT_LEFT ); /* alignment spacer */

  /* ── Learning rate slider ── */
  nk_layout_row_dynamic( app->nk, 24, 1 );
  nk_labelf( app->nk, NK_TEXT_LEFT, "Learning rate: %.5f", cfg->learning_rate );

  nk_layout_row_dynamic( app->nk, 24, 1 );
  float old_lr = cfg->learning_rate;
  float new_lr = old_lr;
  /* Slider over a log-ish practical range: 0.00001 .. 0.1 */
  nk_slider_float( app->nk, 0.00001f, &new_lr, 0.1f, 0.00001f );
  if ( new_lr != old_lr )
  {
    cfg->learning_rate = new_lr;
    cfg->is_dirty = true;
  }

  /* ── SGD momentum (only shown when SGD is selected) ── */
  if ( cfg->optimizer == CONFIG_OPTIMIZER_SGD )
  {
    nk_layout_row_dynamic( app->nk, 24, 1 );
    nk_labelf( app->nk, NK_TEXT_LEFT, "SGD momentum: %.3f", cfg->sgd_momentum );

    nk_layout_row_dynamic( app->nk, 24, 1 );
    float old_mom = cfg->sgd_momentum;
    float new_mom = old_mom;
    nk_slider_float( app->nk, 0.0f, &new_mom, 0.999f, 0.001f );
    if ( new_mom != old_mom )
    {
      cfg->sgd_momentum = new_mom;
      cfg->is_dirty = true;
    }
  }

  /* ── Architecture summary ── */
  nk_layout_row_dynamic( app->nk, 8, 1 );
  nk_label( app->nk, "", NK_TEXT_LEFT );

  char summary[256];
  network_config_summary( cfg, summary, sizeof( summary ) );
  nk_layout_row_dynamic( app->nk, 24, 1 );
  nk_labelf( app->nk, NK_TEXT_LEFT, "Architecture: %s", summary );

  /* ── Status + Apply button ── */
  nk_layout_row_dynamic( app->nk, 24, 1 );
  if ( cfg->is_applied && !cfg->is_dirty ) nk_label_colored( app->nk, "Status: Applied and ready to train", NK_TEXT_LEFT, nk_rgb( 60, 160, 60 ) );
  else if ( cfg->is_dirty && cfg->is_applied )
    nk_label_colored( app->nk, "Status: Modified since last Apply -- press Apply to rebuild", NK_TEXT_LEFT, nk_rgb( 220, 150, 40 ) );
  else nk_label_colored( app->nk, "Status: Not applied yet", NK_TEXT_LEFT, nk_rgb( 220, 60, 60 ) );

  nk_layout_row_dynamic( app->nk, 40, 1 );
  if ( nk_button_label( app->nk, "Apply Configuration" ) && !worker_busy ) apply_network_config( app );

  nk_end( app->nk );
}
