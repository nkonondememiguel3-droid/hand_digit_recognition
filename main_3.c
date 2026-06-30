#include <SDL3/SDL.h>
#include <SDL3/SDL_log.h>
#include <SDL3/SDL_render.h>
#include <SDL3/SDL_surface.h>
#include <SDL3/SDL_video.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

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
#include "layers.h"
/* #include "loses.h" */
#include "networkd.h"
#include "optimizer.h"
#include "tensor.h"
#include "training.h"

/* ── Constants ───────────────────────────────────────────────────────── */
#define WINDOW_W 1300
#define WINDOW_H 820
#define IMG_NATIVE_SIZE 28

#define TRAIN_IMG_PATH "../datasets/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "../datasets/train-labels.idx1-ubyte"
#define TEST_IMG_PATH "../datasets/t10k-images.idx3-ubyte"
#define TEST_LBL_PATH "../datasets/t10k-labels.idx1-ubyte"

#define BG_R 245
#define BG_G 245
#define BG_B 240

/* ── App state ───────────────────────────────────────────────────────── */
typedef struct
{
  SDL_Window *window;
  SDL_Renderer *renderer;
  bool is_running;

  /* Dataset viewer */
  _ds_arena_t_ arena;
  _ds_arena_t_ scratch_arena;
  __dataset__ *train_ds;
  __dataset__ *test_ds;
  uint32_t current_index;
  _tensor_t *current_image;
  uint32_t current_label;
  SDL_Texture *mnist_texture;
  struct nk_image nk_mnist_image;

  /* Network + training */
  _ds_arena_t_ weight_arena;
  _network_t *network;
  _optimizer_t *optimizer;
  _training_state_t *training_state;
  _training_ctx_t *training_ctx;
  SDL_Thread *training_thread;
  _training_metrics_t metrics; /* snapshot updated every frame */

  /* Nuklear */
  struct nk_context *nk;
  struct nk_font *mono_font;
  struct nk_font *ui_font;
} App;

/* ── Helpers ──────────────────────────────────────────────────────────── */
static void apply_white_theme( struct nk_context *nk );
static bool navigate( App *app, int delta );
static SDL_Texture *create_mnist_texture( SDL_Renderer *renderer, const _tensor_t *image );
static void build_network( App *app );
static void render_viewer_panel( App *app, float x, float y, float w, float h );
static void render_training_panel( App *app, float x, float y, float w, float h );

/* ── Entry point ──────────────────────────────────────────────────────── */
int main( void )
{
  if ( !SDL_Init( SDL_INIT_VIDEO ) )
  {
    SDL_LogError( SDL_LOG_CATEGORY_ERROR, "SDL init failed: %s", SDL_GetError() );
    return SDL_APP_FAILURE;
  }

  App app = { .is_running = true, .current_index = 0 };

  if ( !SDL_CreateWindowAndRenderer( "Hand Written Digit Recognizer (HWDR)", WINDOW_W, WINDOW_H, SDL_WINDOW_RESIZABLE, &app.window, &app.renderer ) )
  {
    SDL_LogError( SDL_LOG_CATEGORY_ERROR, "Window failed: %s", SDL_GetError() );
    SDL_Quit();
    return SDL_APP_FAILURE;
  }

  /* ── Arenas ── */
  app.arena = ds_arena_new( 0 );
  app.scratch_arena = ds_arena_new( 0 );
  app.weight_arena = ds_arena_new( 0 );

  /* ── Datasets ── */
  app.train_ds = dataset_init( &app.arena, TRAIN_IMG_PATH, TRAIN_LBL_PATH );
  app.test_ds = dataset_init( &app.arena, TEST_IMG_PATH, TEST_LBL_PATH );
  if ( !app.train_ds )
  {
    SDL_LogError( SDL_LOG_CATEGORY_ERROR, "Failed to load training dataset" );
    return SDL_APP_FAILURE;
  }

  if ( !navigate( &app, 0 ) )
  {
    SDL_LogError( SDL_LOG_CATEGORY_ERROR, "Failed to load first image" );
    return SDL_APP_FAILURE;
  }

  /* ── Build network + optimizer ── */
  build_network( &app );

  /* ── Training state ── */
  app.training_state = training_state_create( &app.arena );

  /* ── Nuklear ── */
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

  /* ── Main loop ── */
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
          navigate( &app, +1 );
          break;
        case SDLK_LEFT:
          navigate( &app, -1 );
          break;
        case SDLK_PAGEDOWN:
          navigate( &app, +100 );
          break;
        case SDLK_PAGEUP:
          navigate( &app, -100 );
          break;
        }
      }

      SDL_ConvertEventToRenderCoordinates( app.renderer, &event );
      nk_sdl_handle_event( app.nk, &event );
    }
    nk_input_end( app.nk );

    /* Snapshot metrics every frame (short lock) */
    training_read_metrics( app.training_state, &app.metrics );

    /* ── Layout: left = viewer, right = training ── */
    int win_w, win_h;
    SDL_GetWindowSize( app.window, &win_w, &win_h );

    float left_w = (float)win_w * 0.45f;
    float right_w = (float)win_w - left_w;

    render_viewer_panel( &app, 0, 0, left_w, (float)win_h );
    render_training_panel( &app, left_w, 0, right_w, (float)win_h );

    /* ── Render ── */
    SDL_SetRenderDrawColor( app.renderer, BG_R, BG_G, BG_B, 255 );
    SDL_RenderClear( app.renderer );
    nk_sdl_render( app.nk, NK_ANTI_ALIASING_ON );
    nk_sdl_update_TextInput( app.nk );
    SDL_RenderPresent( app.renderer );

    nk_input_begin( app.nk );
  }

  /* ── Cleanup ── */
  if ( app.training_thread ) training_stop( app.training_thread, app.training_state );

  training_state_destroy( app.training_state );

  if ( app.mnist_texture ) SDL_DestroyTexture( app.mnist_texture );
  nk_input_end( app.nk );
  nk_sdl_shutdown( app.nk );
  SDL_DestroyRenderer( app.renderer );
  SDL_DestroyWindow( app.window );

  if ( app.train_ds ) dataset_close( app.train_ds );
  if ( app.test_ds ) dataset_close( app.test_ds );

  ds_arena_destroy( &app.scratch_arena );
  ds_arena_destroy( &app.weight_arena );
  ds_arena_destroy( &app.arena );
  SDL_Quit();
  return SDL_APP_SUCCESS;
}

/* ══════════════════════════════════════════════════════════════════════════
 * build_network
 * ══════════════════════════════════════════════════════════════════════════ */
static void build_network( App *app )
{
  app->network = network_create( &app->weight_arena );
  network_add_layer( &app->weight_arena, app->network, layer_create_dense( &app->weight_arena, 784, 128 ) );
  network_add_layer( &app->weight_arena, app->network, layer_create_sigmoid( &app->weight_arena ) );
  network_add_layer( &app->weight_arena, app->network, layer_create_dense( &app->weight_arena, 128, 64 ) );
  network_add_layer( &app->weight_arena, app->network, layer_create_sigmoid( &app->weight_arena ) );
  network_add_layer( &app->weight_arena, app->network, layer_create_dense( &app->weight_arena, 64, 10 ) );

  app->optimizer = optimizer_create_adam( &app->weight_arena, app->network, optimizer_adam_defaults() );
}

/* ══════════════════════════════════════════════════════════════════════════
 * Viewer panel (left)
 * ══════════════════════════════════════════════════════════════════════════ */
static void render_viewer_panel( App *app, float x, float y, float w, float h )
{
  if ( !nk_begin( app->nk, "viewer", nk_rect( x, y, w, h ), NK_WINDOW_BORDER | NK_WINDOW_NO_SCROLLBAR ) )
  {
    nk_end( app->nk );
    return;
  }

  uint32_t total = app->train_ds->image_header.count;

  nk_layout_row_dynamic( app->nk, 30, 1 );
  nk_labelf( app->nk, NK_TEXT_CENTERED, "Image %u / %u     Label: %u", app->current_index + 1, total, app->current_label );

  nk_layout_row_dynamic( app->nk, 35, 2 );
  if ( nk_button_label( app->nk, "<  Prev" ) ) navigate( app, -1 );
  if ( nk_button_label( app->nk, "Next  >" ) ) navigate( app, +1 );

  float content_h = h - 110.0f;
  if ( content_h < 80.0f ) content_h = 80.0f;

  nk_layout_row_begin( app->nk, NK_DYNAMIC, content_h, 2 );

  /* Image panel — black background */
  struct nk_style_item orig_bg = app->nk->style.window.fixed_background;
  app->nk->style.window.fixed_background = nk_style_item_color( nk_rgb( 0, 0, 0 ) );

  nk_layout_row_push( app->nk, 0.42f );
  if ( nk_group_begin( app->nk, "img_panel", NK_WINDOW_BORDER | NK_WINDOW_TITLE ) )
  {
    nk_layout_row_dynamic( app->nk, content_h - 38.0f, 1 );
    nk_image( app->nk, app->nk_mnist_image );
    nk_group_end( app->nk );
  }
  app->nk->style.window.fixed_background = orig_bg;

  /* Pixel grid panel */
  nk_layout_row_push( app->nk, 0.58f );
  if ( nk_group_begin( app->nk, "pix_panel", NK_WINDOW_BORDER | NK_WINDOW_TITLE ) )
  {
    nk_style_set_font( app->nk, &app->mono_font->handle );
    uint32_t rows = app->train_ds->rows;
    uint32_t cols = app->train_ds->cols;
    const _tensor_t *img = app->current_image;

    for ( uint32_t r = 0; r < rows; r++ )
    {
      nk_layout_row_static( app->nk, 18, 20, (int)cols );
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

/* ══════════════════════════════════════════════════════════════════════════
 * Training panel (right)
 * ══════════════════════════════════════════════════════════════════════════ */
static void render_training_panel( App *app, float x, float y, float w, float h )
{
  _training_metrics_t *m = &app->metrics;

  if ( !nk_begin( app->nk, "training", nk_rect( x, y, w, h ), NK_WINDOW_BORDER | NK_WINDOW_NO_SCROLLBAR ) )
  {
    nk_end( app->nk );
    return;
  }

  /* ── Network info ── */
  nk_layout_row_dynamic( app->nk, 24, 1 );
  nk_label( app->nk, "Network: 784 → 128 → 64 → 10  (Adam lr=1e-3)", NK_TEXT_LEFT );

  /* ── Control buttons ── */
  nk_layout_row_dynamic( app->nk, 38, 3 );

  if ( !m->is_training )
  {
    if ( nk_button_label( app->nk, "Start Training" ) )
    {
      /* Build ctx on the persistent arena so it outlives this frame */
      app->training_ctx = ARENA_NEW( &app->weight_arena, _training_ctx_t );
      app->training_ctx->state = app->training_state;
      app->training_ctx->network = app->network;
      app->training_ctx->optimizer = app->optimizer;
      app->training_ctx->train_ds = app->train_ds;
      app->training_ctx->test_ds = app->test_ds;
      app->training_ctx->config = training_config_defaults();
      app->training_ctx->weight_arena = app->weight_arena;
      app->training_ctx->batch_arena = ds_arena_new( 0 );
      app->training_thread = training_start( app->training_ctx );
    }
  }
  else
  {
    if ( nk_button_label( app->nk, "Stop" ) )
    {
      training_stop( app->training_thread, app->training_state );
      app->training_thread = NULL;
    }

    const char *pause_label = m->is_paused ? "Resume" : "Pause";
    if ( nk_button_label( app->nk, pause_label ) )
    {
      SDL_LockMutex( app->training_state->mutex );
      app->training_state->is_paused = !app->training_state->is_paused;
      SDL_UnlockMutex( app->training_state->mutex );
    }

    nk_label( app->nk, "", NK_TEXT_LEFT ); /* spacer */
  }

  /* ── Progress ── */
  nk_layout_row_dynamic( app->nk, 24, 1 );
  if ( m->is_training )
  {
    nk_labelf( app->nk, NK_TEXT_LEFT, "Epoch %d / %d    Batch %d / %d", m->epoch, m->total_epochs, m->batch, m->total_batches );
    nk_labelf( app->nk, NK_TEXT_LEFT, "Epoch loss: %.4f    Epoch acc: %.2f%%", m->epoch_loss, m->epoch_acc * 100.0f );

    /* Epoch progress bar */
    float epoch_prog = m->total_batches > 0 ? (float)m->batch / (float)m->total_batches : 0.0f;
    nk_layout_row_dynamic( app->nk, 18, 1 );
    nk_progress( app->nk, (nk_size *)&( nk_size ){ (nk_size)( epoch_prog * 100 ) }, 100, NK_FIXED );
  }
  else
  {
    nk_label( app->nk, "Not training", NK_TEXT_LEFT );
    nk_layout_row_dynamic( app->nk, 18, 1 );
    nk_label( app->nk, "", NK_TEXT_LEFT );
  }

  /* ── Loss curve ── */
  float chart_h = ( h - 240.0f ) * 0.5f;
  if ( chart_h < 80.0f ) chart_h = 80.0f;

  nk_layout_row_dynamic( app->nk, 20, 1 );
  nk_label( app->nk, "Loss (per batch)", NK_TEXT_LEFT );

  nk_layout_row_dynamic( app->nk, chart_h, 1 );
  if ( m->history_count > 1 && nk_chart_begin( app->nk, NK_CHART_LINES, m->history_count, 0.0f, 5.0f ) )
  {
    int start = m->history_count >= TRAINING_HISTORY_CAP ? m->history_head : 0;
    int n = m->history_count >= TRAINING_HISTORY_CAP ? TRAINING_HISTORY_CAP : m->history_count;

    for ( int i = 0; i < n; i++ )
    {
      int idx = ( start + i ) % TRAINING_HISTORY_CAP;
      nk_chart_push( app->nk, m->loss_history[idx] );
    }
    nk_chart_end( app->nk );
  }

  /* ── Accuracy curve ── */
  nk_layout_row_dynamic( app->nk, 20, 1 );
  nk_label( app->nk, "Accuracy (per epoch)", NK_TEXT_LEFT );

  nk_layout_row_dynamic( app->nk, chart_h, 1 );
  if ( m->history_count > 1 && nk_chart_begin( app->nk, NK_CHART_LINES, m->history_count, 0.0f, 1.0f ) )
  {
    int start = m->history_count >= TRAINING_HISTORY_CAP ? m->history_head : 0;
    int n = m->history_count >= TRAINING_HISTORY_CAP ? TRAINING_HISTORY_CAP : m->history_count;

    for ( int i = 0; i < n; i++ )
    {
      int idx = ( start + i ) % TRAINING_HISTORY_CAP;
      nk_chart_push( app->nk, m->acc_history[idx] );
    }
    nk_chart_end( app->nk );
  }

  nk_end( app->nk );
}

/* ══════════════════════════════════════════════════════════════════════════
 * Helpers
 * ══════════════════════════════════════════════════════════════════════════ */
static SDL_Texture *create_mnist_texture( SDL_Renderer *renderer, const _tensor_t *image )
{
  uint32_t rows = (uint32_t)image->shape[1];
  uint32_t cols = (uint32_t)image->shape[2];

  uint8_t rgba[IMG_NATIVE_SIZE * IMG_NATIVE_SIZE * 4];
  for ( uint32_t i = 0; i < rows * cols; i++ )
  {
    uint8_t v = (uint8_t)( image->data[i] * 255.0f + 0.5f );
    rgba[i * 4 + 0] = v;
    rgba[i * 4 + 1] = v;
    rgba[i * 4 + 2] = v;
    rgba[i * 4 + 3] = 255;
  }

  SDL_Texture *tex = SDL_CreateTexture( renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STATIC, (int)cols, (int)rows );
  if ( !tex ) return NULL;
  SDL_SetTextureScaleMode( tex, SDL_SCALEMODE_NEAREST );
  SDL_UpdateTexture( tex, NULL, rgba, (int)cols * 4 );
  return tex;
}

static bool navigate( App *app, int delta )
{
  uint32_t count = app->train_ds->image_header.count;
  app->current_index = ( app->current_index + count + (uint32_t)delta ) % count;

  _ds_arena_t_ new_scratch = ds_arena_new( 0 );
  _tensor_t *image = load_image( &new_scratch, app->train_ds, (int)app->current_index );
  _tensor_t *label = load_label( &new_scratch, app->train_ds, (int)app->current_index );
  if ( !image || !label )
  {
    ds_arena_destroy( &new_scratch );
    return false;
  }

  ds_arena_destroy( &app->scratch_arena );
  app->scratch_arena = new_scratch;
  app->current_image = image;
  app->current_label = (uint32_t)label->data[0];

  if ( app->mnist_texture ) SDL_DestroyTexture( app->mnist_texture );
  app->mnist_texture = create_mnist_texture( app->renderer, image );
  app->nk_mnist_image = nk_image_ptr( app->mnist_texture );
  return true;
}

static void apply_white_theme( struct nk_context *nk )
{
  struct nk_color table[NK_COLOR_COUNT];
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
  nk_style_from_table( nk, table );
}
