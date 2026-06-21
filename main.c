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

/* Constants */
#define WINDOW_W 1100
#define WINDOW_H 720
#define IMG_NATIVE_SIZE 28

#define IMG_PATH "../datasets/train-images.idx3-ubyte"
#define LABEL_PATH "../datasets/train-labels.idx1-ubyte"

/* Whitish theme colors */
#define BG_R 245
#define BG_G 245
#define BG_B 240

/* App state */
typedef struct
{
  SDL_Window *window;
  SDL_Renderer *renderer;
  bool is_running;

  _ds_arena_t_ arena;
  __dataset__ *dataset;
  uint32_t current_index;

  SDL_Texture *mnist_texture;
  struct nk_image nk_mnist_image; /* nuklear-wrapped handle for nk_image() */
  struct nk_context *nk;
} App;

/* MNIST → SDL3 texture (28×28 RGBA, nearest-neighbour scaled by Nuklear later) */
static SDL_Texture *create_mnist_texture( SDL_Renderer *renderer, const __mnist_image__ *images, uint32_t index )
{
  uint32_t rows = images->number_of_rows;
  uint32_t cols = images->number_of_columns;
  uint32_t offset = index * rows * cols;

  uint8_t rgba[28 * 28 * 4];
  for ( uint32_t i = 0; i < rows * cols; i++ )
  {
    uint8_t v = images->pixels[offset + i];
    rgba[i * 4 + 0] = v;
    rgba[i * 4 + 1] = v;
    rgba[i * 4 + 2] = v;
    rgba[i * 4 + 3] = 255;
  }

  SDL_Texture *tex = SDL_CreateTexture( renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STATIC, (int)cols, (int)rows );
  if ( !tex ) return NULL;
  SDL_SetTextureScaleMode( tex, SDL_SCALEMODE_NEAREST ); /* keep pixels crisp on upscale */
  SDL_UpdateTexture( tex, NULL, rgba, (int)cols * 4 );
  return tex;
}

/* Navigation: rebuild texture + nk_image handle for the new index */
static void navigate( App *app, int delta )
{
  uint32_t count = app->dataset->images->header.count;
  app->current_index = ( app->current_index + count + (uint32_t)delta ) % count;

  if ( app->mnist_texture ) SDL_DestroyTexture( app->mnist_texture );
  app->mnist_texture = create_mnist_texture( app->renderer, app->dataset->images, app->current_index );
  app->nk_mnist_image = nk_image_ptr( app->mnist_texture );
}

/* Entry point */
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

  /* Dataset */
  app.arena = ds_arena_new( 0 );
  app.dataset = load_dataset( &app.arena, IMG_PATH, LABEL_PATH );
  if ( !app.dataset )
  {
    SDL_LogError( SDL_LOG_CATEGORY_ERROR, "Failed to load MNIST" );
    return SDL_APP_FAILURE;
  }
  app.mnist_texture = create_mnist_texture( app.renderer, app.dataset->images, 0 );
  app.nk_mnist_image = nk_image_ptr( app.mnist_texture );

  /* Nuklear init */
  app.nk = nk_sdl_init( app.window, app.renderer, nk_sdl_allocator() );

  struct nk_font *mono_font;
  {
    struct nk_font_atlas *atlas = nk_sdl_font_stash_begin( app.nk );
    struct nk_font_config cfg = nk_font_config( 0 );
    mono_font = nk_font_atlas_add_default( atlas, 16, &cfg ); /* used for pixel grid */
    struct nk_font *ui_font = nk_font_atlas_add_default( atlas, 18, NULL );
    nk_sdl_font_stash_end( app.nk );
    nk_style_set_font( app.nk, &ui_font->handle );
  }

  /* Whitish theme: override Nuklear's default dark colors */
  {
    struct nk_color table[NK_COLOR_COUNT];
    nk_style_default( app.nk ); /* reset first */
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
    table[NK_COLOR_CHART_COLOR] = nk_rgb( 120, 150, 200 );
    table[NK_COLOR_CHART_COLOR_HIGHLIGHT] = nk_rgb( 200, 80, 80 );
    table[NK_COLOR_SCROLLBAR] = nk_rgb( 235, 235, 230 );
    table[NK_COLOR_SCROLLBAR_CURSOR] = nk_rgb( 190, 190, 185 );
    table[NK_COLOR_SCROLLBAR_CURSOR_HOVER] = nk_rgb( 170, 170, 165 );
    table[NK_COLOR_SCROLLBAR_CURSOR_ACTIVE] = nk_rgb( 150, 150, 145 );
    table[NK_COLOR_TAB_HEADER] = nk_rgb( 230, 230, 225 );
    nk_style_from_table( app.nk, table );
  }

  /* Main loop */
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

    /* Get current window size for responsive layout */
    int win_w, win_h;
    SDL_GetWindowSize( app.window, &win_w, &win_h );

    uint32_t label_val = app.dataset->labels->label[app.current_index];
    const __mnist_image__ *img = app.dataset->images;
    uint32_t rows = img->number_of_rows, cols = img->number_of_columns;
    uint32_t offset = app.current_index * rows * cols;

    /* Single full-window Nuklear panel containing everything */
    if ( nk_begin( app.nk, "main", nk_rect( 0, 0, (float)win_w, (float)win_h ), NK_WINDOW_NO_SCROLLBAR ) )
    {
      /* Top: navigation + info bar */
      nk_layout_row_dynamic( app.nk, 40, 1 );
      nk_labelf( app.nk, NK_TEXT_CENTERED, "Image %u / %u     Label: %u", app.current_index + 1, img->header.count, label_val );

      nk_layout_row_dynamic( app.nk, 40, 2 );
      if ( nk_button_label( app.nk, "<  Prev" ) ) navigate( &app, -1 );
      if ( nk_button_label( app.nk, "Next  >" ) ) navigate( &app, +1 );

      /* Side-by-side: image (left) | pixel grid (right) — responsive split */
      float content_h = (float)win_h - 140.0f; /* remaining space below the bar */
      if ( content_h < 100.0f ) content_h = 100.0f;

      nk_layout_row_begin( app.nk, NK_DYNAMIC, content_h, 2 );

      // change the background color for the image rendering.
      struct nk_style_item original_bg = app.nk->style.window.fixed_background;
      app.nk->style.window.fixed_background = nk_style_item_color( nk_rgb( 0, 0, 0 ) );

      /* LEFT: image panel -- ratio 0.4 of width */
      nk_layout_row_push( app.nk, 0.4f );
      if ( nk_group_begin( app.nk, "image_panel", NK_WINDOW_BORDER | NK_WINDOW_TITLE ) )
      {
        nk_layout_row_dynamic( app.nk, content_h - 40.0f, 1 );
        /* nk_image scales to fill the widget's allotted space, nearest-neighbour
           because the underlying texture's scale mode is NEAREST */
        nk_image( app.nk, app.nk_mnist_image );
        nk_group_end( app.nk );
      }
      // restore the background color
      app.nk->style.window.fixed_background = original_bg;

      /* RIGHT: pixel value grid -- ratio 0.6 of width, scrollable */
      nk_layout_row_push( app.nk, 0.6f );
      if ( nk_group_begin( app.nk, "pixel_panel", NK_WINDOW_BORDER | NK_WINDOW_TITLE ) )
      {
        nk_style_set_font( app.nk, &mono_font->handle );

        /* One Nuklear row per MNIST row, 28 small labels per row */
        float cell_w = 22.0f; /* fixed-width cell so columns line up like a grid */
        for ( uint32_t r = 0; r < rows; r++ )
        {
          nk_layout_row_static( app.nk, 20, (int)cell_w, (int)cols );
          for ( uint32_t c = 0; c < cols; c++ )
          {
            uint8_t v = img->pixels[offset + r * cols + c];
            char buf[4];
            SDL_snprintf( buf, sizeof( buf ), "%u", v );
            nk_label( app.nk, buf, NK_TEXT_CENTERED );
          }
        }

        nk_style_set_font( app.nk, &mono_font->handle ); /* keep mono inside group */
        nk_group_end( app.nk );
      }

      nk_layout_row_end( app.nk );
    }
    nk_end( app.nk );

    /* Render -- whitish background */
    SDL_SetRenderDrawColor( app.renderer, BG_R, BG_G, BG_B, 255 );
    SDL_RenderClear( app.renderer );

    nk_sdl_render( app.nk, NK_ANTI_ALIASING_ON );
    nk_sdl_update_TextInput( app.nk );
    SDL_RenderPresent( app.renderer );

    nk_input_begin( app.nk );
  }

  /* Cleanup */
  if ( app.mnist_texture ) SDL_DestroyTexture( app.mnist_texture );
  nk_input_end( app.nk );
  nk_sdl_shutdown( app.nk );
  SDL_DestroyRenderer( app.renderer );
  SDL_DestroyWindow( app.window );
  ds_arena_destroy( &app.arena );
  SDL_Quit();
  return SDL_APP_SUCCESS;
}
