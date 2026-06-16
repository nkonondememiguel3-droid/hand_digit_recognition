#include <SDL3/SDL.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

/* Nuklear defines — must match what nuklear_sdl3_renderer.h expects */
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#define NK_INCLUDE_COMMAND_USERDATA     /* mandatory for sdl3_renderer    */
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT /* mandatory for sdl3_renderer    */

/* Use SDL types instead of stdint -- matches the demo's config */
#define NK_INT8 Sint8
#define NK_UINT8 Uint8
#define NK_INT16 Sint16
#define NK_UINT16 Uint16
#define NK_INT32 Sint32
#define NK_UINT32 Uint32
#define NK_SIZE_TYPE uintptr_t
#define NK_POINTER_TYPE uintptr_t
#define NK_BOOL bool

#define NK_ASSERT( c ) SDL_assert( c )
#define NK_STATIC_ASSERT( e ) SDL_COMPILE_TIME_ASSERT(, e )
#define NK_MEMSET( d, c, l ) SDL_memset( d, c, l )
#define NK_MEMCPY( d, s, l ) SDL_memcpy( d, s, l )
#define NK_VSNPRINTF( s, n, f, a ) SDL_vsnprintf( s, n, f, a )
#define NK_STRTOD( s, e ) SDL_strtod( s, e )
#define NK_INV_SQRT( f ) ( 1.0f / SDL_sqrtf( f ) )
#define NK_SIN( f ) SDL_sinf( f )
#define NK_COS( f ) SDL_cosf( f )

/* stb_truetype overrides so it links against SDL instead of libc */
#define STBTT_ifloor( x ) ( (int)SDL_floor( x ) )
#define STBTT_iceil( x ) ( (int)SDL_ceil( x ) )
#define STBTT_sqrt( x ) SDL_sqrt( x )
#define STBTT_pow( x, y ) SDL_pow( x, y )
#define STBTT_fmod( x, y ) SDL_fmod( x, y )
#define STBTT_cos( x ) SDL_cosf( x )
#define STBTT_acos( x ) SDL_acos( x )
#define STBTT_fabs( x ) SDL_fabs( x )
#define STBTT_assert( x ) SDL_assert( x )
#define STBTT_strlen( x ) SDL_strlen( x )
#define STBTT_memcpy SDL_memcpy
#define STBTT_memset SDL_memset
#define stbtt_uint8 Uint8
#define stbtt_int8 Sint8
#define stbtt_uint16 Uint16
#define stbtt_int16 Sint16
#define stbtt_uint32 Uint32
#define stbtt_int32 Sint32
#define STBRP_SORT SDL_qsort
#define STBRP_ASSERT SDL_assert

static char *nk_sdl_dtoa( char *str, double d )
{
  SDL_snprintf( str, 99999, "%.17g", d );
  return str;
}
#define NK_DTOA( s, d ) nk_sdl_dtoa( s, d )

#define NK_IMPLEMENTATION
#include "externals/nuklear/nuklear.h"
#define NK_SDL3_RENDERER_IMPLEMENTATION
#include "externals/nuklear/nuklear_sdl3_renderer.h"

#include "dataloader.h"
#include "ds_arena.h"

/* Constants */
#define WINDOW_W 900
#define WINDOW_H 700
#define IMG_DISPLAY_W 280
#define IMG_DISPLAY_H 280
#define UI_PANEL_H 120

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
  struct nk_context *nk; /* owned by nuklear_sdl3_renderer     */
} App;

/* MNIST → SDL3 texture */
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

  SDL_Texture *tex = SDL_CreateTexture( renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, (int)cols, (int)rows );
  if ( !tex ) return NULL;
  SDL_SetTextureScaleMode( tex, SDL_SCALEMODE_NEAREST );

  void *px;
  int pitch;
  SDL_LockTexture( tex, NULL, &px, &pitch );
  for ( uint32_t row = 0; row < rows; row++ ) SDL_memcpy( (uint8_t *)px + row * pitch, rgba + row * cols * 4, cols * 4 );
  SDL_UnlockTexture( tex );
  return tex;
}

/* Navigation */
static void navigate( App *app, int delta )
{
  uint32_t count = app->dataset->images->header.count;
  app->current_index = ( app->current_index + count + (uint32_t)delta ) % count;

  if ( app->mnist_texture ) SDL_DestroyTexture( app->mnist_texture );
  app->mnist_texture = create_mnist_texture( app->renderer, app->dataset->images, app->current_index );
}

/* Entry point */
int main( void )
{
  if ( !SDL_Init( SDL_INIT_VIDEO ) )
  {
    SDL_LogError( SDL_LOG_CATEGORY_APPLICATION, "SDL init failed: %s", SDL_GetError() );
    return SDL_APP_FAILURE;
  }

  App app = { .is_running = true, .current_index = 0 };

  if ( !SDL_CreateWindowAndRenderer( "MNIST Viewer", WINDOW_W, WINDOW_H, SDL_WINDOW_RESIZABLE, &app.window, &app.renderer ) )
  {
    SDL_LogError( SDL_LOG_CATEGORY_APPLICATION, "Window failed: %s", SDL_GetError() );
    SDL_Quit();
    return SDL_APP_FAILURE;
  }

  /* Dataset */
  app.arena = ds_arena_new( 0 );
  app.dataset = load_dataset( &app.arena, "../datasets/train-images.idx3-ubyte", "../datasets/train-labels.idx1-ubyte" );
  if ( !app.dataset )
  {
    SDL_LogError( SDL_LOG_CATEGORY_APPLICATION, "Failed to load MNIST" );
    return SDL_APP_FAILURE;
  }
  app.mnist_texture = create_mnist_texture( app.renderer, app.dataset->images, 0 );

  /* Nuklear -- pass ctx everywhere, use nk_sdl_allocator() */
  app.nk = nk_sdl_init( app.window, app.renderer, nk_sdl_allocator() );

  {
    struct nk_font_atlas *atlas = nk_sdl_font_stash_begin( app.nk );
    struct nk_font *font = nk_font_atlas_add_default( atlas, 18, NULL );
    nk_sdl_font_stash_end( app.nk );
    nk_style_set_font( app.nk, &font->handle );
  }

  /* Main loop */
  SDL_Event event;
  nk_input_begin( app.nk ); /* prime the first frame                   */

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
          break; /* fast-forward */
        case SDLK_PAGEUP:
          navigate( &app, -100 );
          break; /* fast-back    */
        }
      }

      /* ✅ pass ctx as first arg, event as second */
      SDL_ConvertEventToRenderCoordinates( app.renderer, &event );
      nk_sdl_handle_event( app.nk, &event );
    }
    nk_input_end( app.nk );

    /* Nuklear UI */
    uint32_t label_val = app.dataset->labels->label[app.current_index];

    if ( nk_begin( app.nk, "controls", nk_rect( 0, WINDOW_H - UI_PANEL_H, WINDOW_W, UI_PANEL_H ), NK_WINDOW_NO_SCROLLBAR ) )
    {
      nk_layout_row_dynamic( app.nk, 40, 1 );
      nk_labelf( app.nk, NK_TEXT_CENTERED, "Image %u / %u     Label: %u", app.current_index + 1, app.dataset->images->header.count, label_val );

      nk_layout_row_dynamic( app.nk, 50, 2 );
      if ( nk_button_label( app.nk, "◄  Prev" ) ) navigate( &app, -1 );
      if ( nk_button_label( app.nk, "Next  ►" ) ) navigate( &app, +1 );
    }
    nk_end( app.nk );

    /* Render */
    SDL_SetRenderDrawColor( app.renderer, 30, 30, 30, 255 );
    SDL_RenderClear( app.renderer );

    if ( app.mnist_texture )
    {
      SDL_FRect dst = {
        .x = ( WINDOW_W - IMG_DISPLAY_W ) / 2.0f,
        .y = ( WINDOW_H - UI_PANEL_H - IMG_DISPLAY_H ) / 2.0f,
        .w = IMG_DISPLAY_W,
        .h = IMG_DISPLAY_H,
      };
      SDL_RenderTexture( app.renderer, app.mnist_texture, NULL, &dst );
    }

    /* ✅ pass ctx as first arg */
    nk_sdl_render( app.nk, NK_ANTI_ALIASING_ON );
    nk_sdl_update_TextInput( app.nk );
    SDL_RenderPresent( app.renderer );

    nk_input_begin( app.nk ); /* begin next frame's input               */
  }

  /* Cleanup */
  if ( app.mnist_texture ) SDL_DestroyTexture( app.mnist_texture );
  nk_input_end( app.nk );
  nk_sdl_shutdown( app.nk ); /* ✅ pass ctx */
  SDL_DestroyRenderer( app.renderer );
  SDL_DestroyWindow( app.window );
  ds_arena_destroy( &app.arena );
  SDL_Quit();
  return SDL_APP_SUCCESS;
}
