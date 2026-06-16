
add_library(
    ds_arena STATIC
    ds_arena.c
    include/ds_arena.h
)

target_include_directories(
    ds_arena PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
