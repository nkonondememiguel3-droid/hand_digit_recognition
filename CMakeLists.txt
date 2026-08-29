cmake_minimum_required(VERSION 3.20)
project(hand_digit_recognition)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O2")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# add the sublibrary for memory
add_subdirectory(memory)

# add he sublibrary for external dependencies
add_subdirectory(externals)

# add the sdl3 library
find_package(SDL3 REQUIRED CONFIG)

add_executable(
  ${PROJECT_NAME}
  main.c
)
target_link_libraries(
  ${PROJECT_NAME} PUBLIC
  recognition_lib
  ds_arena
  stb_img
  m
  SDL3::SDL3
)
target_include_directories(
  ${PROJECT_NAME} PRIVATE
  externals/nuklear
)

# Main library (tensor + loss, no main())
add_library(
  recognition_lib
  src/tensor.c
  src/dataloader.c
  src/activations.c
  src/network.c
  src/layers.c
  src/loses.c
  src/optimizer.c
  src/training.c
)

target_include_directories(recognition_lib PUBLIC src/include/ src/utility/)
target_link_libraries(
  recognition_lib PUBLIC
  ds_arena
  stb_img
  m
)

# Find Criterion
find_package(PkgConfig REQUIRED)
pkg_check_modules(CRITERION REQUIRED criterion)

if(CRITERION_FOUND)
    enable_testing()

    add_executable(test_dataloader tests/test_dataloader.c)
    target_link_libraries(test_dataloader recognition_lib ${CRITERION_LIBRARIES} m)
    target_include_directories(test_dataloader PRIVATE ${CRITERION_INCLUDE_DIRS})
    add_test(NAME dataloader_tests COMMAND test_dataloader)

    add_executable(test_tensor tests/test_tensor.c)
    target_link_libraries(test_tensor recognition_lib ${CRITERION_LIBRARIES} m)
    target_include_directories(test_tensor PRIVATE ${CRITERION_INCLUDE_DIRS})
    add_test(NAME tensor_tests COMMAND test_tensor)

    add_executable(test_activations tests/test_activations.c)
    target_link_libraries(test_activations recognition_lib ${CRITERION_LIBRARIES} m)
    target_include_directories(test_activations PRIVATE ${CRITERION_INCLUDE_DIRS})
    add_test(NAME activations_tests COMMAND test_activations)

    add_executable(test_layers tests/test_layers.c)
    target_link_libraries(test_layers recognition_lib ${CRITERION_LIBRARIES} m)
    target_include_directories(test_layers PRIVATE ${CRITERION_INCLUDE_DIRS})
    add_test(NAME layers_tests COMMAND test_layers)

    add_executable(test_loses tests/test_loses.c)
    target_link_libraries(test_loses recognition_lib ${CRITERION_LIBRARIES} m)
    target_include_directories(test_loses PRIVATE ${CRITERION_INCLUDE_DIRS})
    add_test(NAME loses_tests COMMAND test_loses)

    add_executable(test_optimizer tests/test_optimizer.c)
    target_link_libraries(test_optimizer recognition_lib ${CRITERION_LIBRARIES} m)
    target_include_directories(test_optimizer PRIVATE ${CRITERION_INCLUDE_DIRS})
    add_test(NAME optimizer_tests COMMAND test_optimizer)

    add_executable(test_network tests/test_network.c)
    target_link_libraries(test_network recognition_lib ${CRITERION_LIBRARIES} m)
    target_include_directories(test_network PRIVATE ${CRITERION_INCLUDE_DIRS})
    add_test(NAME network_tests COMMAND test_network)

endif()
