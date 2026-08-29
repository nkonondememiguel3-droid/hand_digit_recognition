# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A from-scratch neural network for MNIST handwritten digit recognition, written in C11 with no ML
dependencies, plus an SDL3 + Nuklear GUI for browsing the dataset, configuring the architecture,
training live, and testing predictions.

## Build & run

```bash
cmake -S . -B build          # configure (only needed after CMakeLists.txt changes)
cmake --build build -j       # build lib, GUI executable, and all test binaries
cd build && ./hand_digit_recognition
```

The executable **must be run from `build/`**: dataset paths in `main.c` are hardcoded relative
(`../datasets/train-images.idx3-ubyte`, etc.).

### Prerequisites not in the repo

- **SDL3** — found via `find_package(SDL3 REQUIRED CONFIG)`.
- **Criterion** — found via `pkg_check_modules(CRITERION REQUIRED criterion)`; without it the test
  targets are silently skipped.
- **`externals/nuklear/`** — gitignored, not a submodule. Clone manually:
  `git clone https://github.com/Immediate-Mode-UI/Nuklear.git externals/nuklear`
- **`datasets/`** — gitignored. The four MNIST IDX files must exist with these exact names:
  `train-images.idx3-ubyte`, `train-labels.idx1-ubyte`, `t10k-images.idx3-ubyte`,
  `t10k-labels.idx1-ubyte`.

## Tests

```bash
ctest --test-dir build --output-on-failure     # all suites
ctest --test-dir build -R tensor_tests         # one suite (suite names: dataloader_tests,
                                               # tensor_tests, activations_tests, layers_tests,
                                               # loses_tests, optimizer_tests, network_tests)
./build/test_layers --filter 'layers/dense_*'  # single test, via Criterion's own filter
```

Each `tests/test_*.c` is a Criterion suite with an `arena` created in `setup` and destroyed in
`teardown`; tests allocate everything from it. `tests/test_activations.c` is empty (activation
layers are covered by `test_layers.c`).

## Formatting

`.clang-format` (Microsoft-based, with spaces inside parens: `foo( a, b )`). Run
`clang-format -i` on touched files.

## Architecture

Everything below `main.c` builds into `recognition_lib`; `main.c` is the only file with `main()`
that is compiled. **`main_2.c` and `main_3.c` are older UI iterations kept for reference and are
not in the build** — do not edit them when changing the GUI.

### Arena memory model (`memory/ds_arena.*`)

The single memory primitive. A bump allocator over a linked list of malloc'd chunks. Every
allocation is **zeroed**, so structs and tensors don't need explicit init. There is no per-object
free — only `ds_arena_destroy` (frees all chunks) and `ds_arena_checkpoint` /
`ds_arena_reset_to` (rewind to a mark).

Arena lifetime is the central invariant of the whole codebase. There are two roles:

- **persist / weight arena** — holds the network struct, its nodes, layer structs, weight and bias
  tensors, and optimizer state. Must never be reset while the network is in use.
- **batch / scratch arena** — holds per-batch inputs, labels, and every intermediate activation
  and activation-gradient. Checkpointed at the top of each batch and reset only *after* backward
  completes. Layers cache a raw pointer to their input in `layer->cache`, which points into this
  arena — resetting it before `backward()` is a use-after-free.

`training_start()` takes ownership of `ctx->batch_arena`: the training thread destroys it when the
run ends, and `training_start` destroys it itself if the thread fails to spawn. The caller must not
touch that arena after handing it over.

`main.c` keeps four: `arena` (datasets, long-lived UI state), `weight_arena`, `scratch_arena`,
`infer_arena`.

### Tensors (`src/tensor.*`)

`_tensor_t` is a flat row-major float buffer with precomputed `strides[]` and an *optional*
parallel `gradients` buffer (`requires_gradients` — true for parameters, false for activations).
Index through the `T1/T2/T3/T4` macros for data and `G1/G2/G3/G4` for gradients; they use strides,
so never index `->data` with hand-rolled arithmetic.

### Layers (`src/layers.*`)

`_layer_t` is a hand-rolled vtable: `forward`/`backward` function pointers plus a `void *params`
blob for layer-specific config, one `cache` slot for the backward pass, and `weights`/`bias` (NULL
for parameterless layers). Extra parameter tensors go in `extra_params`; anything larger (e.g.
maxpool argmax indices) goes inside `params`.

Shapes are declared with `_layer_shape_t`: `dims[0] == 0` means batch-agnostic, and `ndim == 0`
means fully shape-agnostic (how activation layers pass any shape through). `network_add_layer_checked`
validates a new layer's `in_shape` against the tail's `out_shape`.

`_layer_type_t` enumerates conv2d/maxpool/batchnorm/GAP/flatten/dropout, but **only dense, relu,
and sigmoid are implemented** — the enum is forward-looking. `src/activations.c` and
`activations.h` are empty stubs; activation forward/backward live in `layers.c`.

### Network (`src/network.*`, `src/include/networkd.h`)

A doubly-linked list of `_network_node_t`, with `forward`/`backward` stored as function pointers on
the struct itself (call them as `net->forward( net, arena, input )`). Each node caches its `output`,
which is what makes ResNet-style skip connections work: a node with `skip_source` set adds the
source's cached output into its own output (`network_add_skip`). Backward walks `prev` and folds
skip gradients in on the way.

Skip gradients travel through `node->skip_gradients`: when backward reaches a destination it
accumulates the incoming gradient into its *source* node, which folds it in when the walk gets
there. The field is cleared at the start of every backward pass and lives in the batch arena.

Call `network_zero_gradients()` before every backward pass — parameter gradients accumulate.

### Optimizer (`src/optimizer.*`)

SGD (with momentum) and Adam. State is indexed by a **compacted layer index that counts only layers
having `weights` or `bias`** — the same skip-if-parameterless walk appears in `count_param_layers`,
both create functions, and `optimizer_step`. Every one of them must `continue` past a parameterless
layer *before* touching `state[idx]`; a walk that writes the slot first runs off the end of the
arrays when the network ends in an activation. `tests/test_optimizer.c` guards this.

### Loss (`src/loses.*`)

`loss_softmax_cross_entropy` returns a `_loss_result_t` holding both the scalar loss tensor and the
gradient tensor to feed into `network->backward`.

### Data loading (`src/dataloader.*`)

Streams MNIST IDX files — headers are read and cached at `dataset_init`, then `load_image` /
`load_label` seek and read one item at a time into the arena you pass, so the dataset is never fully
resident. Images come back as `[1, 28, 28]` (channel-first) normalized to `[0, 1]`; the training
loop flattens them into `[batch, 784]`. Big-endian headers go through `SWAP32` in `src/utility/common.h`.

### Training (`src/training.*`)

Runs on an SDL thread. `_training_state_t` is shared with the UI thread and **every field except
`mutex` requires holding `state->mutex`**; the trainer writes, the UI reads via
`training_read_metrics` (short lock, full copy, unlock).

Two threading rules constrain everything here. Each `_layer_t` has a single `cache` slot and each
node a single `output`, so **only one thread may run a forward pass at a time** — `main.c` gates
inference, evaluation, training, and config rebuilds through `training_is_active` /
`eval_is_running` / `network_is_idle`. And `fseek`+`fread` is not atomic on a shared `FILE*`, so
**each thread reading a dataset opens its own handle** (`App.trainer_*_ds` for the trainer, a
locally opened pair inside the eval thread). Loss and accuracy are ring buffers of
`TRAINING_HISTORY_CAP` entries indexed by `history_head`. `should_stop` is the cooperative
cancellation flag.

### Config-driven architecture (`src/include/network_config.h`)

Header-only. `_network_config_t` is what the GUI's Config tab edits (up to
`CONFIG_MAX_HIDDEN_LAYERS` dense+activation pairs, optimizer choice, learning rate); fixed
784 input / 10 output. `network_config_build` assembles the network and optimizer from it.

`apply_network_config()` in `main.c` applies a config by **destroying `weight_arena` wholesale** and
rebuilding — which is why it first checks `is_training` and refuses while the training thread is
alive (that thread holds raw pointers into the arena).

### GUI (`main.c`)

Single-file Nuklear immediate-mode UI on SDL3, four tabs (`_app_tab_t`): Viewer, Training, Testing,
Config. All state lives in one `App` struct; each tab is a `render_*_panel( App*, x, y, w, h )`.
Nuklear and the SDL3 renderer backend are compiled in here via `NK_IMPLEMENTATION` /
`NK_SDL3_RENDERER_IMPLEMENTATION`, so those defines must not appear in another translation unit.

## Naming conventions

Types use leading/surrounding underscores rather than a namespace prefix: `_tensor_t`, `_layer_t`,
`_network_t`, `_ds_arena_t_`, `__dataset__`. Public headers live in `src/include/`; the network
header is `networkd.h` (not `network.h`). Functions are plain `module_verb` (`tensor_create`,
`network_add_layer`, `optimizer_step`).
