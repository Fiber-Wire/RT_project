//
// Created by JCW on 2024/10/20.
//

#ifndef GLOBALS_CUH
#define GLOBALS_CUH
#include "helpers.cuh"
/// Macros
/// sample, x, y
inline auto BLOCKDIMS = dim3(blkx_t, blky_t, blkz_t);
/// x, y
inline auto GRIDDIMS = dim3(width_t/blky_t, height_t/blkz_t);
#include "material.cuh"

/// Synchronization between main and render thread
struct MainRendererComm{
    std::binary_semaphore frame_start_render{0};
    std::binary_semaphore frame_rendered{0};
    std::atomic<bool> stop_render{false};
};
inline MainRendererComm mainRendererComm{};

inline void initialize_main_sync_objs(){
    mainRendererComm.frame_start_render.try_acquire();
    mainRendererComm.frame_rendered.try_acquire();
    mainRendererComm.stop_render.store(false);
}

inline void notify_renderer_exit(){
    mainRendererComm.stop_render.store(true);
}

__device__ inline material** CUDA_MATERIALS;

inline material** HOST_MATERIALS;
#endif //GLOBALS_CUH
