#ifndef HELPERS_H
#define HELPERS_H

#include <curand_kernel.h>
#include <limits>
#include <random>
#include "thrust/reduce.h"
#include "thrust/unique.h"
#include "thrust/scan.h"
#include "thrust/execution_policy.h"
#include "thrust/sort.h"
#include "cub/cub.cuh"

// Constants

constexpr float infinity = std::numeric_limits<float>::infinity();
constexpr float pi = 3.1415926535897932385;
// Modify the following values to change run config
// Horizontal resolution
constexpr int width_t = 512;
// Vertical resolution
constexpr int height_t = 512;
// Rays sampled per pixel
constexpr int samplePPx_t = 32;
// Length of thread-local ray array
constexpr int numRays_t = 2;
// blockDims.x
constexpr int blkx_t = 2;
// blockDims.y
constexpr int blky_t = 4;
// blockDims.z
constexpr int blkz_t = 4;
// Toggle for ray reordering
constexpr bool use_reordering = true;
// DO NOT MODIFY
constexpr int grdx_t = width_t / blky_t;
constexpr int grdy_t = height_t / blkz_t;


// Utility Functions

__host__ __device__ inline float degrees_to_radians(const float degrees) {
    return degrees * pi / 180.0f;
}

__host__ __device__ inline float random_float(curandState *rnd) {
    // Returns a random real in [0,1).
#ifdef __CUDA_ARCH__
    return curand_uniform(rnd);
#else
    static thread_local std::random_device rd{};
    static thread_local std::mt19937 gen{rd()};
    static thread_local std::uniform_real_distribution<float> distr{0, 1};
    return distr(gen);
#endif
}

__host__ __device__ inline float random_float(float min, float max, curandState *rnd) {
    // Returns a random real in [min,max).
#ifdef __CUDA_ARCH__
    return random_float(rnd)*(max-min)+min;
#else
    static thread_local std::random_device rd{};
    static thread_local std::mt19937 gen{rd()};
    static thread_local std::uniform_real_distribution<float> distr{min, max};

    return distr(gen);
#endif
}

__host__ __device__ inline int random_int(int min, int max, curandState *rnd) {
    // Returns a random integer in [min,max].
#ifdef __CUDA_ARCH__
    return ceilf(curand_uniform(rnd) * (max-min+1)) + min-1;
#else
    static thread_local std::random_device rd{};
    static thread_local std::mt19937 gen{rd()};
    static thread_local std::uniform_int_distribution<int> distr{min, max};

    return distr(gen);
#endif
}


#endif
