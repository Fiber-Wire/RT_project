#ifndef HELPERS_H
#define HELPERS_H

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>


// C++ Std Usings

using std::make_shared;
using std::shared_ptr;


// Constants

constexpr float infinity = std::numeric_limits<float>::infinity();
constexpr float pi = 3.1415926535897932385;


// Utility Functions

__host__ __device__ inline float degrees_to_radians(float degrees) {
    const float pi = 3.1415926535897932385;
    return degrees * pi / 180.0;
}

inline float random_float() {
    static thread_local std::random_device rd{};
    static thread_local std::mt19937 gen{rd()};
    static thread_local std::uniform_real_distribution<float> distr{0,1};
    // Returns a random real in [0,1).
    return distr(gen);
}

inline float random_float(float min, float max) {
    static thread_local std::random_device rd{};
    static thread_local std::mt19937 gen{rd()};
    static thread_local std::uniform_real_distribution<float> distr{min,max};
    // Returns a random real in [min,max).
    return distr(gen);
}

inline int random_int(int min, int max) {
    static thread_local std::random_device rd{};
    static thread_local std::mt19937 gen{rd()};
    static thread_local std::uniform_int_distribution<int> distr{min,max};
    // Returns a random integer in [min,max].
    return distr(gen);
}


#endif
