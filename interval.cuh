#ifndef INTERVAL_H
#define INTERVAL_H
#include "helpers.cuh"

class interval {
  public:
    float min, max;

    __host__ __device__ interval() : min(+infinity), max(-infinity) {} // Default interval is empty

    __host__ __device__ interval(const float min, const float max) : min(min), max(max) {}

    __host__ __device__ interval(const interval& a, const interval& b) {
        // Create the interval tightly enclosing the two input intervals.
        min = a.min <= b.min ? a.min : b.min;
        max = a.max >= b.max ? a.max : b.max;
    }

    __host__ __device__ float size() const {
        return max - min;
    }

    __host__ __device__ bool contains(const float x) const {
        return min <= x && x <= max;
    }

    __host__ __device__ bool surrounds(const float x) const {
        return min < x && x < max;
    }

    __host__ __device__ float clamp(const float x) const {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    __host__ __device__ float centre() const {
        return (min+max) / 2;
    }

    __host__ __device__ interval expand(const float delta) const {
        const auto padding = delta/2;
        return interval(min - padding, max + padding);
    }

    __host__ __device__ static interval empty() {
        return {+infinity, -infinity};
    }
    __host__ __device__ static interval universe() {
        return {-infinity, +infinity};
    }
};

__host__ __device__ inline interval operator+(const interval& ival, const float displacement) {
    return {ival.min + displacement, ival.max + displacement};
}

__host__ __device__ inline interval operator+(const float displacement, const interval& ival) {
    return ival + displacement;
}


#endif
