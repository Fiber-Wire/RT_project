#ifndef COLOR_H
#define COLOR_H

#include "interval.cuh"
#include "vec.cuh"
#include <iostream>

using color = vec3;


__host__ __device__ inline float linear_to_gamma(const float linear_component)
{
    if (linear_component > 0)
        return std::sqrt(linear_component);

    return 0;
}

__host__ __device__ inline void color_remap(const color &pixel_color, unsigned int &rbyte, unsigned int &gbyte, unsigned int &bbyte) {
    auto r = pixel_color.x;
    auto g = pixel_color.y;
    auto b = pixel_color.z;
    // Translate the [0,1] component values to the byte range [0,255].
    const interval intensity(0.000f, 0.999f);
    // Apply a linear to gamma transform for gamma 2
    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);

    rbyte= static_cast<unsigned int>(256 * intensity.clamp(r));
    gbyte= static_cast<unsigned int>(256 * intensity.clamp(g));
    bbyte= static_cast<unsigned int>(256 * intensity.clamp(b));

}

inline void write_color(std::ostream& out, const color& pixel_color) {
    unsigned int rbyte;
    unsigned int gbyte;
    unsigned int bbyte;
    color_remap(pixel_color, rbyte, gbyte, bbyte);

    // Write out the pixel color components.
    out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}

__host__ __device__ inline unsigned int pixel_from_color(const color& pixel_color) {
    unsigned int rbyte;
    unsigned int gbyte;
    unsigned int bbyte;
    color_remap(pixel_color, rbyte, gbyte, bbyte);
    return (rbyte<<16)+(gbyte<<8)+(bbyte);
}

#endif
