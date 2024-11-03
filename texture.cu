//
// Created by JCW on 2024/11/2.
//
#include "texture.cuh"

__host__ __device__ texture::~texture() {}

__host__ __device__ color texture::value(float u, float v) const {return {};}

/// Implementation of solid_color

__host__ __device__ solid_color::solid_color(const color& albedo)
: albedo(albedo) {
}

__host__ __device__ solid_color::solid_color(const float red, const float green, const float blue)
: solid_color(color(red,green,blue)) {
}

__host__ __device__ color solid_color::value(float u, float v) const {
    return albedo;
}


/// Implementation of image_texture

__host__ __device__ image_texture::image_texture(const image_record &image_rd)
:image_rd(image_rd) {
}

__host__ __device__ color image_texture::value(float u, float v) const {
    // If we have no texture data, then return solid cyan as a debugging aid.
    if (image_rd.image_height <= 0) return color(0.0f,1.0f,1.0f);

    // Clamp input texture coordinates to [0,1] x [1,0]
    u = interval(0.0f,1.0f).clamp(u);
    v = 1.0f - interval(0.0f,1.0f).clamp(v);  // Flip V to image coordinates

    const auto i = static_cast<int>(u * image_rd.image_width);
    const auto j = static_cast<int>(v * image_rd.image_height);
    const auto pixel = pixel_data(i,j);

    constexpr auto color_scale = 1.0f / 255.0f;
    return color(color_scale*pixel[0], color_scale*pixel[1], color_scale*pixel[2]);
}

__host__ __device__ const unsigned char* image_texture::pixel_data(int x, int y) const {
    // Return the address of the three RGB bytes of the pixel at x,y. If there is no image
    // data, returns magenta.
    static unsigned char magenta[] = { 255, 0, 255 };
    if (image_rd.image_data == nullptr) return magenta;

    x = clamp(x, 0, image_rd.image_width);
    y = clamp(y, 0, image_rd.image_height);

    return image_rd.image_data + y*image_rd.bytes_per_scanline() + x*image_rd.bytes_per_pixel;
}

__host__ __device__ int image_texture::clamp(const int x, const int low, const int high) {
    // Return the value clamped to the range [low, high).
    if (x < low) return low;
    if (x < high) return x;
    return high - 1;
}