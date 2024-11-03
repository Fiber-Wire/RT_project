#ifndef TEXTURE_H
#define TEXTURE_H

#include "image_loader.cuh"
#include "color.cuh"


class texture {
  public:
    __host__ __device__ virtual ~texture();

    __host__ __device__ virtual color value(float u, float v) const;
};


class solid_color final : public texture {
  public:
    __host__ __device__ explicit solid_color(const color& albedo);

    __host__ __device__ explicit solid_color(float red, float green, float blue);

    __host__ __device__ color value(float u, float v) const override;

  private:
    color albedo{};
};


class image_texture final : public texture {
  public:
    __host__ __device__ explicit image_texture(const image_record &image_rd);

    __host__ __device__ color value(float u, float v) const override;

    __host__ __device__ const unsigned char* pixel_data(int x, int y) const;

    __host__ __device__ static int clamp(int x, int low, int high);

  private:
    image_record image_rd{};
};


#endif
