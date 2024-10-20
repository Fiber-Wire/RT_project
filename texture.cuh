#ifndef TEXTURE_H
#define TEXTURE_H

#include "image_loader.cuh"
#include "color.cuh"


class texture {
  public:
    __host__ __device__ virtual ~texture() {}

    __host__ __device__ virtual color value(float u, float v, const point3& p) const = 0;
};


class solid_color : public texture {
  public:
    __host__ __device__ solid_color(const color& albedo) : albedo(albedo) {}

    __host__ __device__ solid_color(float red, float green, float blue) : solid_color(color(red,green,blue)) {}

    __host__ __device__ color value(float u, float v, const point3& p) const override {
        return albedo;
    }

  private:
    color albedo;
};


class checker_texture : public texture {
  public:
    __host__ __device__ checker_texture(float scale, texture* even, texture* odd)
      : inv_scale(1.0f / scale), even(even), odd(odd) {}

    __host__ __device__ checker_texture(float scale, const color& c1, const color& c2)
      : inv_scale(1.0f / scale) {
      color_even = solid_color(c1);
      color_odd = solid_color(c2);
      even = &color_even;
      odd = &color_odd;
    }

    __host__ __device__ color value(float u, float v, const point3& p) const override {
        auto xInteger = int(std::floor(inv_scale * p.x));
        auto yInteger = int(std::floor(inv_scale * p.y));
        auto zInteger = int(std::floor(inv_scale * p.z));

        bool isEven = (xInteger + yInteger + zInteger) % 2 == 0;

        return isEven ? even->value(u, v, p) : odd->value(u, v, p);
    }

  private:
    float inv_scale;
    texture* even;
    texture* odd;
    solid_color color_even{{0.5f,0.0f,0.5f}};;
    solid_color color_odd{{0.0f,0.0f,0.0f}};;
};


class image_texture : public texture {
  public:
    __host__ __device__ explicit image_texture(const image_record &image_rd):image_rd(image_rd) {}

    __host__ __device__ color value(float u, float v, const point3& p) const override {
        // If we have no texture data, then return solid cyan as a debugging aid.
        if (image_rd.image_height <= 0) return color(0.0f,1.0f,1.0f);

        // Clamp input texture coordinates to [0,1] x [1,0]
        u = interval(0.0f,1.0f).clamp(u);
        v = 1.0f - interval(0.0f,1.0f).clamp(v);  // Flip V to image coordinates

        auto i = int(u * image_rd.image_width);
        auto j = int(v * image_rd.image_height);
        auto pixel = pixel_data(i,j);

        auto color_scale = 1.0f / 255.0f;
        return color(color_scale*pixel[0], color_scale*pixel[1], color_scale*pixel[2]);
    }

    __host__ __device__ const unsigned char* pixel_data(int x, int y) const {
        // Return the address of the three RGB bytes of the pixel at x,y. If there is no image
        // data, returns magenta.
        static unsigned char magenta[] = { 255, 0, 255 };
        if (image_rd.image_data == nullptr) return magenta;

        x = clamp(x, 0, image_rd.image_width);
        y = clamp(y, 0, image_rd.image_height);

        return image_rd.image_data + y*image_rd.bytes_per_scanline() + x*image_rd.bytes_per_pixel;
      }

    __host__ __device__ static int clamp(int x, int low, int high) {
        // Return the value clamped to the range [low, high).
        if (x < low) return low;
        if (x < high) return x;
        return high - 1;
    }

  private:
    image_record image_rd{};
};


#endif
