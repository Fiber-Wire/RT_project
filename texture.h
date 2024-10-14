#ifndef TEXTURE_H
#define TEXTURE_H
//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include "rtw_stb_image.h"


class texture {
  public:
    virtual ~texture() = default;

    virtual color value(float u, float v, const point3& p) const = 0;
};


class solid_color : public texture {
  public:
    solid_color(const color& albedo) : albedo(albedo) {}

    solid_color(float red, float green, float blue) : solid_color(color(red,green,blue)) {}

    color value(float u, float v, const point3& p) const override {
        return albedo;
    }

  private:
    color albedo;
};


class checker_texture : public texture {
  public:
    checker_texture(float scale, texture* even, texture* odd)
      : inv_scale(1.0 / scale), even(even), odd(odd) {}

    checker_texture(float scale, const color& c1, const color& c2)
      : inv_scale(1 / scale) {
      color_even = solid_color(c1);
      color_odd = solid_color(c2);
      even = &color_even;
      odd = &color_odd;
    }

    color value(float u, float v, const point3& p) const override {
        auto xInteger = int(std::floor(inv_scale * p.x()));
        auto yInteger = int(std::floor(inv_scale * p.y()));
        auto zInteger = int(std::floor(inv_scale * p.z()));

        bool isEven = (xInteger + yInteger + zInteger) % 2 == 0;

        return isEven ? even->value(u, v, p) : odd->value(u, v, p);
    }

  private:
    float inv_scale;
    texture* even;
    texture* odd;
    solid_color color_even{{0.5,0,0.5}};;
    solid_color color_odd{{0.5,0,0.5}};;
};


class image_texture : public texture {
  public:
    image_texture(const char* filename) : image(filename) {}

    color value(float u, float v, const point3& p) const override {
        // If we have no texture data, then return solid cyan as a debugging aid.
        if (image.height() <= 0) return color(0,1,1);

        // Clamp input texture coordinates to [0,1] x [1,0]
        u = interval(0,1).clamp(u);
        v = 1.0 - interval(0,1).clamp(v);  // Flip V to image coordinates

        auto i = int(u * image.width());
        auto j = int(v * image.height());
        auto pixel = image.pixel_data(i,j);

        auto color_scale = 1.0 / 255.0;
        return color(color_scale*pixel[0], color_scale*pixel[1], color_scale*pixel[2]);
    }

  private:
    rtw_image image;
};


#endif
