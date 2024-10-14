#ifndef COLOR_H
#define COLOR_H
//==============================================================================================
// Originally written in 2020 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include "interval.h"
#include "vec3.h"

using color = vec3;


inline float linear_to_gamma(float linear_component)
{
    if (linear_component > 0)
        return std::sqrt(linear_component);

    return 0;
}

void color_remap(const color &pixel_color, unsigned int &rbyte, unsigned int &gbyte, unsigned int &bbyte);

void write_color(std::ostream& out, const color& pixel_color) {
    unsigned int rbyte;
    unsigned int gbyte;
    unsigned int bbyte;
    color_remap(pixel_color, rbyte, gbyte, bbyte);

    // Write out the pixel color components.
    out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}

unsigned int pixel_from_color(const color& pixel_color) {
    unsigned int rbyte;
    unsigned int gbyte;
    unsigned int bbyte;
    color_remap(pixel_color, rbyte, gbyte, bbyte);
    return (rbyte<<16)+(gbyte<<8)+(bbyte);
}

void color_remap(const color &pixel_color, unsigned int &rbyte, unsigned int &gbyte, unsigned int &bbyte) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();
    // Translate the [0,1] component values to the byte range [0,255].
    static const interval intensity(0.000, 0.999);
    // Apply a linear to gamma transform for gamma 2
    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);

    rbyte= int(256 * intensity.clamp(r));
    gbyte= int(256 * intensity.clamp(g));
    bbyte= int(256 * intensity.clamp(b));

}


#endif
