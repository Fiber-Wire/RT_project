#ifndef MATERIAL_H
#define MATERIAL_H
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

#include "hittable.h"
#include "texture.h"


class material {
  public:
    virtual ~material() = default;

    virtual color emitted(double u, double v, const point3& p) const {
        return color(0,0,0);
    }

    virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered
    ) const {
        return false;
    }
};


class lambertian : public material {
  public:
    lambertian(const color& albedo) {
        color_tex = solid_color(albedo);
        tex = &color_tex;
    }
    lambertian(texture* tex) : tex(tex) {}

    lambertian(const lambertian& other) {
        *this = other;
    }

    lambertian& operator=(const lambertian& other) {
        if (this != &other) {
            tex = other.tex;
            if (tex == &other.color_tex) {
                color_tex = other.color_tex;
                tex = &color_tex;
            }
        }
        return *this;
    }

    bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
    const override {
        auto scatter_direction = rec.normal + random_unit_vector();

        // Catch degenerate scatter direction
        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;

        scattered = ray(rec.p, scatter_direction);
        attenuation = tex->value(rec.u, rec.v, rec.p);
        return true;
    }

  private:
    texture* tex;
    solid_color color_tex{{0.5,0,0.5}};
};


class metal : public material {
  public:
    metal(const color& albedo, double fuzz) : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {}

    bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
    const override {
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        reflected = unit_vector(reflected) + (fuzz * random_unit_vector());
        scattered = ray(rec.p, reflected);
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }

  private:
    color albedo;
    double fuzz;
};


class dielectric : public material {
  public:
    dielectric(double refraction_index) : refraction_index(refraction_index) {}

    bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
    const override {
        attenuation = color(1.0, 1.0, 1.0);
        double ri = rec.front_face ? (1.0/refraction_index) : refraction_index;

        vec3 unit_direction = unit_vector(r_in.direction());
        double cos_theta = std::fmin(dot(-unit_direction, rec.normal), 1.0);
        double sin_theta = std::sqrt(1.0 - cos_theta*cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, ri) > random_double())
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, ri);

        scattered = ray(rec.p, direction);
        return true;
    }

  private:
    // Refractive index in vacuum or air, or the ratio of the material's refractive index over
    // the refractive index of the enclosing media
    double refraction_index;

    static double reflectance(double cosine, double refraction_index) {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - refraction_index) / (1 + refraction_index);
        r0 = r0*r0;
        return r0 + (1-r0)*std::pow((1 - cosine),5);
    }
};


class diffuse_light : public material {
  public:
    diffuse_light(texture* tex) : tex(tex) {}
    diffuse_light(const color& emit) {
        color_tex = solid_color(emit);
        tex = &color_tex;
    }

    color emitted(double u, double v, const point3& p) const override {
        return tex->value(u, v, p);
    }
    diffuse_light(const diffuse_light& other) {
        *this = other;
    }

    diffuse_light& operator=(const diffuse_light& other) {
        if (this != &other) {
            tex = other.tex;
            if (tex == &other.color_tex) {
                color_tex = other.color_tex;
                tex = &color_tex;
            }
        }
        return *this;
    }

  private:
    texture* tex;
    solid_color color_tex{{0.5,0,0.5}};
};


class isotropic : public material {
  public:
    isotropic(const color& albedo) {
        color_tex = solid_color(albedo);
        tex = &color_tex;
    }
    isotropic(texture* tex) : tex(tex) {}

    bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
    const override {
        scattered = ray(rec.p, random_unit_vector());
        attenuation = tex->value(rec.u, rec.v, rec.p);
        return true;
    }
    isotropic(const isotropic& other) {
        *this = other;
    }

    isotropic& operator=(const isotropic& other) {
        if (this != &other) {
            tex = other.tex;
            if (tex == &other.color_tex) {
                color_tex = other.color_tex;
                tex = &color_tex;
            }
        }
        return *this;
    }

  private:
    texture* tex;
    solid_color color_tex{{0.5,0,0.5}};
};


#endif
