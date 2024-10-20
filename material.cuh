#ifndef MATERIAL_H
#define MATERIAL_H

#include "hittable.cuh"
#include "texture.cuh"
#include "vec3.cuh"


class material {
  public:
    __host__ __device__ virtual ~material() {}

    __host__ __device__ virtual color emitted(float u, float v, const point3& p) const {
        return color(0,0,0);
    }

    __host__ __device__ virtual void scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* rnd
    ) const {}

    bool will_scatter = false;
};


class lambertian final : public material {
  public:
    __host__ __device__ explicit lambertian(const color& albedo) {
        color_tex = solid_color(albedo);
        tex = &color_tex;
        will_scatter = true;
    }
    __host__ __device__ explicit lambertian(texture* tex) : tex(tex) {
        will_scatter = true;
    }

    __host__ __device__ lambertian(const lambertian& other) {
        *this = other;
    }

    __host__ __device__ lambertian& operator=(const lambertian& other) {
        if (this != &other) {
            tex = other.tex;
            if (tex == &other.color_tex) {
                color_tex = other.color_tex;
                tex = &color_tex;
            }
        }
        return *this;
    }

    __host__ __device__ void scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* rnd)
    const override {
        auto scatter_direction = rec.normal + random_unit_vector(rnd);

        // Catch degenerate scatter direction
        if (near_zero(scatter_direction))
            scatter_direction = rec.normal;

        scattered = ray(rec.p, scatter_direction);
        attenuation = tex->value(rec.u, rec.v, rec.p);
    }

  private:
    texture* tex;
    solid_color color_tex{{0.5f,0.0f,0.5f}};
};


class metal final : public material {
  public:
    __host__ __device__ metal(const color& albedo, const float fuzz) : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {
        will_scatter = true;
    }

    __host__ __device__ void scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* rnd)
    const override {
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        do {
            reflected = unit_vector(reflected) + (fuzz * random_unit_vector(rnd));
        } while (dot(reflected, rec.normal) <= 0);
        scattered = ray(rec.p, reflected);
        attenuation = albedo;
    }

  private:
    color albedo;
    float fuzz;
};


class dielectric final : public material {
  public:
    __host__ __device__ explicit dielectric(const float refraction_index) : refraction_index(refraction_index) {
        will_scatter = true;
    }

    __host__ __device__ void scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* rnd)
    const override {
        attenuation = color(1.0f, 1.0f, 1.0f);
        const float ri = rec.front_face ? (1.0f/refraction_index) : refraction_index;

        const vec3 unit_direction = unit_vector(r_in.direction());
        const float cos_theta = min(dot(-unit_direction, rec.normal), 1.0f);
        const float sin_theta = std::sqrt(1.0f - cos_theta*cos_theta);

        const bool cannot_refract = ri * sin_theta > 1.0f;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, ri) > random_float(rnd))
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, ri);

        scattered = ray(rec.p, direction);
    }

  private:
    // Refractive index in vacuum or air, or the ratio of the material's refractive index over
    // the refractive index of the enclosing media
    float refraction_index;

    __host__ __device__ static float reflectance(const float cosine, const float refraction_index) {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1.0f - refraction_index) / (1.0f + refraction_index);
        r0 = r0*r0;
        return r0 + (1.0f-r0)*std::pow((1.0f - cosine),5);
    }
};


class diffuse_light final : public material {
  public:
    __host__ __device__ explicit diffuse_light(texture* tex) : tex(tex) {}
    __host__ __device__ explicit diffuse_light(const color& emit) {
        color_tex = solid_color(emit);
        tex = &color_tex;
    }

    __host__ __device__ color emitted(const float u, const float v, const point3& p) const override {
        return tex->value(u, v, p);
    }
    __host__ __device__ diffuse_light(const diffuse_light& other) {
        *this = other;
    }

    __host__ __device__ diffuse_light& operator=(const diffuse_light& other) {
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
    solid_color color_tex{{0.5f,0.0f,0.5f}};
};


class isotropic final : public material {
  public:
    __host__ __device__ explicit isotropic(const color& albedo) {
        color_tex = solid_color(albedo);
        tex = &color_tex;
        will_scatter = true;
    }
    __host__ __device__ explicit isotropic(texture* tex) : tex(tex) {
        will_scatter = true;
    }

    __host__ __device__ void scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* rnd)
    const override {
        scattered = ray(rec.p, random_unit_vector(rnd));
        attenuation = tex->value(rec.u, rec.v, rec.p);
    }

    __host__ __device__ isotropic(const isotropic& other) {
        *this = other;
    }

    __host__ __device__ isotropic& operator=(const isotropic& other) {
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
    solid_color color_tex{{0.5f,0.0f,0.5f}};
};


#endif
