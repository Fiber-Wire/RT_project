#ifndef HITTABLE_H
#define HITTABLE_H

#include "aabb.cuh"
#include "vec.cuh"
#include "ray.cuh"


class material;


class hit_record {
  public:
    material* mat;
    float u;
    float v;

    vec3 normal;
    float t;

    bool front_face;

    __host__ __device__ void set_face_normal(const ray& r, const vec3& outward_normal);
};

enum class hit_type {
    eUndefined,
    eBVH,
    eSphere,
    eQuad,
    eTranslate,
    eRotate_y,
    eList
};

class hittable {
  public:
    hit_type type{hit_type::eUndefined};
    __host__ __device__ virtual ~hittable() {}

    __host__ __device__ virtual bool hit(const ray& r, const interval ray_t, hit_record& rec) const = 0;

    __host__ __device__ virtual aabb bounding_box() const = 0;
};


class translate final : public hittable {
  public:
    __host__ __device__ translate();

    __host__ __device__ translate(hittable *object, const vec3& offset);

    __host__ __device__ bool hit(const ray& r, const interval ray_t, hit_record& rec) const override;

    __host__ __device__ aabb bounding_box() const override;

  private:
    hittable* object;
    vec3 offset;
};


class rotate_y final : public hittable {
  public:
    __host__ __device__ rotate_y();
    __host__ __device__ rotate_y(hittable* object, const float angle);

    __host__ __device__ bool hit(const ray& r, const interval ray_t, hit_record& rec) const override;

    __host__ __device__ aabb bounding_box() const override;

  private:
    hittable* object;
    float sin_theta;
    float cos_theta;
    aabb bbox;
};


#endif
