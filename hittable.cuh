#ifndef HITTABLE_H
#define HITTABLE_H

#include "aabb.cuh"
#include "vec3.cuh"
#include "ray.cuh"


class material;


class hit_record {
  public:
    point3 p;
    vec3 normal;
    material* mat;
    float t;
    float u;
    float v;
    bool front_face;

    __host__ __device__ void set_face_normal(const ray& r, const vec3& outward_normal) {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to have unit length.

        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};


class hittable {
  public:
    __host__ __device__ virtual ~hittable() {}

    __host__ __device__ virtual bool hit(const ray& r, const interval ray_t, hit_record& rec) const = 0;

    __host__ __device__ virtual aabb bounding_box() const = 0;
};


class translate final : public hittable {
  public:
    __host__ __device__ translate(hittable *object, const vec3& offset)
      : object(object), offset(offset)
    {
        bbox = object->bounding_box() + offset;
    }

    __host__ __device__ bool hit(const ray& r, const interval ray_t, hit_record& rec) const override {
        // Move the ray backwards by the offset
        const ray offset_r(r.origin() - offset, r.direction());

        // Determine whether an intersection exists along the offset ray (and if so, where)
        if (!object->hit(offset_r, ray_t, rec))
            return false;

        // Move the intersection point forwards by the offset
        rec.p += offset;

        return true;
    }

    __host__ __device__ aabb bounding_box() const override { return bbox; }

  private:
    hittable* object;
    vec3 offset;
    aabb bbox;
};


class rotate_y final : public hittable {
  public:
    __host__ __device__ rotate_y(hittable* object, const float angle) : object(object) {
        const auto radians = degrees_to_radians(angle);
        sin_theta = std::sin(radians);
        cos_theta = std::cos(radians);
        bbox = object->bounding_box();

        point3 min( INFINITY,  INFINITY,  INFINITY);
        point3 max(-INFINITY, -INFINITY, -INFINITY);

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    const auto x = i*bbox.x.max + (1-i)*bbox.x.min;
                    const auto y = j*bbox.y.max + (1-j)*bbox.y.min;
                    const auto z = k*bbox.z.max + (1-k)*bbox.z.min;

                    const auto newx =  cos_theta*x + sin_theta*z;
                    const auto newz = -sin_theta*x + cos_theta*z;

                    vec3 tester(newx, y, newz);

                    for (int c = 0; c < 3; c++) {
                        min[c] = std::fmin(min[c], tester[c]);
                        max[c] = std::fmax(max[c], tester[c]);
                    }
                }
            }
        }

        bbox = aabb(min, max);
    }

    __host__ __device__ bool hit(const ray& r, const interval ray_t, hit_record& rec) const override {

        // Transform the ray from world space to object space.

        const auto origin = point3(
            (cos_theta * r.origin().x) - (sin_theta * r.origin().z),
            r.origin().y,
            (sin_theta * r.origin().x) + (cos_theta * r.origin().z)
        );

        const auto direction = vec3(
            (cos_theta * r.direction().x) - (sin_theta * r.direction().z),
            r.direction().y,
            (sin_theta * r.direction().x) + (cos_theta * r.direction().z)
        );

        const ray rotated_r(origin, direction);

        // Determine whether an intersection exists in object space (and if so, where).

        if (!object->hit(rotated_r, ray_t, rec))
            return false;

        // Transform the intersection from object space back to world space.

        rec.p = point3(
            (cos_theta * rec.p.x) + (sin_theta * rec.p.z),
            rec.p.y,
            (-sin_theta * rec.p.x) + (cos_theta * rec.p.z)
        );

        rec.normal = vec3(
            (cos_theta * rec.normal.x) + (sin_theta * rec.normal.z),
            rec.normal.y,
            (-sin_theta * rec.normal.x) + (cos_theta * rec.normal.z)
        );

        return true;
    }

    __host__ __device__ aabb bounding_box() const override { return bbox; }

  private:
    hittable* object;
    float sin_theta;
    float cos_theta;
    aabb bbox;
};


#endif
