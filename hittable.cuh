#ifndef HITTABLE_H
#define HITTABLE_H

#include "aabb.cuh"
#include "vec.cuh"
#include "ray.cuh"


class material;


class hit_record {
public:
    float u;
    float v;

    NormVec3 normal;
    float t;
    short mat_id;
    bool front_face;

    __host__ __device__ void set_face_normal(const ray &r, const vec3 &outward_normal) {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to have unit length.

        front_face = dot(vec3(r.direction()), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
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
    __host__ __device__ virtual ~hittable() {
    }

    __host__ __device__ virtual aabb bounding_box() const = 0;
};

__host__ __device__ inline bool get_hit(const ray &r, const interval ray_t, hit_record &rec, const hittable *hit);

class translate final : public hittable {
public:
    __host__ __device__ translate(): object(nullptr), offset(0) {
        type = hit_type::eTranslate;
    }

    __host__ __device__ translate(hittable *object, const vec3 &offset)
        : object(object), offset(offset) {
        type = hit_type::eTranslate;
    }

    __host__ __device__ bool hit(const ray &r, const interval ray_t, hit_record &rec) const {
        // Move the ray backwards by the offset
        const ray offset_r(r.origin() - offset, vec3(r.direction()));

        // Determine whether an intersection exists along the offset ray (and if so, where)
        if (!get_hit(offset_r, ray_t, rec, object))
            return false;

        // No need to move the intersection point forwards by the offset,
        // since the hit-point can be inferred by ray.at(rec.t)

        return true;
    }

    __host__ __device__ aabb bounding_box() const override { return object->bounding_box() + offset; }

private:
    hittable *object;
    vec3 offset;
};


class rotate_y final : public hittable {
public:
    __host__ __device__ rotate_y(): object(nullptr), sin_theta(0), cos_theta(0) {
        type = hit_type::eRotate_y;
    }

    __host__ __device__ rotate_y(hittable *object, const float angle) : object(object) {
        type = hit_type::eRotate_y;
        const auto radians = degrees_to_radians(angle);
        sin_theta = std::sin(radians);
        cos_theta = std::cos(radians);
        bbox = object->bounding_box();

        point3 min(infinity, infinity, infinity);
        point3 max(-infinity, -infinity, -infinity);

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    const auto x = i * bbox.x.max + (1 - i) * bbox.x.min;
                    const auto y = j * bbox.y.max + (1 - j) * bbox.y.min;
                    const auto z = k * bbox.z.max + (1 - k) * bbox.z.min;

                    const auto newx = cos_theta * x + sin_theta * z;
                    const auto newz = -sin_theta * x + cos_theta * z;

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

    __host__ __device__ bool hit(const ray &r, const interval ray_t, hit_record &rec) const {
        // Transform the ray from world space to object space.

        const auto origin = point3(
            (cos_theta * r.origin().x) - (sin_theta * r.origin().z),
            r.origin().y,
            (sin_theta * r.origin().x) + (cos_theta * r.origin().z)
        );
        const auto decomp_normal = vec3(r.direction());
        const auto direction = vec3(
            (cos_theta * decomp_normal.x) - (sin_theta * decomp_normal.z),
            decomp_normal.y,
            (sin_theta * decomp_normal.x) + (cos_theta * decomp_normal.z)
        );

        const ray rotated_r(origin, direction);

        // Determine whether an intersection exists in object space (and if so, where).

        if (!get_hit(rotated_r, ray_t, rec, object))
            return false;

        // Transform the intersection from object space back to world space.
        // No need to worry about hit-point
        const auto decomp_rec_normal = vec3(rec.normal);
        rec.normal = vec3(
            (cos_theta * decomp_rec_normal.x) + (sin_theta * decomp_rec_normal.z),
            decomp_rec_normal.y,
            (-sin_theta * decomp_rec_normal.x) + (cos_theta * decomp_rec_normal.z)
        );

        return true;
    }

    __host__ __device__ aabb bounding_box() const override { return bbox; }

private:
    hittable *object;
    float sin_theta;
    float cos_theta;
    aabb bbox;
};


#endif
