#include "hittable.cuh"

__host__ __device__ void hit_record::set_face_normal(const ray& r, const vec3& outward_normal) {
    // Sets the hit record normal vector.
    // NOTE: the parameter `outward_normal` is assumed to have unit length.

    front_face = dot(r.direction(), outward_normal) < 0;
    normal = front_face ? outward_normal : -outward_normal;
}

// Translate class implementation
__host__ __device__ translate::translate(): object(nullptr), offset(0) {
    type=hit_type::eTranslate;
}

__host__ __device__ translate::translate(hittable* object, const vec3& offset): object(object), offset(offset) {
    type=hit_type::eTranslate;
}

__host__ __device__ bool translate::hit(const ray& r, const interval ray_t, hit_record& rec) const {
    // Move the ray backwards by the offset
    const ray offset_r(r.origin() - offset, r.direction());

    // Determine whether an intersection exists along the offset ray (and if so, where)
    if (!object->hit(offset_r, ray_t, rec))
        return false;

    // No need to move the intersection point forwards by the offset,
    // since the hit-point can be inferred by ray.at(rec.t)
    return true;
}

__host__ __device__ aabb translate::bounding_box() const { return object->bounding_box() + offset; }

// Rotate_y class implementation
__host__ __device__ rotate_y::rotate_y(): object(nullptr), sin_theta(0), cos_theta(0) {
    type = hit_type::eRotate_y;
}

__host__ __device__ rotate_y::rotate_y(hittable* object, const float angle) : object(object) {
    type = hit_type::eRotate_y;
    const auto radians = degrees_to_radians(angle);
    sin_theta = std::sin(radians);
    cos_theta = std::cos(radians);
    bbox = object->bounding_box();

    point3 min( infinity,  infinity,  infinity);
    point3 max(-infinity, -infinity, -infinity);

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

__host__ __device__ bool rotate_y::hit(const ray& r, const interval ray_t, hit_record& rec) const {

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
    // No need to worry about hit-point

    rec.normal = vec3(
        (cos_theta * rec.normal.x) + (sin_theta * rec.normal.z),
        rec.normal.y,
        (-sin_theta * rec.normal.x) + (cos_theta * rec.normal.z)
    );

    return true;
}

__host__ __device__ aabb rotate_y::bounding_box() const { return bbox; }