#ifndef VEC3_H
#define VEC3_H

#include "glm/glm.hpp"
#include "glm/ext.hpp"
using vec3 = glm::vec3;
// point3 is just an alias for vec3, but useful for geometric clarity in the code.
using point3 = vec3;


// Vector Utility Functions
__host__ __device__ inline bool near_zero(const vec3 &v) {
    // Return true if the vector is close to zero in all dimensions.
    auto s = 1e-8;
    return (std::fabs(v.x) < s) && (std::fabs(v.y) < s) && (std::fabs(v.z) < s);
}
__host__ __device__ inline vec3 random_in_cube(float min, float max, curandState* rnd) {
    return {random_float(min, max, rnd), random_float(min, max, rnd), random_float(min, max, rnd)};
}

__host__ __device__ inline vec3 unit_vector(const vec3& v) {
    return glm::normalize(v);
}

__host__ __device__ inline vec3 random_in_unit_disk(curandState* rnd) {
    while (true) {
        auto p = vec3(random_float(-1,1, rnd), random_float(-1,1, rnd), 0);
        if (glm::dot(p,p) < 1)
            return p;
    }
}

__host__ __device__ inline vec3 random_unit_vector(curandState* rnd) {
    while (true) {
        auto p = vec3(random_float(-1,1, rnd), random_float(-1,1, rnd), random_float(-1,1, rnd));
        auto lensq = glm::dot(p,p);
        if (1e-160 < lensq && lensq <= 1.0)
            return glm::normalize(p);
    }
}

__host__ __device__ inline vec3 random_on_hemisphere(const vec3& normal, curandState* rnd) {
    vec3 on_unit_sphere = random_unit_vector(rnd);
    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

__host__ __device__ inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2*dot(v,n)*n;
}

__host__ __device__ inline vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) {
    auto cos_theta = std::fmin(dot(-uv, n), 1.0f);
    vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    vec3 r_out_parallel = -std::sqrt(std::fabs(1.0f - glm::dot(r_out_perp,r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}


#endif
