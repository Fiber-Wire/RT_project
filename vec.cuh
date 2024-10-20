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
    constexpr float s = 1e-8f;
    return (std::fabs(v.x) < s) && (std::fabs(v.y) < s) && (std::fabs(v.z) < s);
}
__host__ __device__ inline vec3 random_in_cube(const float min, const float max, curandState* rnd) {
    return {random_float(min, max, rnd), random_float(min, max, rnd), random_float(min, max, rnd)};
}

__host__ __device__ inline vec3 unit_vector(const vec3& v) {
    return glm::normalize(v);
}

__host__ __device__ inline glm::vec2 random_unit_vector2D(curandState* rnd) {
    const float azimuth = random_float(rnd) * 2 * pi;
    return {sin(azimuth), cos(azimuth)};
}

__host__ __device__ inline vec3 random_unit_vector(curandState* rnd) {
    float z = 2.0f*random_float(rnd) - 1.0f;
    // z is in the range [-1,1]
    const auto planar = random_unit_vector2D(rnd) * std::sqrt(1.0f-z*z);
    return {planar.x, planar.y, z};
}

__host__ __device__ inline vec3 random_on_hemisphere(const vec3& normal, curandState* rnd) {
    const vec3 on_unit_sphere = random_unit_vector(rnd);
    if (dot(on_unit_sphere, normal) > 0.0f) {
        // In the same hemisphere as the normal
        return on_unit_sphere;
    }
    else {
        return -on_unit_sphere;
    }
}

__host__ __device__ inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2*dot(v,n)*n;
}

__host__ __device__ inline vec3 refract(const vec3& uv, const vec3& n, const float etai_over_etat) {
    const auto cos_theta = std::fmin(dot(-uv, n), 1.0f);
    const vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    const vec3 r_out_parallel = -std::sqrt(std::fabs(1.0f - glm::dot(r_out_perp,r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

class NormVec3 {
public:
    __host__ __device__ NormVec3() {}
    __host__ __device__ explicit NormVec3(const vec3 n) {
        glm::vec2 p={n.x, n.y};
        p /= abs(n.x) + abs(n.y) + abs(n.z);
        p = n.z >= 0.0f ? p : OctWrap(p);
        p = p * 0.5f + 0.5f;
        vec_ = p;
    }
    __host__ __device__ operator vec3 () const {
        glm::vec3 tmp = {vec_.x, vec_.y, 1.0f - abs(vec_.x) - abs(vec_.y)};

        // https://twitter.com/Stubbesaurus/status/937994790553227264
        const float t = max(-tmp.z, 0.0f);
        tmp.x += tmp.x >= 0.0f ? -t : t;
        tmp.y += tmp.y >= 0.0f ? -t : t;

        return normalize(tmp);
    }
    __host__ __device__ NormVec3 operator=(const vec3 &n) {
        return *this = NormVec3(n);
    }
private:
    glm::vec2 vec_{};

    __host__ __device__ static glm::vec2 OctWrap(const glm::vec2 v) {
        glm::vec2 p;
        p.x = (1.0f-abs(v.y))*(v.x >= 0.0f ? 1.0f : -1.0f);
        p.y = (1.0f-abs(v.x))*(v.y >= 0.0f ? 1.0f : -1.0f);
        return p;
    }
};


#endif
