#ifndef VEC3_H
#define VEC3_H

#include "glm/glm.hpp"
#include "glm/ext.hpp"
using vec3 = glm::vec3;
// point3 is just an alias for vec3, but useful for geometric clarity in the code.
using point3 = vec3;
#include "helpers.cuh"

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
#ifdef __CUDA_ARCH__
    glm::vec2 res;
    __sincosf(azimuth, &res.x, &res.y);
    return res;
#else
    return {std::sin(azimuth), std::cos(azimuth)};
#endif
}

__host__ __device__ inline vec3 random_unit_vector(curandState* rnd) {
    float z = 2.0f*random_float(rnd) - 1.0f;
    // z is in the range [-1,1]
    const auto planar = random_unit_vector2D(rnd) * sqrtf(1.0f-z*z);
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
#ifdef __CUDA_ARCH__
    const float cos_theta = __saturatef(dot(-uv, n));
#else
    const float cos_theta = min(dot(-uv, n), 1.0f);
#endif
    const vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    const vec3 r_out_parallel = -sqrtf(fabs(1.0f - glm::dot(r_out_perp,r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

/// https://github.com/Forceflow/libmorton/blob/7923faa88d7e564020b2d5d408bf8c186ecbe363/include/libmorton/morton3D.h#L105
__host__ __device__ inline unsigned int spreadBits3D(unsigned int v) {
    v = (v | (v << 16)) & 0x030000FF;  // Mask: 00000011 00000000 00000000 11111111
    v = (v | (v << 8))  & 0x0300F00F;  // Mask: 00000011 00000000 11110000 00001111
    v = (v | (v << 4))  & 0x030C30C3;  // Mask: 00000011 00001100 00110000 11000011
    v = (v | (v << 2))  & 0x09249249;  // Mask: 00001001 00100100 10010010 01001001
    return v;
}

__host__ __device__ inline unsigned int floatMapTo10bits(const float v, const interval size) {
    return static_cast<unsigned int>(std::floorf((v - size.min) / size.size() * 1023.0f));
}

__host__ __device__ inline unsigned int mortonEncode3D(const point3 point, const interval sx, const interval sy, const interval sz) {
    return spreadBits3D(floatMapTo10bits(point.x,sx)) |
        spreadBits3D(floatMapTo10bits(point.y,sy)) << 1 |
        spreadBits3D(floatMapTo10bits(point.z,sz)) << 2;
}


__device__ inline int common_upper_bits(const unsigned long long lhs, const unsigned long long rhs)
{
    return __clzll(lhs ^ rhs);
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
    __host__ __device__ explicit operator vec3 () const {
        glm::vec2 const f = vec_ * 2.0f - 1.0f;
        glm::vec3 tmp = {f.x, f.y, 1.0f - abs(f.x) - abs(f.y)};

        // https://twitter.com/Stubbesaurus/status/937994790553227264
        const float t = max(-tmp.z, 0.0f);
        #ifdef __CUDA_ARCH__
        tmp.x += copysignf(t, -tmp.x);
        tmp.y += copysignf(t, -tmp.y);
        #else
        tmp.x += tmp.x >= 0.0f ? -t : t;
        tmp.y += tmp.y >= 0.0f ? -t : t;
        #endif

        return normalize(tmp);
    }
    __host__ __device__ NormVec3 operator=(const vec3 &n) {
        *this = NormVec3(n);
        return *this;
    }
    __host__ __device__ const glm::vec2& get_compressed() const {
        return vec_;
    }
private:
    /// 0<= vec_.xy <= 1
    glm::vec2 vec_{};

    __host__ __device__ static glm::vec2 OctWrap(const glm::vec2 v) {
        glm::vec2 p;
        #ifdef __CUDA_ARCH__
        p.x = copysignf(1.0f-abs(v.y), v.x);
        p.y = copysignf(1.0f-abs(v.x), v.y);
        #else
        p.x = (1.0f-abs(v.y))*(v.x >= 0.0f ? 1.0f : -1.0f);
        p.y = (1.0f-abs(v.x))*(v.y >= 0.0f ? 1.0f : -1.0f);
        #endif

        return p;
    }
};


#endif
