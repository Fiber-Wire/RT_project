#ifndef VEC3_H
#define VEC3_H
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
#include "glm/glm.hpp"
#include "glm/ext.hpp"
using vec3 = glm::vec3;
// point3 is just an alias for vec3, but useful for geometric clarity in the code.
using point3 = vec3;


// Vector Utility Functions
inline bool near_zero(const vec3 &v) {
    // Return true if the vector is close to zero in all dimensions.
    auto s = 1e-8;
    return (std::fabs(v.x) < s) && (std::fabs(v.y) < s) && (std::fabs(v.z) < s);
}
inline vec3 random_in_cube(float min, float max) {
    return {random_float(min, max), random_float(min, max), random_float(min, max)};
}

inline vec3 unit_vector(const vec3& v) {
    return glm::normalize(v);
}

inline vec3 random_in_unit_disk() {
    while (true) {
        auto p = vec3(random_float(-1,1), random_float(-1,1), 0);
        if (glm::dot(p,p) < 1)
            return p;
    }
}

inline vec3 random_unit_vector() {
    while (true) {
        auto p = vec3(random_float(-1,1), random_float(-1,1), random_float(-1,1));
        auto lensq = glm::dot(p,p);
        if (1e-160 < lensq && lensq <= 1.0)
            return glm::normalize(p);
    }
}

inline vec3 random_on_hemisphere(const vec3& normal) {
    vec3 on_unit_sphere = random_unit_vector();
    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2*dot(v,n)*n;
}

inline vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) {
    auto cos_theta = std::fmin(dot(-uv, n), 1.0f);
    vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    vec3 r_out_parallel = -std::sqrt(std::fabs(1.0f - glm::dot(r_out_perp,r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}


#endif
