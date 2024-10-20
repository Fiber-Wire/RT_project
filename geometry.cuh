//
// Created by JCW on 13/10/2024.
//

#ifndef GEOMETRY_CUH
#define GEOMETRY_CUH
#include "hittable.cuh"
#include "hittable_list.cuh"
#include "aabb.cuh"

class sphere final : public hittable {
  public:
    // Stationary Sphere
    __host__ __device__ sphere(const point3& static_center, const float radius, material* mat)
      : center(static_center, vec3(0,0,0)), radius(max(0.0f,radius)), mat(mat)
    {
        const auto rvec = vec3(radius, radius, radius);
        bbox = aabb(static_center - rvec, static_center + rvec);
    }

    __host__ __device__ bool hit(const ray& r, const interval ray_t, hit_record& rec) const override {
        const point3 current_center = center.origin();
        const vec3 oc = current_center - r.origin();
        const auto a = glm::dot(r.direction(), r.direction());
        const auto h = dot(r.direction(), oc);
        const auto c = glm::dot(oc, oc) - radius*radius;

        const auto discriminant = h*h - a*c;
        if (discriminant < 0)
            return false;

        const auto sqrtd = std::sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        auto root = (h - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root))
                return false;
        }

        rec.t = root;
        const vec3 outward_normal = (r.at(rec.t) - current_center) / radius;
        rec.set_face_normal(r, outward_normal);
        get_sphere_uv(outward_normal, rec.u, rec.v);
        rec.mat = mat;

        return true;
    }

    __host__ __device__ aabb bounding_box() const override { return bbox; }

  private:
    ray center;
    float radius;
    material* mat;
    aabb bbox;

    __host__ __device__ static void get_sphere_uv(const point3& p, float& u, float& v) {
        // p: a given point on the sphere of radius one, centered at the origin.
        // u: returned value [0,1] of angle around the Y axis from X=-1.
        // v: returned value [0,1] of angle from Y=-1 to Y=+1.
        //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
        //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
        //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

        const auto theta = std::acos(-p.y);
        const auto phi = std::atan2(-p.z, p.x) + pi;

        u = phi / (2*pi);
        v = theta / pi;
    }
};
class quad final : public hittable {
  public:
    __host__ __device__ quad(const point3& Q, const vec3& u, const vec3& v, material* mat)
      : Q(Q), u(u), v(v), mat(mat)
    {
        auto n = cross(u, v);
        normal = unit_vector(n);
        D = dot(normal, Q);
        w = n / dot(n,n);

        set_bounding_box();
    }

    __host__ __device__ void set_bounding_box() {
        // Compute the bounding box of all four vertices.
        const auto bbox_diagonal1 = aabb(Q, Q + u + v);
        const auto bbox_diagonal2 = aabb(Q + u, Q + v);
        bbox = aabb(bbox_diagonal1, bbox_diagonal2);
    }

    __host__ __device__ aabb bounding_box() const override { return bbox; }

    __host__ __device__ bool hit(const ray& r, const interval ray_t, hit_record& rec) const override {
        const auto denom = dot(normal, r.direction());

        // No hit if the ray is parallel to the plane.
        if (std::fabs(denom) < 1e-8)
            return false;

        // Return false if the hit point parameter t is outside the ray interval.
        const auto t = (D - dot(normal, r.origin())) / denom;
        if (!ray_t.contains(t))
            return false;

        // Determine if the hit point lies within the planar shape using its plane coordinates.
        const auto intersection = r.at(t);
        const vec3 planar_hitpt_vector = intersection - Q;
        const auto alpha = dot(w, cross(planar_hitpt_vector, v));
        const auto beta = dot(w, cross(u, planar_hitpt_vector));

        if (!is_interior(alpha, beta, rec))
            return false;

        // Ray hits the 2D shape; set the rest of the hit record and return true.
        rec.t = t;
        rec.mat = mat;
        rec.set_face_normal(r, normal);

        return true;
    }

    __host__ __device__ static bool is_interior(float a, float b, hit_record& rec) {
        const auto unit_interval = interval(0, 1);
        // Given the hit point in plane coordinates, return false if it is outside the
        // primitive, otherwise set the hit record UV coordinates and return true.

        if (!unit_interval.contains(a) || !unit_interval.contains(b))
            return false;

        rec.u = a;
        rec.v = b;
        return true;
    }

  private:
    point3 Q;
    vec3 u, v;
    vec3 w;
    material* mat;
    aabb bbox;
    vec3 normal;
    float D;
};


__host__ __device__ inline hittable_list* create_box(const point3& a, const point3& b, material* mat)
{
    // Returns the 3D box (six sides) that contains the two opposite vertices a & b.

    const auto sides = new hittable_list(6);

    // Construct the two opposite vertices with the minimum and maximum coordinates.
    const auto min = point3(std::fmin(a.x,b.x), std::fmin(a.y,b.y), std::fmin(a.z,b.z));
    const auto max = point3(std::fmax(a.x,b.x), std::fmax(a.y,b.y), std::fmax(a.z,b.z));

    const auto dx = vec3(max.x - min.x, 0, 0);
    const auto dy = vec3(0, max.y - min.y, 0);
    const auto dz = vec3(0, 0, max.z - min.z);

    sides->add(new quad(point3(min.x, min.y, max.z),  dx,  dy, mat)); // front
    sides->add(new quad(point3(max.x, min.y, max.z), -dz,  dy, mat)); // right
    sides->add(new quad(point3(max.x, min.y, min.z), -dx,  dy, mat)); // back
    sides->add(new quad(point3(min.x, min.y, min.z),  dz,  dy, mat)); // left
    sides->add(new quad(point3(min.x, max.y, max.z),  dx, -dz, mat)); // top
    sides->add(new quad(point3(min.x, min.y, min.z),  dx,  dz, mat)); // bottom

    return sides;
}
#endif //GEOMETRY_CUH
