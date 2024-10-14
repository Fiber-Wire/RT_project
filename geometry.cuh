//
// Created by JCW on 13/10/2024.
//

#ifndef GEOMETRY_CUH
#define GEOMETRY_CUH
#include "hittable.cuh"
#include "hittable_list.cuh"
#include "aabb.cuh"

class sphere : public hittable {
  public:
    // Stationary Sphere
    sphere(const point3& static_center, float radius, material* mat)
      : center(static_center, vec3(0,0,0)), radius(std::fmax(0,radius)), mat(mat)
    {
        auto rvec = vec3(radius, radius, radius);
        bbox = aabb(static_center - rvec, static_center + rvec);
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        point3 current_center = center.origin();
        vec3 oc = current_center - r.origin();
        auto a = glm::dot(r.direction(), r.direction());
        auto h = dot(r.direction(), oc);
        auto c = glm::dot(oc, oc) - radius*radius;

        auto discriminant = h*h - a*c;
        if (discriminant < 0)
            return false;

        auto sqrtd = std::sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        auto root = (h - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root))
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - current_center) / radius;
        rec.set_face_normal(r, outward_normal);
        get_sphere_uv(outward_normal, rec.u, rec.v);
        rec.mat = mat;

        return true;
    }

    aabb bounding_box() const override { return bbox; }

  private:
    ray center;
    float radius;
    material* mat;
    aabb bbox;

    static void get_sphere_uv(const point3& p, float& u, float& v) {
        // p: a given point on the sphere of radius one, centered at the origin.
        // u: returned value [0,1] of angle around the Y axis from X=-1.
        // v: returned value [0,1] of angle from Y=-1 to Y=+1.
        //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
        //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
        //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

        auto theta = std::acos(-p.y);
        auto phi = std::atan2(-p.z, p.x) + pi;

        u = phi / (2*pi);
        v = theta / pi;
    }
};
class quad : public hittable {
  public:
    quad(const point3& Q, const vec3& u, const vec3& v, material* mat)
      : Q(Q), u(u), v(v), mat(mat)
    {
        auto n = cross(u, v);
        normal = unit_vector(n);
        D = dot(normal, Q);
        w = n / dot(n,n);

        set_bounding_box();
    }

    virtual void set_bounding_box() {
        // Compute the bounding box of all four vertices.
        auto bbox_diagonal1 = aabb(Q, Q + u + v);
        auto bbox_diagonal2 = aabb(Q + u, Q + v);
        bbox = aabb(bbox_diagonal1, bbox_diagonal2);
    }

    aabb bounding_box() const override { return bbox; }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        auto denom = dot(normal, r.direction());

        // No hit if the ray is parallel to the plane.
        if (std::fabs(denom) < 1e-8)
            return false;

        // Return false if the hit point parameter t is outside the ray interval.
        auto t = (D - dot(normal, r.origin())) / denom;
        if (!ray_t.contains(t))
            return false;

        // Determine if the hit point lies within the planar shape using its plane coordinates.
        auto intersection = r.at(t);
        vec3 planar_hitpt_vector = intersection - Q;
        auto alpha = dot(w, cross(planar_hitpt_vector, v));
        auto beta = dot(w, cross(u, planar_hitpt_vector));

        if (!is_interior(alpha, beta, rec))
            return false;

        // Ray hits the 2D shape; set the rest of the hit record and return true.
        rec.t = t;
        rec.p = intersection;
        rec.mat = mat;
        rec.set_face_normal(r, normal);

        return true;
    }

    virtual bool is_interior(float a, float b, hit_record& rec) const {
        interval unit_interval = interval(0, 1);
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


inline hittable_list* create_box(const point3& a, const point3& b, material* mat)
{
    // Returns the 3D box (six sides) that contains the two opposite vertices a & b.

    auto sides = new hittable_list();

    // Construct the two opposite vertices with the minimum and maximum coordinates.
    auto min = point3(std::fmin(a.x,b.x), std::fmin(a.y,b.y), std::fmin(a.z,b.z));
    auto max = point3(std::fmax(a.x,b.x), std::fmax(a.y,b.y), std::fmax(a.z,b.z));

    auto dx = vec3(max.x - min.x, 0, 0);
    auto dy = vec3(0, max.y - min.y, 0);
    auto dz = vec3(0, 0, max.z - min.z);

    sides->add(new quad(point3(min.x, min.y, max.z),  dx,  dy, mat)); // front
    sides->add(new quad(point3(max.x, min.y, max.z), -dz,  dy, mat)); // right
    sides->add(new quad(point3(max.x, min.y, min.z), -dx,  dy, mat)); // back
    sides->add(new quad(point3(min.x, min.y, min.z),  dz,  dy, mat)); // left
    sides->add(new quad(point3(min.x, max.y, max.z),  dx, -dz, mat)); // top
    sides->add(new quad(point3(min.x, min.y, min.z),  dx,  dz, mat)); // bottom

    return sides;
}
#endif //GEOMETRY_CUH